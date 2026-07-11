import os
import pickle
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from src.rag.reranker import rerank_with_cross_encoder
import re


MODEL_FILENAME_MAP = {
    "cybertruck": "cybertruck.pdf",
    "cyber truck": "cybertruck.pdf",
    "model 3": "model_3.pdf",
    "model3": "model_3.pdf",
    "model s": "model_s.pdf",
    "model x": "model_x.pdf",
    "modelx": "model_x.pdf",
    "model y": "model_y.pdf",
    "modely": "model_y.pdf",
}


def detect_target_models(query: str):
    """
    Kullanıcı sorgusunda geçen TÜM model ifadelerini tespit edip ilgili PDF
    dosya adlarını, tekrar etmeyecek şekilde, bir liste olarak döndürür.
    Hiçbir model ifadesi geçmiyorsa boş liste döner.
    """
    normalized = query.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    matched_files = []
    for phrase, filename in MODEL_FILENAME_MAP.items():
        if phrase in normalized and filename not in matched_files:
            matched_files.append(filename)
    return matched_files


COMPARATIVE_SIGNAL_WORDS = [
    "which model", "which tesla", "which one", "best", "longest", "shortest",
    "fastest", "slowest", "most", "least", "highest", "lowest", "compare",
    "comparison", "versus", " vs ", "better", "cheapest", "most expensive",
]


ALL_MODEL_FILENAMES = [
    "cybertruck.pdf",
    "model_3.pdf",
    "model_s.pdf",
    "model_x.pdf",
    "model_y.pdf",
]


def is_comparative_query(query: str) -> bool:
    """
    Sorgunun modeller arası bir karşılaştırma/üstünlük sorusu olup olmadığını
    tespit eder (örn. "which model has the longest range").
    """
    normalized = f" {query.lower()} "
    return any(signal in normalized for signal in COMPARATIVE_SIGNAL_WORDS)    


def needs_multi_result_coverage(query: str) -> bool:
    """
    Sorgu birden fazla modelin bilgisini gerektiriyor mu? Ya sorguda birden
    fazla model adı açıkça geçiyordur, ya da klasik bir karşılaştırma
    sinyali vardır.
    """
    return len(detect_target_models(query)) >= 2 or is_comparative_query(query)

def setup_retriever():
    persist_directory = os.path.join("data", "processed", "chroma_db")
    chunks_path = os.path.join("data", "processed", "chunks.pkl")

    print("Arama algoritmaları yükleniyor (Chroma + BM25)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)

    
    return {
        "vectorstore": vectorstore,
        "all_chunks": all_chunks,
    }


def build_ensemble_retriever(components, target_filename=None):
    """
    Verilen bileşenlerden (vectorstore + tüm chunk'lar), gerekirse tek bir
    kılavuza (target_filename) filtrelenmiş bir EnsembleRetriever kurar.
    target_filename None ise filtresiz, tüm kılavuzlarda arama yapılır.
    """
    vectorstore = components["vectorstore"]
    all_chunks = components["all_chunks"]

    search_kwargs = {"k": 10}
    if target_filename is not None:
        search_kwargs["filter"] = {"source": target_filename}
    vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if target_filename is not None:
        filtered_chunks = [
            c for c in all_chunks if c.metadata.get("source") == target_filename
        ]
        
        bm25_source_chunks = filtered_chunks if filtered_chunks else all_chunks
    else:
        bm25_source_chunks = all_chunks

    bm25_retriever = BM25Retriever.from_documents(bm25_source_chunks)
    bm25_retriever.k = 10

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

def get_chunks_across_all_models(components, query, per_model_k=6, filenames=None):
    target_filenames = filenames if filenames is not None else ALL_MODEL_FILENAMES
    all_candidates = []
    for filename in target_filenames:
        retriever = build_ensemble_retriever(components, target_filename=filename)
        docs = retriever.invoke(query)
        all_candidates.extend(docs[:per_model_k])
    return all_candidates

class OllamaUnavailableError(Exception):
    """Ollama servisine ulaşılamadığında veya model bulunamadığında fırlatılır.
    UI katmanı (app.py) bu hatayı yakalayıp kullanıcıya anlaşılır bir mesaj
    gösterebilir; ham bir bağlantı traceback'i göstermez."""
    pass


def generate_hypothetical_answer(query, llm):
    
    prompt_template = PromptTemplate.from_template(
        "You are an expert technical writer for Tesla manuals.\n"
        "Write a short, highly technical paragraph that directly answers the user's question.\n"
        "IMPORTANT RULES:\n"
        "- If the user mentions a specific alert code (e.g., CP_a054, APP_w009), YOU MUST include that exact code in your answer.\n"
        "- DO NOT output any conversational text, greetings, or filler words.\n"
        "- Output ONLY the hypothetical manual extract.\n\n"
        "Question: {query}\n\n"
        "Manual Extract:"
    )
    prompt = prompt_template.format(query=query)

    print("\n[HyDE] Sahte cevap üretiliyor...")
    try:
        hypothetical_answer = llm.invoke(prompt)
    except Exception as e:
        
        raise OllamaUnavailableError(
            "Ollama servisine ulaşılamadı. Lütfen Ollama'nın çalıştığından "
            "('ollama serve') ve 'qwen2:1.5b' modelinin indirildiğinden "
            "('ollama pull qwen2:1.5b') emin olun."
        ) from e

    clean_answer = hypothetical_answer.strip()

    print(f"--- Üretilen Sahte Cevap ---\n{clean_answer}\n---------------------------")
    return clean_answer

def get_relevant_chunks(query, components, llm):
    target_filenames = detect_target_models(query)

    if len(target_filenames) == 1:
        target_filename = target_filenames[0]
        print(f"\n[Model Filtresi] Sorguda '{target_filename}' tespit edildi, arama bu kılavuza sınırlandırılıyor.")
        retriever = build_ensemble_retriever(components, target_filename)

        print(f"Orijinal soru ile Karma Arama yapılıyor: '{query}'")
        real_query_docs = retriever.invoke(query)

        hypothetical_answer = generate_hypothetical_answer(query, llm)
        print("Sahte cevap ile Karma Arama yapılıyor...")
        fake_query_docs = retriever.invoke(hypothetical_answer)

        combined_docs = real_query_docs + fake_query_docs

    elif len(target_filenames) >= 2:
        print(f"\n[Çoklu Model Modu] Sorguda birden fazla model tespit edildi: {target_filenames}.")
        print("Sadece bu modellerin kılavuzlarında ayrı ayrı arama yapılıyor...")

        real_query_docs = get_chunks_across_all_models(components, query, filenames=target_filenames)

        hypothetical_answer = generate_hypothetical_answer(query, llm)
        print("Sahte cevap ile bu modellerde ayrı ayrı arama yapılıyor...")
        fake_query_docs = get_chunks_across_all_models(components, hypothetical_answer, filenames=target_filenames)

        combined_docs = real_query_docs + fake_query_docs

    elif is_comparative_query(query):
        print(f"\n[Karşılaştırma Modu] Sorguda belirli bir model yok ama karşılaştırma sinyali tespit edildi.")
        print("Her modelin kılavuzunda ayrı ayrı arama yapılıyor (adil kapsama garantisi)...")

        real_query_docs = get_chunks_across_all_models(components, query)

        hypothetical_answer = generate_hypothetical_answer(query, llm)
        print("Sahte cevap ile her modelde ayrı ayrı arama yapılıyor...")
        fake_query_docs = get_chunks_across_all_models(components, hypothetical_answer)

        combined_docs = real_query_docs + fake_query_docs

    else:
        print("\n[Model Filtresi] Sorguda belirli bir model tespit edilemedi, tüm kılavuzlarda aranıyor.")
        retriever = build_ensemble_retriever(components, target_filename=None)

        print(f"Orijinal soru ile Karma Arama yapılıyor: '{query}'")
        real_query_docs = retriever.invoke(query)

        hypothetical_answer = generate_hypothetical_answer(query, llm)
        print("Sahte cevap ile Karma Arama yapılıyor...")
        fake_query_docs = retriever.invoke(hypothetical_answer)

        combined_docs = real_query_docs + fake_query_docs

    unique_docs = []
    seen_contents = set()

    for doc in combined_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(content_hash)

    print(f"Toplam {len(combined_docs)} sonuç bulundu, {len(unique_docs)} tanesi benzersiz.")
    return unique_docs
            

if __name__ == "__main__":
    
    components = setup_retriever()
    local_llm = OllamaLLM(model="qwen2.5:3b-instruct", temperature=0) 
    
    test_query = "What should I do if alert CP_a054 appears?"
    
    initial_chunks = get_relevant_chunks(test_query, components, local_llm)
    final_top_3_chunks = rerank_with_cross_encoder(query=test_query, chunks=initial_chunks)
    
    print("\n===== NİHAİ 3 CHUNK =====")
    for i, chunk in enumerate(final_top_3_chunks):
        print(f"\n[Sonuç {i+1}] - Kaynak: {chunk.metadata.get('source', 'Bilinmiyor')}")
        print(chunk.page_content)