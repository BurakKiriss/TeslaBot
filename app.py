import streamlit as st
from langchain_ollama import OllamaLLM


from src.rag.retriever import setup_retriever, get_relevant_chunks, OllamaUnavailableError, needs_multi_result_coverage
from src.rag.reranker import rerank_with_cross_encoder
from src.generation.generator import generate_answer, CONFIDENCE_GATE_THRESHOLD


@st.cache_resource
def load_database():
    return setup_retriever()


st.set_page_config(page_title="Tesla RAG Arama", page_icon="🚗", layout="centered")
st.title("🚗 Tesla- Akıllı Kılavuz Arama")
st.markdown("Bu sistem, sorunuza en uygun kılavuz bölümlerini getirmek için **HyDE**, **Chroma+BM25 Karma Arama** ve **Cross-Encoder Re-ranking** kullanır.")


retriever_components = load_database()


local_llm = OllamaLLM(model="qwen2.5:3b-instruct", temperature=0) 


query = st.text_input("Kılavuzda ne aramak istiyorsunuz?", placeholder="Örn: What should I do if alert CP_a054 appears?")

if st.button("Ara", type="primary"):
    if not query.strip():
        st.warning("Lütfen bir soru girin.")
    else:
        try:
            
            with st.spinner("1/3: Karma arama yapılıyor ve sahte cevap (HyDE) üretiliyor..."):
                initial_chunks = get_relevant_chunks(query, retriever_components, local_llm)

            
            result_count = 8 if needs_multi_result_coverage(query) else 3

            
            with st.spinner(f"2/3: Cross-Encoder ile en mantıklı {result_count} sonuç seçiliyor (Re-ranking)..."):
                ranked_results = rerank_with_cross_encoder(query=query, chunks=initial_chunks, top_k=result_count)

            
            with st.spinner("3/3: Kılavuz içeriğine dayanarak cevap oluşturuluyor..."):
                answer_text, was_answered = generate_answer(query, ranked_results, local_llm)
        except OllamaUnavailableError as e:
            st.error(f"🔌 {e}")
            st.stop()

        st.success("Arama tamamlandı!")

        
        CONFIDENCE_WARNING_THRESHOLD = CONFIDENCE_GATE_THRESHOLD

        
        st.markdown("### 🤖 Asistan Cevabı")
        if was_answered:
            st.markdown(answer_text)
        else:
            st.info(answer_text)

        st.divider()

        
        st.markdown("### 📌 Cevabın Dayandığı Kılavuz Bölümleri")

        if not ranked_results:
            st.info("Sorgunuza uygun bir sonuç bulunamadı.")
        else:
            best_confidence = ranked_results[0][2]
            if best_confidence < CONFIDENCE_WARNING_THRESHOLD:
                st.warning(
                    f"⚠️ En iyi sonucun güven skoru düşük (%{best_confidence}). "
                    "Sorgunuz kılavuz içeriğiyle tam örtüşmüyor olabilir, sonuçları temkinli değerlendirin."
                )

            for i, (chunk, raw_score, confidence_pct) in enumerate(ranked_results):
                
                file_name = chunk.metadata.get('source', 'Bilinmeyen Kaynak')
                page = chunk.metadata.get('page', 'Bilinmiyor')

                
                label = f"Sonuç {i+1} | 🎯 Güven: %{confidence_pct} | 📄 Sayfa: {page} | 📁 Dosya: {file_name}"
                with st.expander(label, expanded=True):
                    st.write(chunk.page_content)