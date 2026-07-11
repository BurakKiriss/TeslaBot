import math
from sentence_transformers import CrossEncoder


print("[Reranker] BAAI/bge-reranker-base modeli yükleniyor...")
reranker_model = CrossEncoder('BAAI/bge-reranker-base', max_length=512)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def rerank_with_cross_encoder(query: str, chunks: list, top_k: int = 3):
    """
    Cross-Encoder kullanarak query ve chunk'lar arasındaki anlamsal ilişkiyi puanlar
    ve en yüksek skora sahip top_k adet chunk'ı, ham skor ve güven yüzdesiyle
    birlikte (chunk, raw_score, confidence_pct) üçlüleri halinde döndürür.
    """
    print(f"\n[Reranker] Aday {len(chunks)} chunk arasından en mantıklı {top_k} sonuç puanlanarak seçiliyor...")

    if not chunks:
        return []

    
    pairs = [[query, chunk.page_content] for chunk in chunks]

    
    raw_scores = reranker_model.predict(pairs)

    
    scored_chunks = list(zip(chunks, raw_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    
    final_results = []
    print("[Reranker] Seçilen Chunk'ların Skorları:")
    for i, (chunk, raw_score) in enumerate(scored_chunks[:top_k]):
        confidence_pct = round(_sigmoid(float(raw_score)) * 100, 1)
        print(f"  -> {i+1}. Ham Skor: {raw_score:.4f} | Güven: %{confidence_pct}")
        final_results.append((chunk, float(raw_score), confidence_pct))

    return final_results