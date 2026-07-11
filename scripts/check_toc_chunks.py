import pickle
import os
import re

chunks_path = os.path.join("data", "processed", "chunks.pkl")

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

# TOC satırlarının tipik imzası: kelime/başlık, ardından 4+ ardışık nokta
# (bazen aralarda boşluk da olabilir), ardından bir sayfa numarası.
DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d{1,4}")

toc_like_chunks = []

for chunk in chunks:
    text = chunk.page_content
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        continue

    dot_leader_line_count = sum(1 for l in lines if DOT_LEADER_RE.search(l))

    # Chunk'taki satırların önemli bir kısmı (>= %40) dot-leader paterni
    # taşıyorsa, bu büyük ihtimalle bir İçindekiler/İndeks sayfasıdır.
    ratio = dot_leader_line_count / len(lines)
    if ratio >= 0.4 and dot_leader_line_count >= 2:
        toc_like_chunks.append((chunk.metadata.get('source'), chunk.metadata.get('page'), ratio, dot_leader_line_count, len(lines)))

print(f"Toplam chunk: {len(chunks)}")
print(f"TOC/İndeks benzeri chunk sayısı: {len(toc_like_chunks)} (%{len(toc_like_chunks)/len(chunks)*100:.2f})")

print("\n--- Örnekler (kaynak, sayfa, dot-leader oranı, dot-leader satır / toplam satır) ---")
for source, page, ratio, dl_count, total_lines in toc_like_chunks[:15]:
    print(f"[{source} - sayfa {page}] oran: %{ratio*100:.0f} ({dl_count}/{total_lines} satır)")