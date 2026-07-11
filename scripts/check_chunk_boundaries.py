import pickle
import os

chunks_path = os.path.join("data", "processed", "chunks.pkl")

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

suspicious_start = 0
suspicious_end = 0
examples_start = []
examples_end = []

for chunk in chunks:
    text = chunk.page_content.strip()
    if not text:
        continue

    first_char = text[0]
    last_char = text[-1]

    # Küçük harfle başlıyorsa muhtemelen cümle ortasından kesilmiştir.
    if first_char.islower():
        suspicious_start += 1
        if len(examples_start) < 5:
            examples_start.append((chunk.metadata.get('source'), chunk.metadata.get('page'), text[:80]))

    # Nokta, ünlem, soru işareti, iki nokta, tırnak gibi "bitiş" karakterleriyle
    # bitmiyorsa muhtemelen cümle devam ediyor ama chunk kesilmiş.
    if last_char not in ".!?:\"')”":
        suspicious_end += 1
        if len(examples_end) < 5:
            examples_end.append((chunk.metadata.get('source'), chunk.metadata.get('page'), text[-80:]))

total = len(chunks)
print(f"Toplam chunk: {total}")
print(f"Küçük harfle başlayan (şüpheli): {suspicious_start} (%{suspicious_start/total*100:.1f})")
print(f"Cümle sonu noktalaması olmadan biten (şüpheli): {suspicious_end} (%{suspicious_end/total*100:.1f})")

print("\n--- Küçük harfle başlayan örnekler ---")
for source, page, snippet in examples_start:
    print(f"[{source} - sayfa {page}] \"{snippet}...\"")

print("\n--- Noktalama olmadan biten örnekler ---")
for source, page, snippet in examples_end:
    print(f"[{source} - sayfa {page}] \"...{snippet}\"")