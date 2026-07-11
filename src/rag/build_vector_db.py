import os
import glob
import pickle
import shutil
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re


FOOTER_LINE_MAX_LEN = 60  


DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d{1,4}")


def is_toc_like_chunk(text, ratio_threshold=0.4, min_dot_leader_lines=2):
    """
    Bir chunk'ın İçindekiler/İndeks sayfası kırıntısı olup olmadığını tespit
    eder. Satırların önemli bir kısmı "kelime....sayfa_no" (dot-leader)
    paterni taşıyorsa True döner; bu tür chunk'lar hiçbir gerçek bilgi
    taşımadığı için vektör veritabanına dahil edilmemelidir.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    dot_leader_line_count = sum(1 for l in lines if DOT_LEADER_RE.search(l))
    ratio = dot_leader_line_count / len(lines)

    return ratio >= ratio_threshold and dot_leader_line_count >= min_dot_leader_lines


def extract_printed_page_number(page_text, num_lines_to_check=5):
    """
    pymupdf4llm'in çıkardığı markdown metninin İÇİNDE yer alan gerçek (basılı)
    sayfa numarasını bulur. Bu numara PDF dosyasının ham sayfa sırasından farklı
    olabilir; çünkü kılavuzlarda kapak/iç kapak/telif gibi numaralandırılmamış
    "ön sayfalar" olabiliyor ve bu da sabit bir kaymaya (örn. +2) yol açıyor.

    Tesla kılavuzlarında footer iki formattan birinde geliyor:
      - Sayı satırın BAŞINDA:  "98 MODEL Y Owner's Manual"
      - Sayı satırın SONUNDA:  "Connectivity 99" / "Towing and Accessories 100"

    Footer genelde sayfanın en son satırlarında yer aldığı için sondan
    `num_lines_to_check` satıra bakılır. Bulunamazsa None döner; bu durumda
    çağıran taraf ham sıra numarasına (page_index + 1) geri düşer.
    """
    lines = [l.strip() for l in page_text.strip().split("\n") if l.strip()]
    if not lines:
        return None

    for line in reversed(lines[-num_lines_to_check:]):
        
        clean_line = line.strip("*# \t")

            
        if not clean_line or len(clean_line) > FOOTER_LINE_MAX_LEN:
            continue

        tokens = clean_line.split()
        if not tokens:
            continue

        
        if tokens[0].isdigit() and 1 <= len(tokens[0]) <= 4:
            return int(tokens[0])

        
        if tokens[-1].isdigit() and 1 <= len(tokens[-1]) <= 4:
            return int(tokens[-1])

    return None


def process_all_pdfs_markdown(raw_dir):
    all_chunks = []
    toc_skipped_count = [0]
    pdf_files = glob.glob(os.path.join(raw_dir, "*.pdf"))
    
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    for pdf in pdf_files:
        print(f"İşleniyor: {pdf} (Markdown formatına dönüştürülüyor)...")
        
        
        md_pages = pymupdf4llm.to_markdown(pdf, page_chunks=True)
        
        fallback_count = 0
        
        for page_index, page_data in enumerate(md_pages):
            page_text = page_data["text"]
            
            
            printed_page = extract_printed_page_number(page_text)
            if printed_page is not None:
                page_num = printed_page
            else:
                page_num = page_index + 1
                fallback_count += 1
            
            
            md_splits = markdown_splitter.split_text(page_text)
            
            
            final_splits = text_splitter.split_documents(md_splits)
            
            MIN_CHUNK_CHARS = 40  

            for split in final_splits:
                cleaned_content = split.page_content.strip()

                
                if len(cleaned_content) < MIN_CHUNK_CHARS:
                    continue

               
                if is_toc_like_chunk(cleaned_content):
                    toc_skipped_count[0] += 1
                    continue

                split.metadata['source'] = os.path.basename(pdf)
                
                split.metadata['page'] = page_num
                split.metadata['page_source'] = 'footer' if printed_page is not None else 'fallback'

                all_chunks.append(split)
        
        if fallback_count > 0:
            print(
                f"  ⚠️  '{os.path.basename(pdf)}' için {fallback_count}/{len(md_pages)} "
                f"sayfada footer'da sayfa numarası bulunamadı, ham sıra numarası kullanıldı. "
                f"Bu sayfaları chunks_readable.txt üzerinden kontrol etmeni öneririm."
            )
            
    print(f"\nToplam {len(pdf_files)} PDF'ten {len(all_chunks)} yapılandırılmış chunk oluşturuldu.")
    if toc_skipped_count[0] > 0:
        print(f"  ℹ️  {toc_skipped_count[0]} adet İçindekiler/İndeks sayfası kırıntısı filtrelendi.")
    return all_chunks

if __name__ == "__main__":
    raw_directory = os.path.join("data", "raw")
    processed_directory = os.path.join("data", "processed")
    persist_directory = os.path.join(processed_directory, "chroma_db")
    
    
    if os.path.exists(processed_directory):
        shutil.rmtree(processed_directory)
    os.makedirs(processed_directory)
    
    chunks = process_all_pdfs_markdown(raw_directory)
    
    if len(chunks) > 0:
        print("\nEmbedding modeli yükleniyor (RTX 3060 CUDA aktif ediliyor)...")
        model_name = "BAAI/bge-small-en-v1.5"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("Vektör veritabanı oluşturuluyor ve diske kaydediliyor...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        chunks_path = os.path.join("data", "processed", "chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        
        txt_path = os.path.join("data", "processed", "chunks_readable.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                source = chunk.metadata.get('source', 'Bilinmiyor')
                page = chunk.metadata.get('page', 'Bilinmiyor')
                f.write(f"===== CHUNK {i+1} | Kaynak: {source} | Sayfa: {page} =====\n")
                f.write(chunk.page_content)
                f.write("\n\n" + "="*50 + "\n\n")
            
        print(f"\n✅ İşlem tamamlandı! Veriler '{persist_directory}' konumuna kaydedildi.")
    else:
        print("Hata: İşlenecek PDF bulunamadı.")