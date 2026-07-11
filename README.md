# 🚗 TeslaBot

**An offline, privacy-first RAG assistant for Tesla owner's manuals.**

TeslaBot lets you ask natural-language questions about your Tesla vehicle and get accurate, grounded answers — sourced directly from official Tesla owner's manuals (Model 3, Model S, Model X, Model Y, Cybertruck). It is built with a strong emphasis on factual accuracy: the system is explicitly designed to refuse answering when the manuals don't contain a clear answer, rather than guessing or hallucinating.

> 🔒 **100% Local** — No data ever leaves your machine. No cloud APIs, no external inference, no telemetry.
> 💸 **100% Free** — Built entirely with open-source models and tools. Zero API costs, zero subscriptions.

---

## ✨ Key Features

- **Hybrid Retrieval** — Combines dense vector search (Chroma) with sparse keyword search (BM25) for more robust document retrieval.
- **HyDE (Hypothetical Document Embeddings)** — Generates a hypothetical answer to the user's query first, then uses it to improve retrieval quality, especially for short or ambiguous questions.
- **Cross-Encoder Re-ranking** — A dedicated re-ranking model (`BAAI/bge-reranker-base`) re-scores retrieved passages against the original query for higher precision.
- **Confidence Gating** — A calibrated confidence threshold prevents the system from generating an answer when retrieval quality is too low, avoiding hallucinated or misleading responses.
- **Anti-Hallucination Prompting** — The generation prompt is carefully engineered with explicit rules and few-shot examples to keep the model strictly grounded in retrieved context, and to reject questions the manuals don't actually answer.
- **Multi-Model Awareness** — Automatically detects which Tesla model(s) a query refers to and can compare specifications across multiple models when asked.
- **Fully Local Inference** — All models (embedding, re-ranking, and the LLM) run entirely on your own hardware via [Ollama](https://ollama.com) and Hugging Face — no internet connection required after setup.

---

## 🏗️ Architecture
```text
User Query
│
▼
┌─────────────────────────┐
│   Model Detection        │  → Detects which Tesla model(s) the query targets
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│   HyDE Generation         │  → LLM drafts a hypothetical answer to enrich the query
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│   Hybrid Retrieval        │  → Chroma (dense) + BM25 (sparse) ensemble search
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│   Cross-Encoder Rerank    │  → Re-scores candidates for relevance precision
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│   Confidence Gate          │  → Blocks generation if top score is below threshold
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│   Grounded Generation      │  → LLM answers strictly from retrieved context
└─────────────────────────┘
│
▼
Answer + Source Citations
```
---

## 🧠 Tech Stack

| Component | Model / Tool |
|---|---|
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Re-ranking Model | `BAAI/bge-reranker-base` (Cross-Encoder) |
| LLM (Generation & HyDE) | `qwen2.5:3b-instruct` via [Ollama](https://ollama.com) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Keyword Search | BM25 (`langchain_community`) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| PDF Parsing | [PyMuPDF4LLM](https://pymupdf.readthedocs.io/) (Markdown-aware extraction) |
| UI | [Streamlit](https://streamlit.io/) |

All models are open-source and run locally — no OpenAI, Anthropic, or any paid API is used anywhere in this project.

---

## 📁 Project Structure

```text
TeslaBot/
├── data/
│   ├── raw/                    # Source PDFs 
│   └── processed/               # Generated vector DB, chunks, and cache files
├── src/
│   ├── rag/
│   │   ├── build_vector_db.py   # Parses PDFs, chunks text, builds the Chroma vector DB
│   │   ├── retriever.py         # Hybrid (Chroma + BM25) retrieval, HyDE, model detection
│   │   └── reranker.py          # Cross-encoder re-ranking of retrieved chunks
│   └── generation/
│       └── generator.py         # Confidence gating and grounded answer generation
├── app.py                       # Streamlit application entry point
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- (Recommended) An NVIDIA GPU with CUDA support for faster embedding/re-ranking — TeslaBot has been developed and tested on a 6GB VRAM GPU, so it runs comfortably on modest consumer hardware.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/TeslaBot.git
cd TeslaBot
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull the local LLM

```bash
ollama pull qwen2.5:3b-instruct
```


### 4. Build the vector database

```bash
python src/rag/build_vector_db.py
```

This will parse the PDFs, chunk them intelligently by heading structure, generate embeddings, and persist everything to `data/processed/`.

### 5. Launch the app

```bash
streamlit run app.py
```

Open the local URL Streamlit provides (typically `http://localhost:8501`) and start asking questions.

---

## 💬 Example Questions

- "What is the maximum tongue weight for a Model Y trailer?"
- "At what speeds does the Forward Collision Warning operate on a Model X"
- "What is the minimum storage capacity required for a USB drive to record videos in a Model 3?"
- "How much pressure do Model S tires lose in cold ambient temperatures?"
- "What is the maximum vertical load capacity when carrying accessories on a Cybertruck tow hitch?"

If a question can't be answered from the manuals, TeslaBot will honestly say so instead of guessing.

---

## 🔐 Privacy & Cost

- **No cloud calls.** Every model — embedding, re-ranking, and the LLM — runs on your own machine.
- **No API keys required.** There is nothing to sign up for and nothing to pay.
- **No data collection.** Your questions and the manual contents never leave your device.

---
