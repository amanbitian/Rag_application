# 🧠 RAG Corp — Robust PDF & GitHub Retrieval-Augmented Generation System  

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Ollama](https://img.shields.io/badge/Ollama-Local--LLM-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> 🚀 A modular, **production-grade Retrieval-Augmented Generation (RAG)** system with **PDF & GitHub ingestion**, **FAISS vector search**, and **local LLM inference** through Ollama — designed for real-world, offline AI pipelines.

---

## 🏗️ Overview  

**RAG Corp** is a corporate-style RAG pipeline designed to answer questions from **PDF documents** and **GitHub repositories** using local LLMs.  
It’s built for **offline, secure, and configurable deployments**, with robust logging, modular architecture, and full error handling.

---

## ✨ Features  

### 🗂️ Multi-Source Ingestion  
- **PDF Loader** – Extracts text from uploaded PDFs  
- **GitHub Repo Loader** – Clones repos and indexes documentation/code  
- **Configurable via YAML** – Control chunk size, model name, retriever depth, etc.  

### 🧮 Vector Store with FAISS  
- Fast, memory-efficient similarity search  
- Persistent offline index in `data/index/faiss/`  
- Incremental updates supported  

### 🤖 Local LLM + Embeddings  
- **Ollama-powered models** (`llama3.2:1b`, `tinyllama:1.1b`, `deepseek-coder:6.7b`)  
- **Embeddings:** `nomic-embed-text` or fallback to `bge-m3`  
- Fully offline pipeline — no API calls or cloud dependency  

### 🧰 Robust Architecture  
- Centralized logging with `logging_conf.py`  
- Error-handled RAG core (graceful model or data fallback)  
- Modular structure for easy extension (Chroma, Weaviate, etc.)  
- Auto-discovery of configuration paths  

### 💻 Streamlit UI  
- Clean, responsive dashboard  
- Upload PDFs or Git repos directly  
- Real-time logs: document count, index progress, error reports  
- Query chat interface for retrieval-based QA  

---

## 🧱 Project Structure  

Rag_application/
├── app/
│ ├── api/ # (Optional) REST endpoints (FastAPI-ready)
│ ├── configs/ # YAML configs
│ ├── rag-corp/
│ │ ├── rag_core/
│ │ │ ├── embeddings/ # Ollama + HF embedding providers
│ │ │ ├── llm/ # Local LLM interface (Ollama)
│ │ │ ├── loaders/ # GitHub + PDF data ingestion
│ │ │ ├── vectorstores/ # FAISS integration
│ │ │ ├── config.py # Auto-path + .env aware settings loader
│ │ │ ├── logging_conf.py # Central logging system
│ │ │ ├── rag_service.py # RAG logic orchestration
│ │ │ └── utils.py # Helpers / validation
│ └── ui/ # Streamlit front-end
├── configs/settings.yaml # Global configuration
├── data/ # Local index storage
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
├── Makefile
├── pyproject.toml
└── .env.example



---

## ⚙️ Tech Stack  

| Layer | Technology | Purpose |
|:------|:------------|:---------|
| UI | Streamlit | Web-based dashboard |
| Core Framework | LangChain | RAG orchestration |
| LLM | Ollama | Local inference engine |
| Vector Database | FAISS | Embedding similarity search |
| Config | YAML + dotenv | Dynamic environment setup |
| Deployment | Docker + Compose | Reproducible environments |

---

## 🧠 How It Works  

1. **Upload PDF / GitHub URL**  
2. **Text Chunking & Embedding**  
   - Split into overlapping chunks  
   - Generate embeddings (Ollama / HF)  
3. **Vector Indexing**  
   - Stored in FAISS index for fast retrieval  
4. **User Query → Retrieval + LLM Answer**  
   - Relevant chunks retrieved  
   - LLM synthesizes answer with context  

---

## 🧾 Configuration  

All runtime settings are stored in `configs/settings.yaml`:

`yaml
env: dev
data_dir: data
index_dir: data/index/faiss
chunk_size: 1000
chunk_overlap: 200
retriever_k: 4
embed_model: bge-m3
llm_model: llama3.2:1b
git:
  branch: main
  include_exts: [".py", ".md", ".txt"]
  exclude_dirs: ["__pycache__", "tests"]

export CONFIG_PATH=./configs/settings.yaml

---


## Environment setup:

git clone https://github.com/<your-username>/Rag_application.git
cd Rag_application

---

## Create environment
python3 -m venv .venv
source .venv/bin/activate
---

## Install dependencies
pip install -r requirements.txt
---

## Copy env template
cp .env.example .env
---

## Run
streamlit run app/ui/app.py
---

## Ollama services

ollama serve

ollama pull llama3.2:1b
ollama pull tinyllama:1.1b
ollama pull deepseek-coder:6.7b-instruct
ollama pull nomic-embed-text
---

## UI

<img width="1912" height="947" alt="Screenshot 2025-11-04 at 8 09 59 PM" src="https://github.com/user-attachments/assets/c48e177d-efda-4ef2-ad93-3ef07045d3a4" />

---


