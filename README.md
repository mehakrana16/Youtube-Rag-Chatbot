# YouTube RAG Chatbot
A local, zero-cost RAG chatbot that answers questions about any YouTube video. 
It fetches the transcript, embeds it with Sentence Transformers, stores it in 
FAISS, and generates grounded answers using a local LLM (Llama 3.2 via Ollama) 
— no paid APIs, runs entirely on your own machine. 

## Features
- Paste any YouTube URL and load its transcript
- Ask questions and get answers based only on the video's content
- See source chunks and timestamps behind each answer
- Runs 100% locally — no API keys, no cost

## Tech Stack
- UI: Streamlit
- Transcript: youtube-transcript-api
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector Store: FAISS
- LLM: Llama 3.2 (3B) via Ollama

## Setup

git clone https://github.com/<your-username>/youtube-rag-chatbot.git
cd youtube-rag-chatbot

python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows

pip install -r requirements.txt
ollama pull llama3.2:3b

streamlit run app.py

Open http://localhost:8501 in your browser.

## Project Structure

youtube-rag-chatbot/
├── app.py
├── requirements.txt
└── src/
    ├── youtube.py
    ├── transcript.py
    ├── text_processing.py
    ├── embeddings.py
    ├── vector_store.py
    └── llm.py
