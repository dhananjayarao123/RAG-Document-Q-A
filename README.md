📄 RAG-Document-QA
A lightweight, fully local Vector-Search RAG application — no heavy ML dependencies required.

Upload a PDF, ask questions, and get answers grounded only in your document. Built with Streamlit, Scikit-Learn, ChromaDB, and Groq — skipping bulky embedding models like HuggingFace/Sentence-Transformers (500MB+ with PyTorch).

✨ Features
🔍 Lightweight Embeddings — Uses Scikit-Learn (TF-IDF) instead of Sentence-Transformers/PyTorch
💾 Local Vector Storage — ChromaDB handles persistent vector search, fully offline
⚡ Fast LLM Responses — Powered by Groq's high-speed inference API
🔒 Grounded Answers — Responses generated strictly from your document's context
🖥️ Simple UI — Clean, interactive Streamlit interface
🪶 Minimal Footprint — No GPU, no PyTorch, no model downloads
🏗️ How It Works
text

PDF Upload → Text Extraction → TF-IDF Vectors (Scikit-Learn)
                                        ↓
                              ChromaDB (Vector Store)
                                        ↓
User Question → Similarity Search → Top Chunks → Groq LLM → Answer
🚀 Getting Started
Prerequisites
Python 3.9+
A free Groq API Key
Installation
bash

# Clone the repository
git clone https://github.com/your-username/RAG-Document-QA.git
cd RAG-Document-QA

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Configuration
Create a .env file in the project root:

env

GROQ_API_KEY=your_groq_api_key_here
Run the App
bash

streamlit run app.py
📖 Usage
Upload your PDF via the sidebar
Wait for processing (text extraction + indexing)
Ask questions about your document in the chat box
Get accurate, source-grounded answers instantly
🛠️ Tech Stack
Component
Technology
UI	Streamlit
Embeddings	Scikit-Learn (TF-IDF)
Vector Store	ChromaDB
LLM	Groq API
PDF Parsing	PyPDF2 / pdfplumber

