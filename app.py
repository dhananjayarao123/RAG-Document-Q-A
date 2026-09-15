import streamlit as st
import chromadb

from groq import Groq
from pypdf import PdfReader

from sklearn.feature_extraction.text import HashingVectorizer

from dotenv import load_dotenv

import os
import hashlib


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing")
    st.stop()


# =========================================================
# GROQ
# =========================================================

groq_client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# NOTE: DELETE the "chroma_db" folder manually before running this!
collection = chroma_client.get_or_create_collection(name="documents")


# =========================================================
# PAGE
# =========================================================

st.set_page_config(page_title="Lightweight Document Q&A", page_icon="📚", layout="wide")
st.title("📚 Lightweight Document Q&A")


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf(file):
    reader = PdfReader(file)
    pages = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"page": page_number + 1, "text": text})
    return pages


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []
    for page in pages:
        text = page["text"]
        page_number = page["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({"text": chunk, "page": page_number})
            start += chunk_size - overlap
    return chunks


# =========================================================
# DOCUMENT HASH
# =========================================================

def get_document_id(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()


# =========================================================
# STORE IN CHROMADB (FIXED VARIABLE NAMING)
# =========================================================

def store_embeddings(chunks, filename, document_id):
    texts = [chunk["text"] for chunk in chunks]
    
    # Keep the embedding width fixed for the persisted ChromaDB collection.
    vectorizer = HashingVectorizer(
        n_features=1024,
        alternate_sign=False,
        norm="l2"
    )
    embedding_matrix = vectorizer.transform(texts)
    
    documents = []
    embedding_list = [] # RENAMED list
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):
        documents.append(chunk["text"])
        
        # Convert the sparse matrix row to a list and add to our list
        embedding_list.append(embedding_matrix[index].toarray()[0].tolist())
        
        metadatas.append({
            "filename": filename,
            "page": chunk["page"],
            "document_id": document_id
        })
        ids.append(f"{document_id}_{index}")

    collection.add(
        documents=documents,
        embeddings=embedding_list,
        metadatas=metadatas,
        ids=ids
    )
    
    return vectorizer


def search_documents(question, vectorizer, document_id, top_k=5):
    query_embedding = vectorizer.transform([question]).toarray()[0].tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_id": document_id}
    )


# =========================================================
# GROQ
# =========================================================

def ask_groq(question, context):
    prompt = f"""
You are a document question-answering assistant.
Answer the question ONLY using the provided context.
Do not use outside knowledge.
If the answer is not present in the context, say:
"I couldn't find the answer in the uploaded document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Answer only from the provided document."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

# =========================================================
# UPLOAD & PROCESS
# =========================================================

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    document_id = get_document_id(uploaded_file)

    # Check if old broken data exists
    existing = collection.get(where={"document_id": document_id})

    if existing["ids"]:
        st.info("This document is already stored in ChromaDB (possibly with old, broken vectors).")
        # NEW: Give the user a button to delete the old data and reprocess it
        if st.button("Force Reprocess Document", type="primary"):
            collection.delete(where={"document_id": document_id})
            st.rerun() # This reloads the app and triggers the processing step below
    else:
        if st.button("Process Document"):
            with st.spinner("Extracting PDF..."):
                pages = extract_pdf(uploaded_file)

            st.info(f"Extracted {len(pages)} pages")

            chunks = create_chunks(pages)
            st.info(f"Created {len(chunks)} chunks")

            with st.spinner("Creating lightweight TF-IDF vectors..."):
                fitted_vectorizer = store_embeddings(chunks, uploaded_file.name, document_id)
                st.session_state["vectorizer"] = fitted_vectorizer

            st.success("Document stored in ChromaDB!")

# =========================================================
# QUESTION
# =========================================================

st.divider()
st.subheader("Ask a question")

question = st.text_input("Question")

if st.button("Ask"):
    if not uploaded_file:
        st.warning("Upload a document first.")
    elif not question:
        st.warning("Enter a question.")
    elif "vectorizer" not in st.session_state:
        st.warning("Please click 'Process Document' first.")
    else:
        document_id = get_document_id(uploaded_file)
        vectorizer = st.session_state["vectorizer"]

        with st.spinner("Searching ChromaDB..."):
            results = search_documents(question, vectorizer, document_id, top_k=5)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_parts = []
        for document, metadata in zip(documents, metadatas):
            context_parts.append(f"Page {metadata['page']}:\n{document}")

        context = "\n\n".join(context_parts)

        with st.spinner("Generating answer..."):
            answer = ask_groq(question, context)

        st.subheader("Answer")
        st.write(answer)