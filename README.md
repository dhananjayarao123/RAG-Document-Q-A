📚 Lightweight Document Q&A
A lightweight, fully local vector-search RAG (Retrieval-Augmented Generation) application built with Streamlit, Scikit-Learn, ChromaDB, and Groq.

Upload a PDF, ask questions, and get answers grounded only in your document—without needing heavy embedding models like HuggingFace/Sentence-Transformors (which can exceed 500MB+ with PyTorch dependencies).

🏗️ Tech Stack
UI: Streamlit
LLM: Groq (using llama-3.1-8b-instant for ultra-fast inference)
Vector Database: ChromaDB (Local/Persistent)
Embeddings: Scikit-Learn TfidfVectorizer (Ultra-lightweight alternative to Sentence-Transformers)
PDF Parsing: pypdf
🚀 Setup & Installation
Clone the repository and navigate to the folder:
cd lightweight-document-qa
Create and activate a virtual environment:
python -m venv venvsource venv/bin/activate  # On Windows use: venv\Scripts\activate
Install the lightweight dependencies:
pip install streamlit chromadb groq pypdf scikit-learn python-dotenv numpy
Set up your environment variables: Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY=gsk_your_api_key_here
Run the application:
streamlit run app.py
⚠️ Architecture & Debugging Notes
Building a "lightweight" RAG app comes with a few non-obvious pitfalls. If you are modifying this code or experiencing blank outputs, read this section carefully.

Problem 1: The "Blank Screen" on LLM Response
Symptom: You ask a question, the spinner finishes, but nothing appears on the screen.Cause: The Groq API does not throw a standard Python error if you pass an invalid model name (e.g., openai/gpt-oss-20b). It simply returns an empty response object. Solution: Ensure GROQ_MODEL is set to a valid Groq model ID (e.g., llama-3.1-8b-instant).

Problem 2: Why HashingVectorizer Fails for RAG
Symptom: The app runs, but answers are completely hallucinated or the LLM says "I don't know", even for simple facts.Cause: HashingVectorizer uses a fixed mathematical hash. It does not understand semantic similarity. The vector for "What is the revenue?" and "The revenue was $5M" have completely different mathematical representations and high distance. ChromaDB returns random chunks, and the LLM fails.Solution: We use TfidfVectorizer instead. It matches on shared vocabulary (e.g., the word "revenue"), which works reliably for keyword-matching RAG without requiring heavy ML models.

Problem 3: Scikit-Learn csr_matrix Append Error
Symptom: AttributeError: 'csr#_matrix' object has no attribute 'append'Cause: Scikit-Learn's vectorizers return a sparse matrix (.transform()). If you name your sparse matrix variable embeddings and also try to create an empty list called embeddings to append to, Python will confuse the two.Solution: Always rename the sparse matrix (e.g., tfidf_matrix) and keep the list separate (e.g., embedding_list).

Problem 4: ChromaDB Dimension Mismatch
Symptom: InvalidArgumentError: Collection expecting embedding with dimension of 1024, got 16839Cause: If you previously ran the app with HashingVectorizer (which forces 1024 dimensions), ChromaDB saves that schema permanently. TfidfVectorizer creates a dimension for every unique word in your document (e.g., 16,839 dimensions). ChromaDB will refuse to accept the new shape.Solution: You must completely delete the chroma_db folder so ChromaDB can regenerate the collection with the new dimension size.

# You can do this programmatically before creating the collection:import shutilshutil.rmtree("./chroma_db", ignore_errors=True)
Problem 5: "Document Already Stored" Skipping Vectorizer
Symptom: You upload a document, it says "Already stored", you ask a question, and the app crashes because st.session_state["vectorizer"] is missing.
Cause: Because TfidfVectorizer must be fitted to the specific document's vocabulary to work, we store the fitted vectorizer in Streamlit's session state. If the app skips processing because the doc is already in ChromaDB, the vectorizer is never created.
Solution: Implement a st.button("Force Reprocess Document") that deletes the old ChromaDB records for that document ID and triggers a rerun, ensuring the vectorizer gets fitted and saved to session state.

💡 How It Works
1.Upload & Parse: pypdf extracts text page-by-page.
2.Chunking: Text is split into 1000-character chunks with a 200-character overlap.
3.Vectorization: TfidfVectorizer fits to the document's vocabulary and converts chunks into TF-IDF vectors.
4.Storage: Vectors, text, and metadata are saved in local ChromaDB.
5.Query: The user's question is vectorized using the same fitted vectorizer.
6.Retrieval: ChromaDB performs a similarity search to find the top 5 most relevant chunks.
7.Generation: The chunks are injected into a prompt as "Context" and sent to Groq for a fast, accurate answer.
