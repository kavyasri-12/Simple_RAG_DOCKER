import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import numpy as np
import faiss
import time
import json
import streamlit as st
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set or is empty.")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}. "
             "Please ensure GOOGLE_API_KEY is set in your .env file or as an environment variable.")
    st.stop()

EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "gemini-1.5-pro-latest"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- 1. Document Loading ---
def load_txt_document(file):
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return None

def load_pdf_document(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# --- 2. Text Splitting (Chunking) ---
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start >= len(text) - chunk_overlap and start < len(text):
            if len(text) - (start + chunk_overlap) > 0:
                last_chunk_start = max(0, len(text) - chunk_size)
                if not chunks or chunks[-1] != text[last_chunk_start:]:
                    chunks.append(text[last_chunk_start:])
            break
        elif start >= len(text):
            break
    return [c for c in chunks if c.strip()]

# --- 3. Embedding Generation ---
@st.cache_data(show_spinner=False)
def generate_embedding(text):
    if not text:
        return None
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response['embedding']
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# --- 4. Vector Storage (FAISS) and Retrieval ---
class VectorStore:
    def __init__(self, embedding_dimension=768):
        self.index = None
        self.documents = [] # This will store {"text": chunk_text, "metadata": metadata}
        self.embedding_dimension = embedding_dimension

    def build_from_chunks(self, chunks_with_metadata):
        """Builds (or rebuilds) the vector store from a fresh list of chunks."""
        self.index = None # Reset the index
        self.documents = [] # Clear existing documents
        st.write(f"Building vector store with {len(chunks_with_metadata)} chunks...")
        
        if not chunks_with_metadata:
            st.info("No chunks to add to the vector store.")
            return

        new_embeddings = []
        new_documents = []

        progress_text = "Generating embeddings..."
        my_bar = st.progress(0, text=progress_text)

        for i, (chunk_text, metadata) in enumerate(chunks_with_metadata):
            embedding = generate_embedding(chunk_text)
            if embedding:
                new_embeddings.append(embedding)
                new_documents.append({"text": chunk_text, "metadata": metadata})
            my_bar.progress((i + 1) / len(chunks_with_metadata), text=progress_text)

        my_bar.empty() # Remove the progress bar

        if not new_embeddings:
            st.warning("No embeddings generated from the provided chunks.")
            return

        new_embeddings_array = np.array(new_embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index.add(new_embeddings_array)
        self.documents.extend(new_documents)
        st.success(f"Vector store successfully built with {self.index.ntotal} embeddings.")


    def search(self, query_text, k=5):
        """Searches for the top-k most similar documents."""
        query_embedding = generate_embedding(query_text)
        if not query_embedding:
            return []

        query_embedding_array = np.array([query_embedding]).astype('float32')

        if self.index is None or self.index.ntotal == 0:
            st.info("Vector store is empty. No search performed.")
            return []

        actual_k = min(k, self.index.ntotal)
        if actual_k == 0: return []

        distances, indices = self.index.search(query_embedding_array, actual_k)

        results = []
        for i in indices[0]:
            if i < len(self.documents):
                results.append(self.documents[i])
        return results

# --- 5. Generation (with retrieved context) ---
def generate_response_with_context(query, retrieved_chunks, generation_model=GENERATION_MODEL):
    context_str = "\n".join([f"Source ({doc['metadata']['source']}): {doc['text']}" for doc in retrieved_chunks])

    if not context_str:
        prompt = f"Answer the following question: {query}"
    else:
        prompt = f"""
You are an AI assistant tasked with answering questions based on provided information.
Use the following context to answer the question. If the answer is not found in the context,
state that you cannot answer based on the provided information.

Context:
{context_str}

Question: {query}

Answer:
"""

    try:
        model = genai.GenerativeModel(generation_model)
        with st.spinner("Generating response..."):
            response = model.generate_content(prompt)
        return response.text, retrieved_chunks
    except Exception as e:
        st.error(f"Error generating content from Gemini: {e}")
        return "Sorry, I couldn't generate a response at this time.", []


# --- Streamlit UI ---
st.set_page_config(page_title="Gemini RAG App", layout="wide")
st.title("📚 Gemini 1.5 Pro RAG for TXT/PDF Documents")
st.markdown("Upload your TXT or PDF documents and ask questions based on their content.")

# Initialize vector store and a counter for the file uploader key
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0 # Used to clear the file uploader widget

# File Uploader
# Assign a dynamic key to the uploader. Changing this key will clear its state.
uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.uploader_key}", # Dynamic key
    help="Select one or more .txt or .pdf files to process. Any previously processed documents not currently selected will be removed from the knowledge base upon processing."
)

# Button to trigger processing and rebuilding the knowledge base
if st.button("Process Selected Documents", key="process_docs_button"):
    all_chunks_with_metadata = []
    if uploaded_files:
        with st.spinner("Preparing documents for processing..."):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_type = uploaded_file.type

                content = None
                if file_type == "text/plain":
                    content = load_txt_document(uploaded_file)
                elif file_type == "application/pdf":
                    content = load_pdf_document(uploaded_file)
                else:
                    st.warning(f"Unsupported file type: {file_type} for '{file_name}'")
                    continue

                if content:
                    chunks = chunk_text(content)
                    for chunk in chunks:
                        all_chunks_with_metadata.append((chunk, {"source": file_name, "type": file_type}))
                    st.info(f"Loaded '{file_name}' ({len(chunks)} chunks).")
                else:
                    st.warning(f"Could not extract content from '{file_name}'. Skipping.")

        # Rebuild the entire vector store with only the currently selected files
        st.session_state.vector_store.build_from_chunks(all_chunks_with_metadata)

    else:
        # If no files are uploaded but button is clicked, clear the vector store
        st.session_state.vector_store = VectorStore() # Reset to empty
        st.warning("No files selected. Knowledge base cleared.")

    # Optionally, clear the file uploader widget after processing if desired
    # This makes the UI feel cleaner but users might find it annoying if they
    # want to quickly add another file without re-selecting previous ones.
    # To clear it:
    # st.session_state.uploader_key += 1
    # st.rerun() # Forces a rerun to apply the new key and clear the widget

st.markdown("---")

# Chat Interface
if st.session_state.vector_store.index is not None and st.session_state.vector_store.index.ntotal > 0:
    st.header("Ask a Question")
    query = st.text_area("Enter your question:", placeholder="E.g., What is RAG?", height=100, key="query_input")

    if st.button("Get Answer", key="get_answer_button"):
        if query:
            with st.spinner("Retrieving relevant information..."):
                start_time = time.time()
                retrieved_docs = st.session_state.vector_store.search(query, k=5)
                retrieval_time = time.time() - start_time
                st.sidebar.info(f"Retrieval time: {retrieval_time:.2f} seconds")

            if retrieved_docs:
                response_text, sources = generate_response_with_context(query, retrieved_docs)
                st.subheader("AI Response:")
                st.write(response_text)

                st.subheader("Source Documents:")
                for i, doc in enumerate(sources):
                    with st.expander(f"Source {i+1}: {doc['metadata']['source']} (Type: {doc['metadata']['type']})"):
                        st.write(doc['text'])
            else:
                st.warning("No relevant information found in the documents for your query.")
                response_text, _ = generate_response_with_context(query, [])
                st.subheader("AI Response (no relevant context found):")
                st.write(response_text)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload documents and click 'Process Selected Documents' to start asking questions.")

st.markdown("---")
st.caption("Powered by Google Gemini 1.5 Pro and FAISS")