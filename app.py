import streamlit as st
import os
import tempfile
from groq import Groq
from chromadb.utils import embedding_functions
import chromadb

import PyPDF2
from docx import Document
st.set_page_config(page_title="Student RAG Project", layout="wide")

try:
    api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar to start.")
        st.stop()

client = Groq(api_key=api_key)

if os.path.exists("/mount/data"):
    persist_directory = "/mount/data/chroma_db"
else:
    persist_directory = "./chroma_db"

embedding_model_name = "all-MiniLM-L6-v2"
def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def read_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading TXT: {e}")
        return ""

def split_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

st.title("📚 Enterprise Knowledge Base (RAG)")
st.markdown("Upload your company documents. Once uploaded, they stay saved even if you close the browser.")

@st.cache_resource
def get_vector_db():
    client = chromadb.PersistentClient(path=persist_directory)    
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )    
    collection = client.get_or_create_collection(
        name="enterprise_kb",
        embedding_function=embed_fn
    )
    return collection

db_collection = get_vector_db()
st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Drag and drop files here", 
    type=["pdf", "txt", "docx"], 
    accept_multiple_files=True
)
if st.sidebar.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Reading and embedding documents..."):
            temp_dir = tempfile.mkdtemp()
            all_chunks = []
            all_metadatas = []
            
            for file in uploaded_files:
                file_ext = os.path.splitext(file.name)[1].lower()
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                text_content = ""
                if file_ext == ".pdf":
                    text_content = read_pdf(temp_path)
                elif file_ext == ".docx":
                    text_content = read_docx(temp_path)
                elif file_ext == ".txt":
                    text_content = read_txt(temp_path)
                
                if text_content:
                    chunks = split_text(text_content)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadatas.append({"source": file.name, "chunk_id": i})   
            if all_chunks:                
                db_collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=[f"doc_{i}_{os.urandom(4).hex()}" for i in range(len(all_chunks))]
                )                
                st.sidebar.success(f"Added {len(all_chunks)} chunks to database!")
            else:
                st.sidebar.error("No text found.")
    else:
        st.sidebar.warning("Upload files first.")
st.markdown("---")
st.header("2. Ask a Question")
count = db_collection.count()
if count == 0:
    st.info("Knowledge base is empty. Please upload documents on the left.")
else:
    st.success(f"Knowledge base active with {count} data chunks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            results = db_collection.query(
                query_texts=[prompt],
                n_results=3
            )
            
            context_text = ""
            sources = set()
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    context_text += f"Chunk {i+1}:\n{doc}\n\n"
                    if results['metadatas'] and results['metadatas'][0]:
                         sources.add(results['metadatas'][0][i].get('source', 'Unknown'))

            system_prompt = "You are a helpful assistant. Answer based ONLY on the context. If unknown, say 'I don't know'."
            user_message = f"Context:\n{context_text}\n\nQuestion: {prompt}"
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
            )
            
            answer = chat_completion.choices[0].message.content
            if sources:
                answer += f"\n\n*Sources: {', '.join(list(sources))}*"
            
            full_response = answer

        except Exception as e:
            full_response = f"Error: {str(e)}"

        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
