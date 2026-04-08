# 📚 Enterprise Knowledge Base (RAG)

A production-ready **Question-Answering application** that allows employees to query internal documents using natural language.

Instead of keyword searching (Ctrl+F), this system uses **RAG (Retrieval-Augmented Generation)** to understand the semantic meaning of your documents and provide citation-backed answers.

---

## 🚀 Why I built it this way

The original requirements asked for AWS Bedrock. While powerful, setting up Bedrock requires configuring S3 buckets, IAM roles, and dealing with complex cloud permissions—which is overkill for a project that needs to be demonstrable quickly.

I chose this specific stack to keep it **fast, free, and locally runnable** while still proving the core RAG concept:

*   **Groq (instead of OpenAI/AWS):** Groq hosts Llama 3 on LPUs (Language Processing Units). The speed is instant, and the API is free for developers. No AWS bill surprises.
*   **Streamlit:** I wanted to focus on the logic, not CSS. Streamlit turns Python scripts into shareable web apps in minutes.
*   **ChromaDB:** I needed a "memory" for the app. Instead of paying for a cloud vector database (like Pinecone), Chroma saves the embeddings to the local disk. It's lightweight and open-source.

---
Sample live - [https://enterprise-knowledge-base-q-a-99.streamlit.app/]

<img width="1899" height="777" alt="Screenshot 2026-04-08 155709" src="https://github.com/user-attachments/assets/b6fa381e-9b07-4dff-9343-08423170b9ef" />


## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **LLM/Inference:** Groq (Llama 3.3 70B Versatile)
*   **Vector Database:** ChromaDB (Persistent Client)
*   **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
*   **File Parsing:** PyPDF2, python-docx

---

## ✨ Features

1.  **Multi-Format Support:** Upload PDF, DOCX, or TXT files.
2.  **Semantic Search:** Finds answers based on meaning, not just exact keywords.
3.  **Persistent Memory:** Once you upload documents, they stay saved even if you refresh the page (or redeploy, if using persistent storage).
4.  **Citations:** The AI tells you exactly which file it used to answer the question.

---

## 📋 Installation & Usage

### Prerequisites
*   Python 3.9 or higher
*   A Groq API Key ([Get one here](https://console.groq.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/PavanKumarAadelli/Enterprise-Knowledge-Base-Q-A.git
cd Enterprise-Knowledge-Base-Q-A
```

### 2. Install Dependencies
**Important:** I have pinned specific library versions in `requirements.txt` to avoid common compatibility errors (like NumPy 2.0 conflicts).

```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

1.  Paste your Groq API Key in the sidebar.
2.  Upload your documents.
3.  Click "Process Documents."
4.  Start chatting!

---

## ☁️ Deployment on Streamlit Cloud

To run this 24/7 for free:

1.  Push this code to a **GitHub Repository**.
2.  Go to [share.streamlit.io](https://share.streamlit.io) and click "New app".
3.  Connect your repository and deploy.
4.  **Secrets:** Do not hardcode your API key.
    *   Go to your App Settings on Streamlit Cloud.
    *   Add a Secret: `GROQ_API_KEY` = `your_actual_key_here`.

---

## 🐛 Troubleshooting

**Error: `AttributeError: module 'numpy' has no attribute 'float_'`**
*   **Cause:** ChromaDB is not yet compatible with NumPy 2.0.
*   **Fix:** Ensure your `requirements.txt` includes `numpy==1.26.4`.

**Error: `TypeError` related to `protobuf`**
*   **Cause:** Version mismatch between Google's protobuf and Chroma.
*   **Fix:** Ensure your `requirements.txt` includes `protobuf==4.25.1`.

---

## 📝 Current `requirements.txt`

If you are having trouble installing, use this exact configuration:

```text
streamlit
groq
chromadb==0.4.24
sentence-transformers
PyPDF2
python-docx
protobuf==4.25.1
numpy==1.26.4
```

---

## 📄 License

This project is open source and available for educational purposes.

---

*Built with ❤️ using Python and Groq.*
