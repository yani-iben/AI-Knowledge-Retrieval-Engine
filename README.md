# AI-Knowledge-Retrieval-Engine

# PDF Q&A App

A Streamlit-based application that allows users to **upload PDFs and ask questions** about their contents. The app leverages **local LLMs via Ollama (Llama 3)** for free, local inference, and can optionally be configured to use **OpenAI GPT models** for cloud-based deployments.

---

## Demo

A short video demonstrating the app’s functionality is included in this repository:

- [Download / Play the Demo Video]([demo/pdf_chat_demo.mp4](https://github.com/yani-iben/AI-Knowledge-Retrieval-Engine/blob/main/Chat%20with%20Pdfs%20Demo%20(1).mp4))

**Features shown in the demo:**
- Uploading multiple PDF documents
- Splitting PDFs into manageable text chunks
- Querying content with natural language questions
- Retrieving contextually relevant snippets
- Generating responses using a local LLM (Ollama)

---

## Technology Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)  
- **PDF Processing:** [PyPDF2](https://pypi.org/project/PyPDF2/)  
- **Text Splitting:** [LangChain Text Splitters](https://python.langchain.com/en/latest/modules/indexes/text_splitters.html)  
- **Vector Store:** FAISS via LangChain  
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`  
- **LLM Backend:**  
  - Local: [Ollama / Llama 3](https://ollama.com/)  
  - Optional cloud: OpenAI GPT (requires API key & billing)  

---

## Local Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/AI-Knowledge-Retrieval-Engine.git
cd AI-Knowledge-Retrieval-Engine
