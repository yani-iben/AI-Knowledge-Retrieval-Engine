import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage




@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return Ollama(
        model="llama3",
        temperature=0,
    )


# ---------------------------------
# Helper functions
# ---------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks):
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embeddings)


# ---------------------------------
# Main app
# ---------------------------------
def main():
    # Streamlit config
    st.image(
    "logo.jpg",
    width=250
    )

    st.set_page_config(
        page_title="Chat with PDFs (Local LLaMA)",
        layout="wide",
    )

    # Styling
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #f0f0f0;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # UI
    st.title("Chat with PDFs")

    user_question = st.text_input("Ask a question about your PDFs")

    with st.sidebar:
        st.header("Your documents")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True,
        )

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = split_text(raw_text)
                    st.session_state.vectorstore = build_vectorstore(chunks)
                    st.success("PDFs processed successfully")

        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.success("Chat cleared")

    # Chat logic
    if user_question:
        if not st.session_state.vectorstore:
            st.warning("Please upload and process PDFs first.")
            return

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(user_question)
        context = "\n\n".join(d.page_content for d in docs)

        llm = load_llm()

        prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not contained in the context, say you do not know.

Context:
{context}

Question:
{user_question}
"""

        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)

        # Store history
        st.session_state.chat_history.append(
            HumanMessage(content=user_question)
        )
        st.session_state.chat_history.append(
            AIMessage(content=response)
        )

        # Display messages
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(response)

        with st.expander("Retrieved document snippets"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Snippet {i}:** {doc.page_content[:500]}...")


# ---------------------------------
# Entry point
# ---------------------------------
if __name__ == "__main__":
    main()
