

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import GROQ_API_KEY


st.set_page_config(page_title="Academic RAG Assistant", layout="wide")
st.title("Academic RAG Assistant")
st.write("Upload PDFs, then ask questions about them.")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model_name="openai/gpt-oss-120b")

embeddings = load_embeddings()
llm = load_llm()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None




def pdfs_to_chunks(uploaded_files):
    """Turn a list of uploaded PDF files into small text chunks."""
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for uploaded_file in uploaded_files:
        # PyPDFLoader needs a real file path, so save the upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        pages = PyPDFLoader(tmp_path).load()
        os.unlink(tmp_path) 
        chunks.extend(splitter.split_documents(pages))

    return chunks


def answer_question(question, vectorstore):
    """Retrieve relevant chunks and ask the LLM to answer using them."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""You are an academic assistant helping a student study.
Use the context below to answer the question.
If the answer isn't in the context, say "I couldn't find this in the uploaded material."

Context:
{context}

Question: {question}

Answer:"""

    return llm.invoke(prompt).content


def generate_questions(topic, num_questions, vectorstore):
    """Generate practice exam questions on a topic from the uploaded material."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(topic)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Based on the study material below, write {num_questions} exam-style
questions about '{topic}'. Mix short-answer and conceptual questions. Number them.

Study material:
{context}

Questions:"""

    return llm.invoke(prompt).content



with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            chunks = pdfs_to_chunks(uploaded_files)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)")

    st.divider()
    st.header("Generate practice questions")
    topic = st.text_input("Topic")
    num_questions = st.slider("Number of questions", 3, 10, 5)

    if st.button("Generate questions") and topic:
        if st.session_state.vectorstore is None:
            st.warning("Upload and process PDFs first.")
        else:
            with st.spinner("Generating..."):
                questions = generate_questions(topic, num_questions, st.session_state.vectorstore)
            st.markdown("### Practice questions")
            st.write(questions)


st.divider()

if st.session_state.vectorstore is None:
    st.info("Upload PDFs from the sidebar to get started.")
else:
    # Re-show past messages (needed because the script reruns on every action)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about your study material...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(question, st.session_state.vectorstore)
                st.write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})