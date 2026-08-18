import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import tempfile
import os
from config import GROQ_API_KEY

VECTOR_STORE_PATH = "vectorstore"

st.set_page_config(page_title="Academic RAG Assistant", layout="wide")
st.title("📚 Academic RAG Assistant")
st.write("Upload lecture notes, study material, or previous year papers and ask questions.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template("""You are an academic assistant helping a student study.
Use the following context from their study material to answer the question.
If the answer is not in the context, say "I couldn't find this in the uploaded material."

Context:
{context}

Question: {question}

Answer:""")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Load existing vectorstore on startup
if st.session_state.vectorstore is None and os.path.exists(VECTOR_STORE_PATH):
    with st.spinner("Loading existing knowledge base..."):
        vs = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        st.session_state.vectorstore = vs
        st.session_state.rag_chain = build_rag_chain(vs)

with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs (lecture notes, PYQs, study material)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs"):
        all_chunks = []
        with st.spinner("Processing PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                os.unlink(tmp_path)

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(documents)
                all_chunks.extend(chunks)

            if os.path.exists(VECTOR_STORE_PATH):
                vs = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                vs.add_documents(all_chunks)
            else:
                vs = FAISS.from_documents(all_chunks, embeddings)

            vs.save_local(VECTOR_STORE_PATH)
            st.session_state.vectorstore = vs
            st.session_state.rag_chain = build_rag_chain(vs)

        st.success(f"✅ {len(all_chunks)} chunks indexed from {len(uploaded_files)} file(s)")

    st.divider()
    st.header("🎯 Generate Practice Questions")
    topic = st.text_input("Topic (e.g. 'thermodynamics', 'polymers')")
    num_questions = st.slider("Number of questions", 3, 10, 5)

    if st.button("Generate Questions") and topic:
        if st.session_state.vectorstore is None:
            st.warning("Upload and process PDFs first.")
        else:
            with st.spinner("Generating questions..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(topic)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""You are an exam question generator. Based on the following study material, generate {num_questions} exam-style questions on the topic of '{topic}'. 
Mix short answer and conceptual questions. Number them clearly.

Study material:
{context}

Generate {num_questions} exam questions:"""

                response = llm.invoke(prompt)
                st.markdown("### Practice Questions")
                st.write(response.content)

st.divider()

if st.session_state.rag_chain is None:
    st.info("👈 Upload PDFs from the sidebar to get started.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question from your study material...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)
                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})