import re
from pathlib import Path
import streamlit as st

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# -------------------------
# CONFIG
# -------------------------
PDF_FOLDER = Path("pdf_data")
FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"


# -------------------------
# TEXT CLEANING
# -------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


# -------------------------
# LOAD / BUILD VECTORSTORE
# -------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(FAISS_PATH).exists():
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Build from PDFs if index not found
    documents = []
    for pdf in PDF_FOLDER.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())

    cleaned_docs = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata
        )
        for doc in documents
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(cleaned_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)

    return vectorstore


# -------------------------
# LOAD LLM
# -------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300
    )

    return HuggingFacePipeline(pipeline=pipe)


# -------------------------
# PROMPT
# -------------------------
def build_prompt(context: str, question: str) -> str:
    return f"""
You are a technical assistant.

Answer the question strictly using the context below.
Do NOT use outside knowledge.
If the answer is not found, say:
"Not found in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
# -------------------------
# STREAMLIT CHAT UI
# -------------------------
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.title("📄 PDF Q&A Chatbot")
    st.markdown(
        """
        **Document-based Question Answering**
        
        - Ask questions in natural language  
        - Answers are generated only from the PDFs  
        - Powered by FAISS + Hugging Face
        """
    )
    st.markdown("---")
    st.markdown("Type `exit` to clear chat")

# Main title
st.title("💬 Document Chatbot")

# Load backend (cached)
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = load_llm()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is sensor fusion?"
)

send = st.button("Send")

# Handle question
if send and question.strip():

    # Clear chat command
    if question.lower() == "exit":
        st.session_state.messages = []
        st.experimental_rerun()

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = build_prompt(context, question)
    answer = llm.invoke(prompt)

    # Save to history
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    st.session_state.messages.append(
        {"role": "bot", "content": answer}
    )

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="
                background:#E8F0FE;
                padding:12px 16px;
                border-radius:12px;
                margin:10px 0;
                max-width:80%;
                margin-left:auto;
                font-size:15px;
            ">
            <b>You</b><br>{msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background:#F1F3F4;
                padding:12px 16px;
                border-radius:12px;
                margin:10px 0;
                max-width:80%;
                font-size:15px;
            ">
            <b>Chatbot</b><br>{msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
