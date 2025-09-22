# streamlit.py
import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please create a .env file with your key.")
    st.stop()

# -------------------------
# Page config (must be before other Streamlit commands)
# -------------------------
st.set_page_config(page_title="SVY GPT Academic Agent", layout="wide")

# -------------------------
# App title
# -------------------------
st.title(" SVY GPT Academic Agent")
st.markdown(
    "Ask any question related to *Geomatics/Surveying*. "
    "SVY GPT will answer using your PDF knowledge base."
)

# -------------------------
# Load FAISS index
# -------------------------
SAVE_DIR = "faiss_index"

@st.cache_resource(show_spinner=True)
def load_vector_store(save_dir):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True)
    return vector_store

with st.spinner("Loading FAISS index..."):
    vector_store = load_vector_store(SAVE_DIR)

# -------------------------
# Sidebar settings
# -------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("LLM Model", ["gpt-4o", "gpt-3.5-turbo"], index=0)
    show_sources = st.checkbox("Show retrieved documents", value=True)

# -------------------------
# Initialize LLM, memory, and chain
# -------------------------
llm = ChatOpenAI(
    model=model_name,
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    max_tokens=500
)

system_prompt = (
    "You are SVY Agent, an expert AI specializing in Geomatics-related topics. "
    "Provide clear, accurate, and concise answers in a professional yet approachable tone. "
    "If no relevant context is available, respond based on your general knowledge of Geomatics."
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "{system_prompt}\n\n"
        "Context (if available):\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer: "
    )
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Hard-code top_k = 5
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template.partial(system_prompt=system_prompt)},
    return_source_documents=True
)

# -------------------------
# Session state for chat
# -------------------------
if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
    st.session_state.messages = []

# Clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []

# -------------------------
# User input
# -------------------------
user_input = st.text_input("Type your question here and press Enter:")

if user_input:
    with st.spinner("SVY GPT is thinking..."):
        try:
            response = chain({"question": user_input})
            answer = response["answer"]
            source_docs = response.get("source_documents", [])
        except Exception as e:
            answer = f"An error occurred: {e}"
            source_docs = []
            st.error(answer)

    # Append to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------
# Display chat messages in chat-style bubbles
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Show retrieved document chunks
# -------------------------
if show_sources and user_input and source_docs:
    with st.expander("Retrieved Document Chunks"):
        for i, doc in enumerate(source_docs, 1):
            meta = getattr(doc, "metadata", {})
            pdf_name = meta.get("pdf_name", "Unknown PDF")
            st.markdown(f"*Chunk {i} from {pdf_name}*")
            st.write(doc.page_content)