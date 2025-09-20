from pypdf import PdfReader
import os
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Loading environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable in .env file.")

# PDF folder path
PDF_PATH = r"C:\Users\user\Documents\svy agent\ALL_PDF_FILES"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        print(f"Extracted {len(text)} characters from {pdf_path}")
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Extracting text from all PDFs
all_texts = {}
for root, dirs, files in os.walk(PDF_PATH):
    for file in files:
        if file.endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            text = extract_text_from_pdf(pdf_path)
            all_texts[file] = text
            if not text:
                print(f"Warning: No text extracted from {file}")

# Spliting texts into larger chunks 
text_splitter = CharacterTextSplitter(
    chunk_size=1200,   
    chunk_overlap=100,
    separator="\n"
)

documents = []
metadata = []
for pdf_name, text in all_texts.items():
    if text:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({"pdf_name": pdf_name, "chunk_id": f"{pdf_name}_{i}"})
        print(f"Split {pdf_name} into {len(chunks)} chunks.")
    else:
        print(f"Skipping {pdf_name}: No text to split.")

print("Documents count:", len(documents), "Metadata count:", len(metadata))
if not documents:
    raise ValueError("No documents to embed. Check PDF extraction.")

# Hugging Face Embeddings 
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embeding in batches with progress bar
def embed_with_progress(texts, embedding, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=" Embedding chunks"):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

print(" Starting embeddings...")
all_embeddings = embed_with_progress(documents, embedding, batch_size=32)

# Building FAISS index
vector_store = FAISS.from_embeddings(
    [(doc, emb) for doc, emb in zip(documents, all_embeddings)],
    embedding,
    metadatas=metadata
)
print(" Finished embeddings and created FAISS index.")
SAVE_DIR="faiss_index"
vector_store.save_local(SAVE_DIR)
print(f"Faiss Index saved to {SAVE_DIR}")
#  ChatOpenAI 
llm = ChatOpenAI(
    model="gpt-4o",  
    openai_api_key=openai_api_key,
    temperature=0.7,
    max_tokens=500
)

# System prompt
system_prompt = (
    "You are SVY Agent, an expert AI specializing in Geomatics-related topics. "
    "Answer user questions with clear, accurate, and concise explanations in a professional yet approachable tone."
)
prompt_template = PromptTemplate(
    input_variables=["context", "question", ],
    template=(
        "{system_prompt}\n\n"
        "Relevant Context:\n{context}\n\n"
        "Human: {question}\n\n"
        "Assistant: "
    )
)

# Conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template.partial(system_prompt=system_prompt)},
    return_source_documents=False
)

# Query function
def query_svy_gpt(human_message):
    """Process a human query and return the assistant's response."""
    try:
        response = chain({"question": human_message})
        return response["answer"]
    except Exception as e:
        print(f"Error processing query: {e}")
        return "Sorry, an error occurred while processing your query."

# Interactive loop
if __name__ == "__main__":
    print("Welcome to SVY GPT! Type your question (or 'quit' to exit):")
    while True:
        human_message = input("You: ")
        if human_message.lower() == "quit":
            break
        answer = query_svy_gpt(human_message)
        print(f"\nAssistant Response: {answer}\n")