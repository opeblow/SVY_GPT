from fastapi import FastAPI ,HTTPException,status
from pydantic import BaseModel,field_validator
from typing import Optional
import torch
import faiss
import json
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uvicorn

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
app=FastAPI(
    title="SVY AGENT API",
    discription="A FastAPI application for a Geomatics-related RAG agent",
    version="1.0.0"
)
INDEX_DIR="faiss_index"
llm=None
embeddings=None
vector_store=None
chain=None
def initialize_components():
    global llm,embeddings,vector_store,chain
    print("Initializing components..")
    try:
        device="cuda" if torch.cuda.is_available() else "cpu"
        embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device":device}
        )
        print("Embeddings model successfully loaded!")
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings model :{e}")
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(f"Faiss index dictionary '{INDEX_DIR}' not found")
    try:
        vector_store=FAISS.load_local(folder_path=INDEX_DIR,
                                      embeddings=embeddings,allow_dangerous_deserialization=True)
        print("FAISS Index loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS Index from '{INDEX_DIR}':{e}")
    try:
        llm=ChatOpenAI(
            model="gpt-4o",
            openai_api_key=openai_api_key,
            temperature=0.7,
            max_tokens=500

        )
        print("LLM Initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM:{e}")
    system_prompt=(
        "You are SVY AGENT,an expert AI specializing in Geomatics-related topics."
        "Answer user questions with clear,aacurate,and concise explanation."
        "In a professional yet approachable tone."

    )
    prompt_template=PromptTemplate(
        input_variables=["system_prompt","context","question"],
        template=(
            "{system_prompt}\n\n"
            "Relevant Context:\n{context}\n\n"
            "Human:{question}\n\n"
            "Assistant:"
        )
    )
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    try:
        chain=ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k":5}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt":prompt_template.partial(system_prompt=system_prompt)},
            return_source_documents=True
        )
        print("ConversationalRetrievalChain created successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to create retrieval chain:{e}")

@app.on_event("startup")
async def startup_event():
    try:
        initialize_components()
    except Exception as e:
        print(f"Start up failed:{e}")
        raise
class QueryRequest(BaseModel):
    message:str
    debug: Optional[bool] = False
    @field_validator("message")
    def validate_message(cls,value):
        if len(value)>1000:
            raise ValueError("Message is too long")
        return value
@app.get("/health")
async def health_check():
    if chain:
        return{"status":"ok","message":"SVY AGENT API Is running and ready"}
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="SVY AGENT is not ready. Check server logs for details." 
    )
@app.post("/query")
async def query_agent(request:QueryRequest):
    if not chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is not initialized. Please check server status."
        )
    try:
        # Always run a similarity search first so we can log deterministic top-k with scores
        try:
            top_k_with_scores = vector_store.similarity_search_with_score(request.message, k=5)
            print("\n===== Retrieved Top-k Documents (k=5) =====")
            for rank, (doc, score) in enumerate(top_k_with_scores, start=1):
                meta = getattr(doc, "metadata", {}) if doc else {}
                pdf_name = meta.get("pdf_name", "Unknown PDF")
                chunk_id = meta.get("chunk_id", "unknown_chunk")
                snippet = (doc.page_content[:300] + "...") if getattr(doc, "page_content", None) else "<no content>"
                print(f"#{rank} score={score:.4f} source={pdf_name} chunk={chunk_id}\n{snippet}\n")
            print("===== End Retrieved Documents =====\n")
        except Exception as e:
            print(f"Similarity search logging failed: {e}")

        response=chain.invoke({"question":request.message})
        answer=response.get("answer",response)

        # Optionally return sources when debug flag is true
        if request.debug:
            source_docs = response.get("source_documents", []) if isinstance(response, dict) else []
            serialized_sources = []
            for doc in source_docs:
                meta = getattr(doc, "metadata", {})
                serialized_sources.append({
                    "pdf_name": meta.get("pdf_name", "Unknown PDF"),
                    "chunk_id": meta.get("chunk_id", "unknown_chunk"),
                    "content_preview": getattr(doc, "page_content", "")[:500]
                })
            return {"answer": answer, "sources": serialized_sources}

        return {"answer":answer}
    except Exception as e:
        print(f"Error during query:{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured:{e}"
        )
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)



    






       
            

    
            


            
                

