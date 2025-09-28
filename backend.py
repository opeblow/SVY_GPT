from fastapi import FastAPI,HTTPException,status
from pydantic import BaseModel ,field_validator
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
from typing import List,Tuple
import uvicorn
from contextlib import asynccontextmanager

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in environment variable.")
app=FastAPI(
    title="SVY AGENT API",
    description="A fastapi application for a Geomatics-related RAG agent.",
    version="1.0.0"
)
INDEX_DIR="faiss_index"
llm=None
embeddings=None
vector_store=None
chain=None

MEMORY_FILE="chat_memory.json"
MEMORY_VERSION="1.0"
PROMPT_DIR="prompts"
PROMPT_VERSION="v1"
app.state.chat_history=[]

def load_memory()->List[Tuple[str,str]]:
    """Load chat history from json file"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE,"r")as file:
            try:
               data=json.load(file)
               if isinstance(data,dict) and data.get("version")==MEMORY_VERSION:
                 history=data["history"]
               else:
                  print("Unversioned memory file detected. Attempting to load as list.")
                  history=data if isinstance(data,list) else []
               for item in history:
                   if not (isinstance(item,list) and len(item)==2 and all(isinstance(s,str) for s in item)):
                      raise ValueError(f"Invalid memory entry:{item}. Expected [str,str]")
               print("Chat memory loaded successfully.")
               return history
            except (json.JSONDecodeError,ValueError)as e:
                print(f"Error loading memory:{e}.Returning empty history.")
                return []

        

def save_memory(memory:List[Tuple[str,str]]):
    """Save chat history to json file"""
    try:
        with open(MEMORY_FILE,"w")as file:
            json.dump({"version":MEMORY_VERSION,"history":memory},file,indent=2)
        print("Chat memory saved successfully.")
    except Exception as e:
        print(f"Error saving memory:{e}")
        raise

def migrate_memory():
    try:
        old_history=load_memory()
        new_history=[(q,a,{"timestamp":"2025-09-28"}) for q,a in old_history]
        with open(MEMORY_FILE,"w")as file:
            json.dump({"version":"2.0","history":new_history},file,indent=2)
        print("Memory migrated to version 2.0 with timestamps.")
        return new_history
    except Exception as e:
        print(f"Memory migration failed:{e}")
        raise

def load_system_prompt()->str:
    """Load system prompt from a versioned file"""
    prompt_file=os.path.join(PROMPT_DIR,f"system_prompt_{PROMPT_VERSION}.txt")
    if not os.path.exists(prompt_file):
        print(f"Prompt file{prompt_file} not found. Using default prompt.")
        return(
            "You are SVY AGENT,an expert AI specializing in Geomatics-related topics."
            "Answer user questions with clear,accurate,and  concise explanations."
            "In a professional yet approachable tone."
            "Only use chat history when the user explicitly asks about previous questions,"
            "Such as 'what did i just ask?' or 'what was my last question?'"
            "Do not provide formulas or answers to mathematical problems in LaTeX code;"
            "Instead,explain mathematical concepts in plain text if neccessary."
        )
    with open(prompt_file,"r")as file:
        print(f"Loaded system prompt from {prompt_file}")
        return file.read()


def initialize_components():
    global llm,embeddings,vector_store,chain
    print("Initializing components...")
    try:
        device="cuda" if torch.cuda.is_available() else "cpu"
        embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device":device}
        )
        print("Embeddings model successfully loaded!")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model;{e}")
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(f"Faiss index dictionary '{INDEX_DIR}' not found.")
    try:
        vector_store=FAISS.load_local(
            folder_path=INDEX_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("Faiss index loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load faiss index from '{INDEX_DIR}':{e}")
    try:
        llm=ChatOpenAI(
            model="gpt-4o",
            openai_api_key=openai_api_key,
            temperature=0.7,
            max_tokens=500

        )
        print("LLM Initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize llm:{e}")
    #system prompt
    system_prompt=(
        
    "You are SVY Agent, an expert AI specializing in Geomatics-related topics. "
    "Answer user questions with clear, accurate, and concise explanations in a professional yet approachable tone."
    "Only use chat history when the user explicitly asks about previous questions,"
    "such as 'what did i just ask?.'"
    "Do not provide formulas or answers to mathematical problems in LaTex code;"
    'Instead,explain mathematical concepts in plain text if neccessary.'
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
            combine_docs_chain_kwargs={"prompt":prompt_template.partial(system_prompt=system_prompt)}
        )
        print("ConversationalRetrievalChain created successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to create retrieval chain :{e}")
@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        os.makedirs(PROMPT_DIR,exist_ok=True)
        initialize_components()
        if not os.path.exists(MEMORY_FILE):
            save_memory([])
        app.state.chat_history=load_memory()
        print("Startup completed successfully.")
    except (OSError,RuntimeError)as e:
        print(f"Startup failed:{e}")
        raise 
    yield
    print("Shutdown completed")
app.lifespan=lifespan
class QueryRequest(BaseModel):
    message:str
    @field_validator("message")
    def validate_message(cls,value):
        if len(value)>1000:
            raise ValueError("Message is too long")
        return value
@app.get("/health")
async def health_check():
    if chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SVY AGENT is not ready.Check server logs for details."
        )
    try:
        load_memory()
        return {"status":"ok","message":"SVY AGENT API is running and ready."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Memory file error:{e}"
        )
@app.post("/query")
async def query_agent(request:QueryRequest):
    global chat_history
    if not chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AGENT is not initialized. Please check server status."
        )
    try:
        response=chain.invoke({"question":request.message})
        answer=response.get("answer",response)
        chat_history.append((request.message,answer))
        save_memory(chat_history)
        return {"answer":answer}
    except Exception as e:
        print(f"Error during query:{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured:{e}"
        )
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)



    






       
            

    
            


            
                

