from fastapi import FastAPI ,HTTPException,UploadFile,File,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from typing import Optional,List
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import faiss
import json
import os
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uvicorn



#setting up logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

load_dotenv()
faiss_index_path=r"C:\Users\user\Documents\svy agent\faiss_index"
pdf_base_path=r"C:\Users\user\Documents\svy agent\ALL_PDF_FILES"

#setiing up request
class QueryRequest(BaseModel):
    question:str=Field(
        ...,min_length=1,max_length=1000,
        description="The question to ask SVY AGENT"
    )
    sesion_id:Optional[str]=Field(None,description="Optional session id for conversation memory.")

#setting up response
class QueryResponse(BaseModel):
    answer:str=Field(
        ...,description="The AGENT'S response"
    )
    session_id:Optional[str]=Field(
        None,description="Session ID if provided"
    )  
    timestamp:datetime=Field(default_factory=datetime.now)


#setting up healthy response
class HealthResponse(BaseModel):
    status:str
    message:str
    timestamp:datetime=Field(default_factory=datetime.now)

#setting up the agent
class SVYAgent:
    """ SVY Agent class to encapsulate all the AI functionality"""
    def __init__(self):
        self.chain=None
        self.vector_store=None
        self.llm=None
        self.embedding=None
        self.sessions={}
        self.is_initialized=False

    async def initialize(self):
        """Initializing SVY AGENT asynchronously"""
        try:
            logger.info("Initializing SVY AGENT...") 
            opeanai_api_key=os.getenv("OPENAI_API_KEY")
            if not opeanai_api_key:
                raise ValueError("Please set up openai api key...")
            self.embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")#initialize embeddings
            if os.path.exits(faiss_index_path):
                logger.info("Loading existing FAISS Index...")
                self.vector_store=FAISS.load_loacl(faiss_index_path,self.embedding,allow_dangerous_deserialization=True)
            else:
                logger.warning("No existing FAISS Index found.Please run the indexing process first..")
                await self._create_initial_index()
            self.llm=ChatOpenAI(
                opeanai_api_key=opeanai_api_key,
                temperature=0.7,
                max_tokens=500
            )
            system_prompt=(
                "You are SVY AGENT,an expert AI specializing in Geomatics related topics."
                "Answer user questions with clear,accurate,and concise explanations in a professional yet approachable tone."
            )
            self.prompt_template=PromptTemplate(
                input_variables=["context","question"],
                template=(
                    f"{system_prompt}\n\n"
                    "Relevant Context:\n{contect}\n\n"
                    "Human:{question}\n\n"
                    "Assistant"
                )
            )
            self.is_initialized=True
            logger.info("SVY AGENT successfully initialized!")
        except Exception as e:
            logger.error(f"Error initializing SVY AGENT:{e}")
            raise
    async def _create_initial_index(self):
        
            """ Create initial FAISS index from PDFs if none exits"""
            try:
                if not os.paths.exits(pdf_base_path):
                    logger.warning(f"PDF PATH {pdf_base_path} not found. Creating empty index....")
                    self.vector_store=FAISS.from_texts(["Empty index"],self.embedding)
                    return 
                all_texts={}
                for root,dirs,files in os.walk(pdf_base_path):
                    for file in files:
                        if file.endswith(".pdf"):
                            pdf_file_path=os.path.join(root,file)
                            text=await self._extract_text_from_pdf(pdf_file_path)
                            if text:
                                all_texts[file]=text
                if not all_texts:
                    logger.warning("No PDFs found . Creating empty index.")
                    self.vector_store=FAISS.from_texts(["Empty index"],self.embedding)
                    return
                #splitting texts and creating vector store
                text_splitter=CharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=100,
                    separator="\n"
                )
                documents=[]
                metadata=[]
                for pdf_name,text in all_texts.items():
                    chunks=text_splitter.split_text(text)
                    for i,chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadata.append({"pdf_name":pdf_name,"chunk_id":f"{pdf_name}_{i}"})
                if documents:
                    self.vector_store=FAISS.from_texts(documents,self.embedding,metadatas=metadata)
                    self.vector_store.save_local("faiss_index")
                    logger.info(f'Created and saved FAISS Index with {len(documents)}chunks....')
                else:
                    logger.error(f'Error creating initial index:{e}')
            except Exception as e:
                logger.error(f"Error creating index:{e}")
                self.vector_store=FAISS.from_texts(["Empty index"],self.embedding)
    async def _extract_text_from_pdf(self,pdf_path:str):
        """Extracts texts from PDFs asynchronously"""
        #Running pdf extraction in thread pool to avoid blocking
        try:
                loop=asyncio.get_event_loop()
                return await loop.run_in_executor(None,self._sync_extract_text_from_pdf,pdf_path)
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}:{e}")
            return ""
    def _sync_extract_text_from_pdf(self,pdf_path:str):
        """Synchronous pdf extraction"""
        try:
            reader=PdfReader(pdf_path)
            text=""
            for page in reader.pages:
                content=page.extract_text()
                if content:
                    text+=content+"\n"
            return text
        except Exception as e:
            logger.error(f"Error reading {pdf_path}:{e}")
            return ""
    def get_or_create_session(self,session_id:Optional[str]=None):
        """Get or create a conversation memory for conversation"""
        if session_id is None:
            session_id="default"
        if session_id not in self.sessions:
            self.sessions[session_id]=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.sessions[session_id]
    async def query(self,question:str,session_id:Optional[str]):
        if not self.is_initialized:
            raise RuntimeError("SVY AGENT not initialized..")
        try:
            memory=self.get_or_create_session(session_id)
            chain=ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k:3"}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt":self.prompt_template},
                return_source_documents=False
            )
            loop=asyncio.get_event_loop()
            response=await loop.run_in_executor(None,lambda:chain({"question":question}))
            return response["answer"]
        except Exception as e:
            logger.error(f"Error processing query:{e}")
            raise HTTPException(status_code=500,detail="Error processing query.")
agent=SVYAgent()
@asynccontextmanager
async def lifespan(app:FastAPI):
    """Application lifespan manager."""
    logger.info("Starting SVY AGENT FastAPI application.")
    await agent.initialize()
    yield
#creating FastAPI app
app=FastAPI(
    title="SVY AGENT API",
    description="AI-powered Geomatics expert using RAG with PDF knowledge base",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.get("/",response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="success",
        message="SVY AGENT API is running"
    )
@app.get("/health",response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if agent.is_initialized:
        return HealthResponse(
            status="Initializing",
            message="SVY AGENT is still initializing."
        )
@app.post("/query",response_model=QueryResponse)
async def query_agent(request:QueryRequest):
    """Query the SVY AGENT with a question"""
    if not agent.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="SVY AGENT is still initializing please try again in a moment."
        )
    try:
        answer=await agent.query(request.qustion,request.session_id)
        return QueryResponse(
            answer=answer,
            session_id=request.sesion_id
        )
    except Exception as e:
        logger.error(f"Error in query endpoint:{e}")
        raise HTTPException(status_code=500,detail="Internal server error.")
    
@app.delete("/sessions/{session_id}")
async def clear_session(session_id:str):
    """Clear conversation memory for a specific session."""
    if session_id in agent.sessions:
        del agent.sessions[session_id]
        return {"message":f"Session{session_id} cleared successfully.."}
    else:
        raise HTTPException(status_code=404,detail="Session not found.")
    
@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {"sessions":list(agent.sessions.keys())}

@app.post("/upload-pdf")
async def upload_pdf(background_tasks:BackgroundTasks,file:UploadFile=File(...)):
    """Upload a PDF file to add to the knowledge base"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400,detail="File must be a PDF.")
    try:
        upload_dir="upload_pdfs"
        os.makedirs(upload_dir,exist_ok=True)
        file_path=os.path.join(upload_dir,file.filename)

        with open(file_path,"wb")as buffer:
            content=await file.read()
            buffer.write(content)
        background_tasks.add_task(process_upload_pdf,file_path)
        return {"message":f"PDF {file.filename} uploaded successfully and will be processed shortly"}
    except Exception as e:
        logger.error(f"Error uploading pdf:{e}")
        raise HTTPException(status_code=500,detail="Error uploading pdf.")

async def process_upload_pdf(file_path:str):
    """    Background task to process uploaded pdf and update vector store."""
    try:
        text=await agent._extract_text_from_pdf(file_path)
        if text:
            text_splitter=CharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100,
                separator="/n"
            )
            chunks=text_splitter.split_text(text)
            pdf_name=os.path.basename(file_path)
            metadata=[{"pdf_name":pdf_name,"chunk_id":f"{pdf_name}_{i}"} for i in range (len(chunks))]
            agent.vector_store.aadd_texts(chunks,metadatas=metadata)
            agent.vector_store.save_local(faiss_index_path)
            logger.info(f"Successfully processed and added {pdf_name} to knowledge base.")
        else:
            logger.warning(f"No text extracted from {file_path}")
    except Exception as e:
        logger.error(f"Error processing uploaded pdf {file_path}")

if __name__=="__main__":
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000
    )





       
            

    
            


            
                

