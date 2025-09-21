from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import json
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uvicorn
load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
app=FastAPI()#initializing fastapi app
INDEX_DIR="faiss_index"#path to faiss index folder
vector_store=FAISS.load_local(INDEX_DIR,embeddings=None,allow_dangerous_deserialization=True)
with open(os.path.join(INDEX_DIR,"index.json"),"r")as f:
    docs_metadata=json.load(f)
llm=ChatOpenAI(
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
class QueryRequest(BaseModel):
    message:str
@app.post("/query")
async def query_agent(request:QueryRequest):
    try:
        human_message=request.message
        response=chain({"question":human_message})
        return {"answer":response["answer"]}
    except Exception as e:
        return {"error":str(e)}
if __name__=="__main__":
    uvicorn.run("backend:app",host="127.0.0.1",port=8000,reload=True)
