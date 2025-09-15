from pypdf import PdfReader
import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()#loading the environment
#initializing open api key

openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable.")
#setting the file path of the folder that contains all pdf files
PDF_PATH=r"C:\Users\user\Documents\svy agent\ALL_PDF_FILES"
def extract_text_from_pdf(pdf_path):
            """ Extracts texts from the pdf files"""
            try:
                    reader=PdfReader(pdf_path)
                    text=""
                    for page in reader.pages:
                            content=page.extract_text()
                            if content:
                                    text+=content + "\n"
                    return text
            except Exception as e:
                    print(f"Error reading{pdf_path}:{e}")
                    return ""
#Going through each subfolder in the ALL_PDF_FILES folder and extracting their text
all_texts={}
for root,dirs,files in os.walk(PDF_PATH):
        for file in files:
                if file.endswith("pdf"):
                        pdf_path=os.path.join(root,file)
                        text=extract_text_from_pdf(pdf_path)
                        #storing extraxted texts from pdfs inside a dictionary
                        all_texts[file]=text
                        print(f"Extracted {len(text)} characters from {file}")
#splitting texts
text_splitter=CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
)
#prepare both documents and metadata
documents=[]
metadata=[]
for pdf_name,text in all_texts.items():
        chunks=text_splitter.split_text(text)
        for i,chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({"pdf_name":pdf_name,"chunk_id":f"{pdf_name}_{i}"})
        print(f"Split {pdf_name} into {len(chunks)} chunks.")
#generating embeddings
embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",openai_api_key=openai_api_key
)
vector_store=FAISS.from_texts(documents,embedding,metadatas=metadata)
print("Generated embeddings and created FAISS index.")
#initializing chatopenai with gpt-40
llm=ChatOpenAI(
        model="gpt-40",
        openai_api_key=openai_api_key,
        temperature=0.7,
        max_tokens=500
)
#message schema part
system_prompt=(
        "You are SVY Agent,and expert AI specializing in Geomatics related topics. Answer user questions with clear,accurate,and concise explanations in a professional yet approachable tone."
)
prompt_template=PromptTemplate(
        input_variables=["context","question","chat_history"],
        template=(
                "{system_prompt}\n\n"
                "Chat_History:\n{chat_history}\n\n"
                "Human:{question}\n\n"
                "Assistant"
        )
)
#setting up conversation
memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
)          
#setting up a conversational retrieval chain
chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k":3}),#getting top 3 chunks
        memory=memory,
        combine_docs_chain_kwargs={"prompt":prompt_template.partial(system_prompt=system_prompt)},
        return_source_documents=False
)
#defining a query function
def query_svy_gpt(human_message):
        """ Process a human query and return the assistant's response"""
        response=chain({"question":human_message})
        return response["answer"]
#setting up interactive loop
if __name__=="__main__":
        print("Welcome to SVYY GPT!Type your question(or 'quit' to exit):")
        while True:
                human_message=input("You:")
                if human_message.lower()=='quit':
                        break
                answer=query_svy_gpt(human_message)
                print(f"\nAssistant Response:{answer}\n")