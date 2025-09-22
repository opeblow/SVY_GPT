SVY AgentOverviewSVY Agent is an AI-powered conversational tool designed to answer questions related to Geomatics by leveraging a Retrieval-Augmented Generation (RAG) pipeline. It processes PDF documents, extracts relevant information, and provides accurate, professional responses using natural language processing. The project includes two implementations:OpenAI Implementation (main.py): Uses OpenAI's text-embedding-3-small for retrieval and GPT-4o for generation.
Hugging Face Implementation (hugginface.py): Uses Hugging Face's sentence-transformers/all-MiniLM-L6-v2 for retrieval and GPT-4o for generation.

Both implementations extract text from PDFs, create a FAISS vector store for retrieval, and support interactive queries via a command-line interface.FeaturesExtracts text from PDF files in a specified directory.
Splits text into chunks (1200 characters, 100-character overlap) for efficient retrieval.
Generates embeddings using either OpenAI or Hugging Face models.
Stores embeddings in a FAISS index for fast similarity search.
Provides conversational responses using OpenAI's GPT-4o model, tailored to Geomatics topics.
Maintains conversation history for context-aware responses.
Supports batch processing for large document sets with progress logging.


PrerequisitesPython: 3.8 or higher
OpenAI API Key: Required for both implementations (for embeddings in main.py and generation in both).
PDF Files: Place Geomatics-related PDFs in a directory (default: C:\Users\user\Documents\svy agent\ALL_PDF_FILES).
Hardware:Minimum: 8GB RAM, 2-core CPU.
Recommended: 16GB RAM, 4-core CPU for large document sets.



Installation
1.Clone the Respiratory:
git clone https://github.com/opeblow/SVY_GPT.git
cd SVY_GPT


2.Install Dependencies:
Create a virtual environment and install required packages:
python -m venv myenv
source myenv/bin/activate #On windows:myenv\Scripts\activate
pip install pypdf langchain langchain-openai langchain-community faiss-cpu python-dotenv
for the Hugging Face implementation,also install:
pip install sentence-transformers tqdm


3.Set Up Environment Variables:
Create a .env file in this project root with your OpenAIAPIKey:
env
OPENAI_API_KEY="your-openai-api-key


4.Prepare pdf Files
Place your Geomatics-related pdf files in the directory specified in the code(default:c:Users\user\Documents\svyagent\ALL_PDF_FILES).Update the PDF_PATH variable in thr scripts if needed.
Usage

1.Run the OpenAI Implementation
python main.py
This will:
Extract text from PDFs
Generate embeddings using OpenAI'S text-embedding-3-small.
Build a FAISS index
Start an interactive loop for queries

2.Run the hugging Face Implementation:
python hugging_face.py
This will:
Extract text from PDFs
Generate embeddings using Hugging Face's sentence -transformers/all-MiniLN-L6-v2
Build and save a FAISS INDEX TO the faiss_index dictionary
Start and interactive loop for queries

3.Interaction with SVY Agent:
Enter a question(e.g,"What is the role of GIS IN Geomatics?)
The agent will retrieve relevant document chunks and generate a response
Type quit to exit the interactive loop



Example Interaction
Welcome to SVY GPT!Type your question(or 'quit' to exit):
You:What is the role of GIS in Geomatics?
Assistant Response:Geographic information Systems(GIS) play a central role in Geomatics by enabling the collection,storage,analysis,and visualization of spatial data.GIS tools help professionals map and analyze geographic features,supporting applications like urban planning,land surveying,and environmental management.
You:quit



Implementation DetailsOpenAI Implementation (main.py)Embedding Model: OpenAI text-embedding-3-small.
Generation Model: OpenAI GPT-4o.
Pros: High-quality embeddings, seamless integration with OpenAI's ecosystem.
Cons: Requires OpenAI API credits, potentially higher cost.



Hugging Face Implementation (hugginface.py)Embedding Model: Hugging Face sentence-transformers/all-MiniLM-L6-v2.
Generation Model: OpenAI GPT-4o.
Pros: Cost-effective embeddings, suitable for local deployment.
Cons: May have slightly lower retrieval accuracy compared to OpenAI embeddings.



Both implementations use the LangChain framework for text splitting, vector storage (FAISS), and conversational retrieval. The FAISS index in hugginface.py is saved locally for reuse, while main.py generates it dynamically.ConfigurationPDF Directory: Modify PDF_PATH in the scripts to point to your PDF folder.
Chunk Size: Adjust chunk_size (default: 1200) and chunk_overlap (default: 100) in the CharacterTextSplitter for different text granularity.
Retrieval Parameters: Change search_kwargs={"k": 3} in the ConversationalRetrievalChain to retrieve more or fewer document chunks.
Model Parameters: Adjust temperature (default: 0.7) or max_tokens (default: 500) in the ChatOpenAI configuration for response style.



TroubleshootingNo text extracted from PDFs: Ensure PDFs contain extractable text (not image-based). Consider adding OCR support for scanned documents.
OpenAI API errors: Verify your API key and check for rate limits or network issues.
Memory issues: Reduce the batch size in main.py (e.g., batch_size=5) or hugginface.py (e.g., batch_size=16) for large document sets.
No documents to embed: Check the PDF_PATH directory and ensure PDFs are present and readable.



Future EnhancementsAdd support for additional document formats (e.g., DOCX, TXT).
Implement a web or GUI interface for easier access.
Explore local LLMs for generation to reduce OpenAI API dependency.
Optimize FAISS index for larger datasets with hierarchical indexing.


Contact:
For questions or support ,contact opeblow2021@gmail.com
