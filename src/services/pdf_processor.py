"""PDF processing service for extracting and embedding documents."""
import os
import json
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

from config import config
from src.utils.logging_config import logger


class PDFProcessor:
    """Service for processing PDF documents and building the vector store."""

    def __init__(self, pdf_directory: Optional[str] = None):
        self.pdf_directory = pdf_directory or config.pdf_directory
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator="\n"
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
            return ""

    def process_all_pdfs(self) -> tuple[list, list]:
        """Process all PDFs in the directory.
        
        Returns:
            Tuple of (documents, metadata)
        """
        all_texts = {}
        
        if not os.path.exists(self.pdf_directory):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_directory}")
        
        for root, _, files in os.walk(self.pdf_directory):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    text = self.extract_text_from_pdf(pdf_path)
                    all_texts[file] = text
                    if not text:
                        logger.warning(f"No text extracted from {file}")
        
        return self._split_and_chunk(all_texts)

    def _split_and_chunk(self, all_texts: dict) -> tuple[list, list]:
        """Split texts into chunks and create metadata.
        
        Args:
            all_texts: Dictionary mapping filenames to text content
            
        Returns:
            Tuple of (document chunks, metadata)
        """
        documents = []
        metadata = []
        
        for pdf_name, text in all_texts.items():
            if not text:
                logger.warning(f"Skipping {pdf_name}: No text to split")
                continue
                
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    "pdf_name": pdf_name,
                    "chunk_id": f"{pdf_name}_{i}"
                })
            logger.info(f"Split {pdf_name} into {len(chunks)} chunks")
        
        logger.info(f"Total documents: {len(documents)}, Total metadata: {len(metadata)}")
        
        if not documents:
            raise ValueError("No documents to embed. Check PDF extraction.")
        
        return documents, metadata

    def build_vector_store(
        self,
        documents: list,
        metadata: list,
        save_directory: Optional[str] = None
    ) -> FAISS:
        """Build and save the FAISS vector store.
        
        Args:
            documents: List of document chunks
            metadata: List of metadata dictionaries
            save_directory: Directory to save the vector store
            
        Returns:
            The created FAISS vector store
        """
        save_dir = save_directory or config.index_dir
        
        logger.info("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": config.embedding_device}
        )
        
        logger.info("Creating embeddings (this may take a while)...")
        embeddings_list = self._embed_documents_with_progress(documents, embeddings)
        
        logger.info("Building FAISS index...")
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(documents, embeddings_list)),
            embedding=embeddings,
            metadatas=metadata
        )
        
        logger.info(f"Saving vector store to {save_dir}...")
        vector_store.save_local(save_dir)
        
        # Save metadata
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "index.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Vector store built and saved successfully")
        return vector_store

    def _embed_documents_with_progress(
        self,
        documents: list,
        embeddings: HuggingFaceEmbeddings,
        batch_size: int = 32
    ) -> list:
        """Embed documents in batches with progress tracking.
        
        Args:
            documents: List of document chunks
            embeddings: Embeddings model
            batch_size: Number of documents per batch
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        for i in tqdm(
            range(0, len(documents), batch_size),
            desc="Embedding chunks",
            unit="batch"
        ):
            batch = documents[i:i + batch_size]
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
