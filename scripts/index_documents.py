"""Indexing script to build the FAISS vector store from PDFs.

This script processes PDF documents and creates a searchable vector store
that can be used by the RAG service.
"""
from src.services.pdf_processor import PDFProcessor
from src.utils.logging_config import logger
from config import config


def main():
    """Build the FAISS vector store from PDF documents."""
    logger.info("=" * 50)
    logger.info("Starting PDF indexing process...")
    logger.info("=" * 50)
    
    try:
        processor = PDFProcessor(pdf_directory=config.pdf_directory)
        
        logger.info(f"Processing PDFs from: {config.pdf_directory}")
        documents, metadata = processor.process_all_pdfs()
        
        logger.info("Building vector store...")
        processor.build_vector_store(documents, metadata)
        
        logger.info("=" * 50)
        logger.info("Indexing complete! The vector store is ready.")
        logger.info("=" * 50)
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        raise
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
        
    except Exception as e:
        logger.exception(f"Unexpected error during indexing: {e}")
        raise


if __name__ == "__main__":
    main()
