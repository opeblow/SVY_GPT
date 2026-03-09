"""FastAPI backend for SVY Agent."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from config import config
from src.utils.logging_config import logger
from src.services.rag_service import RAGService

rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    logger.info("Starting SVY Agent API...")
    try:
        rag_service.initialize()
        logger.info("SVY Agent API started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    yield
    logger.info("Shutting down SVY Agent API...")


app = FastAPI(
    title="SVY Agent API",
    description="A FastAPI application for a Geomatics-related RAG agent",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    message: str
    debug: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Message cannot be empty")
        if len(value) > 1000:
            raise ValueError("Message is too long (max 1000 characters)")
        return value.strip()


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    if rag_service.is_ready:
        return HealthResponse(
            status="ok",
            message="SVY Agent API is running and ready"
        )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="SVY Agent is not ready. Check server logs for details."
    )


@app.post("/query")
async def query_agent(request: QueryRequest) -> dict:
    """Query the RAG agent.
    
    Args:
        request: Query request with message and optional debug flag
        
    Returns:
        Response with answer and optionally source documents
    """
    if not rag_service.is_ready:
        logger.error("Query received but RAG service is not ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is not initialized. Please check server status."
        )
    
    logger.info(f"Received query: {request.message[:100]}...")
    
    try:
        result = rag_service.query(request.message, debug=request.debug)
        logger.info(f"Query processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        )


def main():
    """Run the FastAPI server."""
    import uvicorn
    
    port = config.fastapi_port
    uvicorn.run(
        "src.main:app",
        host=config.fastapi_host,
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()
