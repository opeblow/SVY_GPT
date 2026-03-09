"""Message handler for processing user messages."""
from telegram import Update
from telegram.ext import ContextTypes

from config import config
from src.utils.logging_config import logger
from src.services.rag_service import RAGService

rag_service = RAGService()

PROCESSING_MESSAGE = " Processing your query... please wait"
ERROR_MESSAGE = " Sorry, the Geomatics Agent encountered an error. Please try again later."
UNAVAILABLE_MESSAGE = " The service is temporarily unavailable. Please try again later."


def initialize_rag():
    """Initialize the RAG service."""
    rag_service.initialize()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    user_message = update.message.text
    user_id = update.effective_user.id
    
    logger.info(f"Message from user {user_id}: {user_message[:50]}...")
    
    if not rag_service.is_ready:
        logger.error(f"RAG service not ready for user {user_id}")
        await update.message.reply_text(UNAVAILABLE_MESSAGE)
        return
    
    processing_msg = await update.message.reply_text(PROCESSING_MESSAGE)
    
    try:
        result = rag_service.query(user_message, debug=config.debug)
        answer = result.get("answer", "No response received.")
        
        await processing_msg.delete()
        await update.message.reply_text(answer)
        logger.info(f"Response sent to user {user_id}")
        
    except Exception as e:
        logger.exception(f"Error processing message for user {user_id}: {e}")
        await processing_msg.delete()
        await update.message.reply_text(ERROR_MESSAGE)
