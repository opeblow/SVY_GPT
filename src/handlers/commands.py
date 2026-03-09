"""Telegram bot command handlers for SVY Agent."""
from telegram import Update
from telegram.ext import ContextTypes

from src.utils.logging_config import logger

START_MESSAGE = """ Welcome to SVY Agent!

I am your Geomatics and Surveying expert AI assistant. 
Ask me anything related to:
• Land Surveying
• Geographic Information Systems (GIS)
• Remote Sensing
• Cartography
• Geoinformatics
• Urban and Regional Planning

How can I help you today? """


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    logger.info(f"Start command from user {update.effective_user.id}")
    await update.message.reply_text(START_MESSAGE)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    help_text = """ SVY Agent Help

Commands:
/start - Start the bot and see welcome message
/help - Show this help message
/status - Check if the backend is running

Just send me any question about Geomatics, 
Surveying, or related topics and I'll answer!
"""
    logger.info(f"Help command from user {update.effective_user.id}")
    await update.message.reply_text(help_text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    from src.main import rag_service
    
    if rag_service.is_ready:
        await update.message.reply_text(" Backend is running and ready!")
    else:
        await update.message.reply_text(" Backend is initializing or not available.")
