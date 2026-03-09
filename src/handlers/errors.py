"""Error handler for Telegram bot."""
from telegram import Update
from telegram.ext import ContextTypes

from src.utils.logging_config import logger

ERROR_RESPONSE = " An internal error occurred. Please try a different query."


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot.
    
    Args:
        update: Telegram update object
        context: Bot context with error information
    """
    logger.error(f"Update {update} caused error {context.error}")
    
    if update and update.message:
        try:
            await update.message.reply_text(ERROR_RESPONSE)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")


