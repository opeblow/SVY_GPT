"""Telegram bot for SVY Agent."""
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import config
from src.utils.logging_config import logger
from src.handlers.commands import start_command, help_command, status_command
from src.handlers.messages import handle_message, initialize_rag
from src.handlers.errors import error_handler


def create_bot_application() -> Application:
    """Create and configure the Telegram bot application.
    
    Returns:
        Configured Application instance
    """
    app = Application.builder().token(config.telegram_token).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.add_error_handler(error_handler)
    
    logger.info("Bot application created successfully")
    return app


def main():
    """Run the Telegram bot."""
    logger.info("Initializing RAG service...")
    initialize_rag()
    logger.info("Starting Telegram bot...")
    
    app = create_bot_application()
    
    app.run_polling(
        allowed_updates=["message", "channel_post", "edited_message", "edited_channel_post"],
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
