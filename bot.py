import requests
import os
from telegram.ext import Application,CommandHandler,filters,MessageHandler
import logging
from dotenv import load_dotenv
import httpx
load_dotenv()
TELEGRAM_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Please set TELEGRAM_BOT-TOKEN in your .env file.")
FASTAPI_URL=os.getenv("FASTAPI_URL")
if not FASTAPI_URL:
    raise ValueError("Please set FASTAPI_URL in your environment.")
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
REQUEST_TIMEOUT=30
client=httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
async def start(update,context):
    """Handles /start command"""
    await update.message.reply_text(
        "Hi! I am SVY AGENT,your Geomatics expert.\n"
        "Ask me anything about surveying,mapping,and geospatial data."
    )
async def handle_message(update,context):
    """Handles incoming messages"""
    user_message=update.message.text
    logger.info(f"User:{user_message}")
    await context.bot.sed_chat_action(chat_id=update.effective_chat.id,action="typing")
    try:
        response=await client.post(
            FASTAPI_URL,
            json={"message":user_message,"debug":False}
        )

        if response.status_code==200:
            answer=response.json().get("answer","No response received.")
            await update.message.reply_text(answer)
        else:
            logger.error(f"API error{response.status_code}:{response.text}")
            await update.message.reply_text(
                "Service temporarily unavailable.Please try again."
            )
    except httpx.TimeoutException:
        logger.error("Request Timeout")
        await update.message.reply_text("Request timed out.Please try again.")

    except Exception as e:
        logger.error(f"Error:{e}",exc_info=True)
        await update.message.reply_text(
            "An error occured.Please try again later."
        )

async def error_handler(update,context):
    """Handles errors"""
    logger.error(f"Update {update} caused error:{context.error}",exc_info=context.error)

async def post_shutdown(application):
    """Cleanup on shutdown"""
    await client.aclose()

def main():
    """Starts the bot"""
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(MessageHandler(filters.TEXT & filters.COMMAND,handle_message))
    app.add_error_handler(error_handler)

    app.post_shutdown=post_shutdown
    logger.info("Bot Started")
    app.run_polling(allowed_updates=["message"])


if __name__=="__main__":
    main()

