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
FASTAPI_URL="http://127.0,0,1:8000/query"
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
#Handles for /start command
async def start(update,context):
    await update.message.reply_text("Hi i am SVY AGENT,your Geomatics expert.Ask me anything!")

#Handles text messages
async def handle_message(update,context):
    user_message=update.message.text
    logger.info(f"Received message:{user_message}")

    #Handles temporary message to let user know the query is processing
    temp_message=await update.message.reply_text("Processing your query..please wait")

    try:
        async with httpx.AsyncClient(timeout=60.0)as client:
            payload={"message":user_message,"debug":False}
            response=await client.post(FASTAPI_URL,json=payload)
        #removes the temporary message now that there is a response

        await temp_message.delete()
        if response.status_code==200:
            answer=response.json().get("answer","No response from the agent.")
            await update.message.reply_text(answer)

        else:
            error_detail=response.json().get("detail","Unknown error")
            logger.error(f"FastAPI error:{response.status_code}-Detail:{error_detail}")
            await update.message.reply_text(
                f"Sorry,the Geomatics Agent encountered an error (status:{response.status_code}).Please try again"
            )
    except httpx.ConnectError:
        #handles connection failure
        logger.error(f"HTTPX ConnectError:Could not connect to FastAPI seerver at {FASTAPI_URL}")
        await update.message.reply_text(
            "Error:Could not connect to the Geomatics Agent server.Please ensure the backend is running."
        )
        await temp_message.delete()

async def error_handler(update,context):
    """Log the error and send a message."""
    logger.error(f"Update{update}caused error {context.error}")
    if update and update.message:
        await update.message.reply_text("An  internal bot error occured.Please try a different query.")


# Main Function
def main():
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_message))
    app.add_error_handler(error_handler)

    logger.info("Starting Telegram bot....Ensure FastAPI is running on http://127.0.0.1:8000")
    app.run_polling(allowed_updates=["message","channel_post","edited_message","edited_channel_post"])

if __name__=="__main__":
    main()

