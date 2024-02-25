import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from utils.get_answer_hf_api import get_answer
from config_data.config import Config, load_config

config: Config = load_config()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text=f"Hi, {update.effective_user.first_name}! Write me anything you want!"
    )


async def respond_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        reply_to_message_id=update.effective_message.id,
        text="Loadind the best answer...\nPlease wait couple of minutes :)"
    )
    response_text = get_answer(update.message.text)
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text=response_text
    )


application = ApplicationBuilder().token(config.tg_bot.token).build()

start_handler = CommandHandler('start', start)
respond_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), respond_to_user)

application.add_handler(start_handler)
application.add_handler(respond_handler)


if __name__ == '__main__':    
    application.run_polling()