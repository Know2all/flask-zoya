from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes,MessageHandler,filters
from telegram import ReplyKeyboardRemove
from Zoya import ZoyaChatbot

BOT_TOKEN = '5786024728:AAGQFmtp5wEhy7Kzq_1ruoLzDKyX4LixSC8'
GEMINI_API_KEY = "AIzaSyCqgpJTOLeA-BIk2lrHw2YojZA37NRBTJo"
PROJECT_ID = "116817772526"
DATASET_PATH = "dataset/zoya_mini_v1.json"
VECTOR_STORE_PATH = "vs_zoya_model_v1"
MEMORY_DIR = "memories"

zoyaChatBot = ZoyaChatbot(
    api_key=GEMINI_API_KEY,
    project_id=PROJECT_ID,
    dataset_path=DATASET_PATH,
    vector_db_path=VECTOR_STORE_PATH,
    memory_dir=MEMORY_DIR
)

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I'm your friendly bot 🤖",
        reply_markup=ReplyKeyboardRemove()  # This removes any visible custom keyboard
    )

# /help command handler
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Available commands:\n/start - Start the bot\n/help - Show this help message")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    uid = update.message.from_user.id
    result = zoyaChatBot.ask(user_id=uid,question=user_message)
    await update.message.reply_text(result)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
