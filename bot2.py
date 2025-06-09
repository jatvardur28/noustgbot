import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
import httpx
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
BOT_TOKEN = os.getenv('BOT_TOKEN')
NOUS_API_KEY = os.getenv('NOUS_API_KEY')

if not BOT_TOKEN or not NOUS_API_KEY:
    raise ValueError("Не найдены BOT_TOKEN или NOUS_API_KEY в .env файле")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# URL для API Nous Research
NOUS_API_URL = "https://inference-api.nousresearch.com/v1/chat/completions"

# [ИЗМЕНЕНО] Доступные модели - НАСТРОЙКИ ДЛЯ МАКСИМАЛЬНОГО КАЧЕСТВА
AVAILABLE_MODELS = {
    "deephermes": {
        "name": "DeepHermes-3-Mistral-24B-Preview",
        "display_name": "🧠 DeepHermes 24B (Быстрая и глубокая)",
        "context": "32k",
        "description": "Быстрая модель, демонстрирующая процесс мышления",
        "system_prompt": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.",
        "max_tokens": 1024,
        "temperature": 0.7,
        "timeout": 120.0
    },
    "hermes405b": {
        "name": "Hermes-3-Llama-3.1-405B",
        "display_name": "🚀 Hermes 405B (Максимальная мощь)",
        "context": "32k", 
        "description": "Самая мощная модель для сложных задач с глубоким анализом",
        # [ИЗМЕНЕНО] Промпт, поощряющий глубокий и развернутый ответ, а не краткость.
        "system_prompt": "You are Hermes 3, one of the most powerful AI assistants in the world. Your goal is to provide deeply reasoned, comprehensive, and accurate answers. Before providing the final response, you can use a long chain of thought to analyze the problem from multiple angles. Be thorough and detailed.",
        # [ИЗМЕНЕНО] Увеличен лимит токенов для полноценных ответов.
        "max_tokens": 2048,
        # [ИЗМЕНЕНО] Температура повышена для более качественных и креативных ответов.
        "temperature": 0.6,
        # [ИЗМЕНЕНО] Таймаут увеличен, чтобы дать модели время подумать.
        "timeout": 240.0
    }
}

# Настройки пользователей: модель, режим отладки
user_settings = {}

# Простая статистика и rate limiting
user_stats = {}
user_last_request = {}  # Для rate limiting
RATE_LIMIT_SECONDS = 3  # Минимальный интервал между запросами

def get_user_model(user_id: int) -> str:
    """Получить текущую модель пользователя"""
    return user_settings.get(user_id, {}).get('model', 'deephermes')

def set_user_model(user_id: int, model: str):
    """Установить модель для пользователя"""
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id]['model'] = model

def get_user_debug_mode(user_id: int) -> bool:
    """Получить режим отладки пользователя"""
    return user_settings.get(user_id, {}).get('debug', False)

def set_user_debug_mode(user_id: int, debug: bool):
    """Установить режим отладки для пользователя"""
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id]['debug'] = debug

def check_rate_limit(user_id: int) -> bool:
    """Проверка rate limiting для пользователя"""
    current_time = time.time()
    last_request = user_last_request.get(user_id, 0)
    
    if current_time - last_request < RATE_LIMIT_SECONDS:
        return False
    
    user_last_request[user_id] = current_time
    return True

def get_time_until_next_request(user_id: int) -> int:
    """Получить время до следующего разрешенного запроса"""
    current_time = time.time()
    last_request = user_last_request.get(user_id, 0)
    remaining = RATE_LIMIT_SECONDS - (current_time - last_request)
    return max(0, int(remaining))

def create_model_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру для выбора модели"""
    keyboard = []
    for model_key, model_info in AVAILABLE_MODELS.items():
        keyboard.append([
            InlineKeyboardButton(
                text=model_info["display_name"],
                callback_data=f"model_{model_key}"
            )
        ])
    keyboard.append([
        InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")
    ])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

async def call_nous_api(user_message: str, user_id: int, retry_count: int = 0) -> str:
    """
    Функция отправки запроса к API Nous Research
    """
    headers = {
        "Authorization": f"Bearer {NOUS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    user_model = get_user_model(user_id)
    model_config = AVAILABLE_MODELS[user_model]
    
    # [ИЗМЕНЕНО] Агрессивная оптимизация (урезание запроса) полностью убрана.
    # Модель должна получать полный контекст от пользователя.
    # Упрощение запроса происходит только при повторной попытке после ошибки.
    if len(user_message) > 1000 and retry_count > 0:
        user_message = user_message[:800] + "... (сообщение сокращено для повторной попытки)"

    # Системный промпт - упрощаем только при повторной попытке
    system_prompt = model_config["system_prompt"]
    if retry_count > 0:
        system_prompt = "You are a helpful AI assistant. Answer briefly and clearly."
    
    max_tokens = model_config["max_tokens"]
    temperature = model_config["temperature"]
    
    # Ограничения при повторной попытке после ошибки
    if retry_count > 0:
        max_tokens = min(max_tokens, 512)
        temperature = 0.2  # Более детерминированный ответ для ретрая
    
    payload = {
        "model": model_config["name"],
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_message
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    try:
        timeout_read = model_config["timeout"]
        if retry_count > 0:
            timeout_read = min(timeout_read, 60.0)
            
        timeout = httpx.Timeout(
            connect=30.0, 
            read=timeout_read, 
            write=30.0, 
            pool=30.0
        )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Запрос к {model_config['display_name']} для пользователя {user_id} (попытка {retry_count + 1})")
            
            response = await client.post(NOUS_API_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data['choices'][0]['message']['content'].strip()
                logger.info(f"✅ Получен ответ от {model_config['display_name']} (длина: {len(ai_response)} символов)")
                return ai_response
                
            elif response.status_code == 429:
                logger.warning("Rate limit достигнут")
                return "🚫 **Превышен лимит запросов API**\n\nПопробуйте через 1-2 минуты. Nous Research ограничивает количество запросов."
                
            elif response.status_code == 400:
                logger.error(f"Ошибка в запросе: {response.text}")
                if retry_count == 0:
                    return await call_nous_api("Ответь кратко на вопрос: " + user_message[:200], user_id, retry_count + 1)
                return "❌ Ошибка в формате запроса. Попробуйте переформулировать вопрос."
                
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return f"❌ Ошибка сервера API ({response.status_code}). Попробуйте позже."
                
    except httpx.TimeoutException:
        logger.error(f"⏰ Timeout при обращении к {model_config['display_name']} (попытка {retry_count + 1})")
        
        if retry_count < 1:
            logger.info("🔄 Повторная попытка с упрощенным запросом...")
            return await call_nous_api(user_message, user_id, retry_count + 1)
            
        if user_model == "hermes405b":
            return f"⏰ **{model_config['display_name']} перегружена или запрос слишком сложный**\n\nМодель не успела ответить за {model_config['timeout']} секунд. Попробуйте:\n• Упростить/разбить ваш вопрос на части\n• Переключиться на DeepHermes 24B (/model)\n• Повторить запрос через минуту"
        else:
            return f"⏰ {model_config['display_name']} работает медленно. Попробуйте задать более короткий вопрос."
            
    except httpx.ConnectError:
        logger.error("🌐 Ошибка соединения с Nous API")
        return "🌐 **Ошибка соединения**\n\nНе удается подключиться к AI-сервису. Проверьте интернет-соединение или попробуйте позже."
        
    except Exception as e:
        logger.error(f"💥 Неожиданная ошибка при запросе к API: {str(e)}")
        return f"❌ Произошла неожиданная ошибка: {str(e)}"

def clean_ai_response(response: str) -> str:
    """
    Очищает ответ AI от технических тегов для пользователя
    """
    import re
    think_pattern = r'<think>.*?</think>'
    cleaned = re.sub(think_pattern, '', response, flags=re.DOTALL)
    return cleaned.strip()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    user_name = message.from_user.first_name or "Друг"
    user_id = message.from_user.id
    current_model = AVAILABLE_MODELS[get_user_model(user_id)]
    
    welcome_text = f"""
🤖 **Привет, {user_name}!** Добро пожаловать в AI-бота на базе Nous Research!

🧠 **Текущая модель:** {current_model['display_name']}

**📋 Доступные модели:**
• 🧠 **DeepHermes 24B** - Быстрая, показывает процесс мышления.
• 🚀 **Hermes 405B** - Максимально мощная, для самых сложных задач и глубокого анализа.

**💡 Возможности:**
• Решение сложных задач по науке и программированию
• Глубокий анализ и развернутые рассуждения  
• Создание качественного творческого контента

**⚡ Команды:**
/start - это сообщение
/model - выбрать модель
/debug - режим отладки (для 24B)
/stats - ваша статистика  
/help - подробная помощь

Просто отправьте мне свой вопрос! 🚀
    """
    await message.answer(welcome_text)

@dp.message(Command("model"))
async def cmd_model(message: Message):
    user_id = message.from_user.id
    current_model = AVAILABLE_MODELS[get_user_model(user_id)]
    
    # [ИЗМЕНЕНО] Обновлены описания моделей
    text = f"""
🤖 **Выбор модели AI**

**Текущая модель:** {current_model['display_name']}

**🧠 DeepHermes 24B (Быстрая и глубокая)**
• ⚡ Скорость: Относительно высокая (~30-60 сек)
• 🎯 Особенность: Показывает процесс мышления в тегах `<think>` (в режиме /debug).
• 👍 Лучше для: Большинства повседневных задач, быстрых рассуждений, программирования.
• 📝 Токены ответа: до {AVAILABLE_MODELS['deephermes']['max_tokens']}

**🚀 Hermes 405B (Максимальная мощь)**
• ⚡ Скорость: Низкая (~2-4 минуты). Требует терпения!
• 🎯 Особенность: Наивысшее качество, глубина и детализация ответа.
• 👍 Лучше для: Очень сложных вопросов, научного анализа, создания высококачественного контента.
• 📝 Токены ответа: до {AVAILABLE_MODELS['hermes405b']['max_tokens']}

**💡 Совет:** Используйте DeepHermes для скорости и Hermes 405B, когда требуется максимальное качество и вы готовы подождать.

Выберите новую модель:
    """
    await message.answer(text, reply_markup=create_model_keyboard())

@dp.callback_query(lambda c: c.data.startswith('model_'))
async def process_model_selection(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    model_key = callback_query.data.split('_')[1]
    
    if model_key in AVAILABLE_MODELS:
        set_user_model(user_id, model_key)
        model_info = AVAILABLE_MODELS[model_key]
        
        # [ИЗМЕНЕНО] Обновлен дополнительный текст
        extra_info = ""
        if model_key == "hermes405b":
            extra_info = "\n\n**⚠️ Внимание:** Эта модель значительно медленнее, но предоставляет ответы высочайшего качества. Будьте готовы к ожиданию."
        
        text = f"""
✅ **Модель изменена!**

**Выбрана:** {model_info['display_name']}
**Описание:** {model_info['description']}
**Контекст:** {model_info['context']} токенов
**Максимум токенов ответа:** {model_info['max_tokens']}{extra_info}

Теперь можете задавать вопросы! 🚀
        """
        
        await callback_query.message.edit_text(text)
        await callback_query.answer(f"✅ Переключено на {model_info['display_name']}")
    else:
        await callback_query.answer("❌ Неизвестная модель")

@dp.callback_query(lambda c: c.data == 'cancel')
async def process_cancel(callback_query: CallbackQuery):
    await callback_query.message.delete()
    await callback_query.answer("Отменено")

# Остальные обработчики команд (/help, /premium, /debug, /stats) остаются без изменений,
# так как их логика не зависит напрямую от настроек моделей.
# Можно лишь поправить тексты в них для актуальности, но это не критично.
@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = """
📚 **Подробная помощь**

🎯 **Возможности:**
• Решение задач с пошаговым анализом
• Программирование на любых языках
• Креативный контент (стихи, рассказы)
• Объяснение сложных концепций
• Анализ данных и выводы
• Помощь с учебными заданиями

🤖 **Модели:**
• **DeepHermes 24B** - Быстрая, показывает мышление.
• **Hermes 405B** - Самая мощная для глубочайшего анализа.

💡 **Примеры запросов:**
• "Объясни теорию струн так, как будто мне 15 лет"
• "Напиши класс для работы с API на Python, используя httpx, с обработкой ошибок и ретраями"  
• "Проанализируй сильные и слабые стороны рыночной и плановой экономики"
• "Реши интеграл: ∫(x^2 * sin(x)) dx"

🔧 **Команды:**
/start - Главное меню
/model - Выбор модели
/debug - Режим отладки (показать <think> для 24B)
/stats - Статистика запросов
/help - Эта справка

⚙️ **Лимиты:**
• 1 запрос в 3 секунды.
• Ожидание ответа от 405B может занимать несколько минут.

🔍 **Режим отладки:**
Команда /debug покажет процесс размышлений DeepHermes в тегах <think>. На Hermes 405B не влияет.
    """
    await message.answer(help_text)

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    user_id = message.from_user.id
    count = user_stats.get(user_id, 0)
    current_model_key = get_user_model(user_id)
    current_model = AVAILABLE_MODELS[current_model_key]
    debug_mode = get_user_debug_mode(user_id)
    last_request_time = user_last_request.get(user_id, 0)
    last_request_str = "Никогда" if last_request_time == 0 else datetime.fromtimestamp(last_request_time).strftime("%H:%M:%S")
    
    stats_text = f"""
📊 **Ваша статистика:**

👤 **Пользователь ID:** {user_id}
• Всего запросов: {count}
• Последний запрос: {last_request_str}

🤖 **Текущие настройки:**
• Модель: {current_model['display_name']}
• Max токенов ответа: {current_model['max_tokens']}
• Температура: {current_model['temperature']}
• Таймаут: {current_model['timeout']}s
• Режим отладки: {'🟢 Включен' if debug_mode else '🔴 Выключен'} (для DeepHermes)

⚡ **О боте:**
• Версия: 3.0 (Unleashed)
• API: Nous Research
    """
    await message.answer(stats_text)


@dp.message()
async def handle_message(message: Message):
    user_message = message.text
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "Пользователь"
    
    if not check_rate_limit(user_id):
        remaining_time = get_time_until_next_request(user_id)
        await message.answer(f"⏳ Подождите {remaining_time} сек. до следующего запроса")
        return
    
    user_stats[user_id] = user_stats.get(user_id, 0) + 1
    
    user_model = get_user_model(user_id)
    model_info = AVAILABLE_MODELS[user_model]
    debug_mode = get_user_debug_mode(user_id)
    
    logger.info(f"📝 Запрос #{user_stats[user_id]} от {user_name} (ID: {user_id}, модель: {model_info['display_name']}): {user_message[:50]}...")
    
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    # [ИЗМЕНЕНО] Обновлены сообщения об ожидании
    status_msg = None
    if user_model == "hermes405b":
        status_msg = await message.answer(f"""
🚀 **Модель {model_info['display_name']}** приняла ваш запрос в работу.

Это самая мощная модель, поэтому ей нужно больше времени на глубокий анализ.

⏱ **Ожидаемое время ответа: 2-4 минуты.**
☕️ Можете пока сделать себе чай. Качественный ответ стоит того, чтобы подождать!
        """)
    elif len(user_message) > 300: # Для длинных запросов к быстрой модели тоже покажем статус
        status_msg = await message.answer(f"🧠 **{model_info['display_name']}** анализирует ваш запрос...\n⏱ Ожидайте до 1 минуты")
    
    try:
        ai_response = await call_nous_api(user_message, user_id)
        
        if status_msg:
            try:
                await status_msg.delete()  
            except:
                pass
        
        if "rate limit" in ai_response.lower() or "429" in ai_response or "превышен лимит" in ai_response.lower():
            await message.answer(ai_response)
            return

        if debug_mode and user_model == "deephermes":
            final_response = ai_response
        else:
            final_response = clean_ai_response(ai_response)
        
        model_prefix = f"🤖 **Ответ от {model_info['display_name']}:**\n\n"
        final_response = model_prefix + final_response
        
        if len(final_response) > 4000:
            parts = []
            current_part = ""
            for line in final_response.split('\n'):
                if len(current_part) + len(line) + 1 > 4000:
                    parts.append(current_part)
                    current_part = line
                else:
                    current_part += '\n' + line
            parts.append(current_part)
            
            for i, part in enumerate(parts):
                if i == 0:
                    await message.answer(part)
                else:
                    await message.answer(part)
                await asyncio.sleep(0.5) # Небольшая задержка между частями
        else:
            await message.answer(final_response)
            
        logger.info(f"✅ Ответ отправлен пользователю {user_name} (модель: {model_info['display_name']}, длина: {len(final_response)} символов)")
        
    except Exception as e:
        logger.error(f"💥 Ошибка при обработке сообщения от {user_name}: {str(e)}")
        if status_msg:
            try:
                await status_msg.delete()
            except:
                pass
        await message.answer("❌ Произошла ошибка при обработке вашего сообщения. Попробуйте позже.")

async def main():
    logger.info("🚀 Запуск Telegram бота с Nous Research API (Версия 3.0, Unleashed)...")
    logger.info(f"Bot Token: {BOT_TOKEN[:10]}...")
    logger.info(f"API Key: {NOUS_API_KEY[:10]}...")
    
    logger.info("✅ Настройки моделей обновлены для максимального качества:")
    logger.info(f"   • Hermes 405B: max_tokens={AVAILABLE_MODELS['hermes405b']['max_tokens']}, temperature={AVAILABLE_MODELS['hermes405b']['temperature']}")
    logger.info("   • Агрессивное урезание запросов отключено.")
    
    try:
        bot_info = await bot.get_me()
        logger.info(f"✅ Бот успешно авторизован: @{bot_info.username}")
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("🔄 Webhook удален, начинаем поллинг...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске бота: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {str(e)}")
