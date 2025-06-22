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
from datetime import datetime, timedelta
from collections import defaultdict, deque

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

# [НОВОЕ] Настройки памяти
MEMORY_SETTINGS = {
    "max_messages_per_user": 100,  # Максимум сообщений в памяти
    "context_cleanup_hours": 5,   # Очистка старых контекстов через 5 часов
    "max_context_tokens": 8000,   # Примерная оценка токенов для контекста
}

# [НОВОЕ] Хранилище памяти пользователей
# Структура: user_id -> deque([{"role": "user/assistant", "content": "text", "timestamp": datetime}, ...])
user_memory = defaultdict(lambda: deque(maxlen=MEMORY_SETTINGS["max_messages_per_user"]))
user_memory_timestamps = {}  # Для отслеживания времени последнего обновления

# Доступные модели
AVAILABLE_MODELS = {
    "deephermes": {
        "name": "DeepHermes-3-Mistral-24B-Preview",
        "display_name": "🧠 Шнырь 24B (Быстрый и глубокий)",
        "context": "32k",
        "description": "Быстрая модель, демонстрирующая процесс мышления",
        "system_prompt": "You are Шнырь, a deep thinking AI assistant engaged in an ongoing conversation. You remember previous messages in this conversation and can refer to them. You may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.",
        "max_tokens": 2048,
        "temperature": 0.7,
        "timeout": 240.0
    },
    "hermes405b": {
        "name": "Hermes-3-Llama-3.1-405B",
        "display_name": "🚀 Шнырь 405B (Самый сильный)",
        "context": "32k", 
        "description": "Самая мощная модель для сложных задач с глубоким анализом",
        "system_prompt": "You are Шнырь, one of the most powerful AI assistants in the world, engaged in an ongoing conversation with the user. You have access to the conversation history and should use it to provide contextual, coherent responses. Your goal is to provide deeply reasoned, comprehensive, and accurate answers that build upon the previous conversation. Before providing the final response, you can use a long chain of thought to analyze the problem from multiple angles. Be thorough and detailed while maintaining conversation continuity.",
        "max_tokens": 2048,
        "temperature": 0.6,
        "timeout": 240.0
    }
}

# Настройки пользователей: модель, режим отладки
user_settings = {}

# Простая статистика и rate limiting
user_stats = {}
user_last_request = {}
RATE_LIMIT_SECONDS = 3

# [НОВОЕ] Функции управления памятью
def add_to_memory(user_id: int, role: str, content: str):
    """Добавить сообщение в память пользователя"""
    timestamp = datetime.now()
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp
    }
    user_memory[user_id].append(message)
    user_memory_timestamps[user_id] = timestamp
    logger.info(f"💾 Добавлено в память пользователя {user_id}: {role} - {len(content)} символов")

def get_user_context(user_id: int) -> list:
    """Получить контекст пользователя для API"""
    cleanup_old_contexts()
    
    messages = list(user_memory[user_id])
    if not messages:
        return []
    
    # Оценка токенов и обрезка при необходимости
    context_messages = []
    total_chars = 0
    
    # Идем с конца (самые новые сообщения)
    for message in reversed(messages):
        message_chars = len(message["content"])
        if total_chars + message_chars > MEMORY_SETTINGS["max_context_tokens"] * 4:  # ~4 символа на токен
            break
        context_messages.insert(0, {
            "role": message["role"],
            "content": message["content"]
        })
        total_chars += message_chars
    
    logger.info(f"📖 Загружен контекст для пользователя {user_id}: {len(context_messages)} сообщений, ~{total_chars} символов")
    return context_messages

def cleanup_old_contexts():
    """Очистка старых контекстов"""
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=MEMORY_SETTINGS["context_cleanup_hours"])
    
    users_to_cleanup = []
    for user_id, last_update in user_memory_timestamps.items():
        if last_update < cutoff_time:
            users_to_cleanup.append(user_id)
    
    for user_id in users_to_cleanup:
        if user_id in user_memory:
            del user_memory[user_id]
        if user_id in user_memory_timestamps:
            del user_memory_timestamps[user_id]
        logger.info(f"🧹 Очищен старый контекст пользователя {user_id}")

def clear_user_memory(user_id: int):
    """Очистить память пользователя"""
    if user_id in user_memory:
        user_memory[user_id].clear()
    if user_id in user_memory_timestamps:
        del user_memory_timestamps[user_id]
    logger.info(f"🗑️ Память пользователя {user_id} очищена")

def get_memory_stats(user_id: int) -> dict:
    """Получить статистику памяти пользователя"""
    messages = list(user_memory[user_id])
    if not messages:
        return {"count": 0, "oldest": None, "newest": None}
    
    return {
        "count": len(messages),
        "oldest": messages[0]["timestamp"],
        "newest": messages[-1]["timestamp"]
    }

# Остальные вспомогательные функции остаются прежними
def get_user_model(user_id: int) -> str:
    return user_settings.get(user_id, {}).get('model', 'deephermes')

def set_user_model(user_id: int, model: str):
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id]['model'] = model

def get_user_debug_mode(user_id: int) -> bool:
    return user_settings.get(user_id, {}).get('debug', False)

def set_user_debug_mode(user_id: int, debug: bool):
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id]['debug'] = debug

def check_rate_limit(user_id: int) -> bool:
    current_time = time.time()
    last_request = user_last_request.get(user_id, 0)
    
    if current_time - last_request < RATE_LIMIT_SECONDS:
        return False
    
    user_last_request[user_id] = current_time
    return True

def get_time_until_next_request(user_id: int) -> int:
    current_time = time.time()
    last_request = user_last_request.get(user_id, 0)
    remaining = RATE_LIMIT_SECONDS - (current_time - last_request)
    return max(0, int(remaining))

def create_model_keyboard() -> InlineKeyboardMarkup:
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

# [ИЗМЕНЕНО] Функция вызова API с поддержкой контекста
async def call_nous_api(user_message: str, user_id: int, retry_count: int = 0) -> str:
    """
    Функция отправки запроса к API Nous Research с поддержкой контекста
    """
    headers = {
        "Authorization": f"Bearer {NOUS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    user_model = get_user_model(user_id)
    model_config = AVAILABLE_MODELS[user_model]
    
    # Упрощение только при повторной попытке
    if len(user_message) > 1000 and retry_count > 0:
        user_message = user_message[:800] + "... (сообщение сокращено для повторной попытки)"

    system_prompt = model_config["system_prompt"]
    if retry_count > 0:
        system_prompt = "You are a helpful AI assistant. Answer briefly and clearly based on the conversation context."
    
    max_tokens = model_config["max_tokens"]
    temperature = model_config["temperature"]
    
    if retry_count > 0:
        max_tokens = min(max_tokens, 512)
        temperature = 0.2
    
    # [НОВОЕ] Построение сообщений с контекстом
    messages = [{"role": "system", "content": system_prompt}]
    
    # Добавляем контекст предыдущих сообщений
    context_messages = get_user_context(user_id)
    messages.extend(context_messages)
    
    # Добавляем текущее сообщение пользователя
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": model_config["name"],
        "messages": messages,
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
            logger.info(f"Запрос к {model_config['display_name']} для пользователя {user_id} (попытка {retry_count + 1}) с контекстом: {len(context_messages)} сообщений")
            
            response = await client.post(NOUS_API_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data['choices'][0]['message']['content'].strip()
                
                # [НОВОЕ] Сохраняем сообщения в память
                add_to_memory(user_id, "user", user_message)
                add_to_memory(user_id, "assistant", ai_response)
                
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
    import re
    think_pattern = r'<think>.*?</think>'
    cleaned = re.sub(think_pattern, '', response, flags=re.DOTALL)
    return cleaned.strip()

# [НОВОЕ] Команда для управления памятью
def create_memory_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру для управления памятью"""
    keyboard = [
        [InlineKeyboardButton(text="📊 Статистика памяти", callback_data="memory_stats")],
        [InlineKeyboardButton(text="🗑️ Очистить память", callback_data="memory_clear")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

@dp.message(Command("memory"))
async def cmd_memory(message: Message):
    """Команда для управления памятью"""
    user_id = message.from_user.id
    memory_stats = get_memory_stats(user_id)
    
    if memory_stats["count"] == 0:
        text = """
🧠 **Управление памятью**

📭 **Память пуста**
Пока нет сохраненных сообщений. Начните общение, и бот будет запоминать контекст разговора!

💡 **Как работает память:**
• Запоминается до {max_messages} последних сообщений
• Контекст очищается через {cleanup_hours} часа бездействия
• Память помогает боту понимать контекст разговора

Выберите действие:
        """.format(
            max_messages=MEMORY_SETTINGS["max_messages_per_user"],
            cleanup_hours=MEMORY_SETTINGS["context_cleanup_hours"]
        )
    else:
        oldest_str = memory_stats["oldest"].strftime("%H:%M:%S")
        newest_str = memory_stats["newest"].strftime("%H:%M:%S")
        
        text = f"""
🧠 **Управление памятью**

📊 **Текущая статистика:**
• Сообщений в памяти: {memory_stats["count"]}/{MEMORY_SETTINGS["max_messages_per_user"]}
• Самое старое: {oldest_str}
• Самое новое: {newest_str}

💡 **Настройки памяти:**
• Максимум сообщений: {MEMORY_SETTINGS["max_messages_per_user"]}
• Автоочистка через: {MEMORY_SETTINGS["context_cleanup_hours"]} часа
• Лимит контекста: ~{MEMORY_SETTINGS["max_context_tokens"]} токенов

Выберите действие:
        """
    
    await message.answer(text, reply_markup=create_memory_keyboard())

@dp.callback_query(lambda c: c.data.startswith('memory_'))
async def process_memory_action(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    action = callback_query.data.split('_')[1]
    
    if action == "stats":
        memory_stats = get_memory_stats(user_id)
        messages = list(user_memory[user_id])
        
        if memory_stats["count"] == 0:
            text = "📭 Память пуста"
        else:
            # Детальная статистика
            user_messages = sum(1 for m in messages if m["role"] == "user")
            assistant_messages = sum(1 for m in messages if m["role"] == "assistant")
            total_chars = sum(len(m["content"]) for m in messages)
            
            text = f"""
📊 **Детальная статистика памяти**

💬 **Сообщения:**
• Всего: {memory_stats["count"]}
• От пользователя: {user_messages}
• От ассистента: {assistant_messages}

📝 **Размер:**
• Всего символов: {total_chars:,}
• Среднее на сообщение: {total_chars // memory_stats["count"] if memory_stats["count"] > 0 else 0}

⏰ **Время:**
• Период: {memory_stats["oldest"].strftime("%H:%M:%S")} - {memory_stats["newest"].strftime("%H:%M:%S")}
• Длительность: {str(memory_stats["newest"] - memory_stats["oldest"]).split('.')[0]}
            """
        
        await callback_query.message.edit_text(text)
        await callback_query.answer()
        
    elif action == "clear":
        clear_user_memory(user_id)
        text = """
🗑️ **Память очищена!**

Вся история разговора удалена. Следующий запрос начнет новый диалог без контекста предыдущих сообщений.

💡 Память будет снова накапливаться по мере общения.
        """
        await callback_query.message.edit_text(text)
        await callback_query.answer("✅ Память очищена")

# Обновленные команды
@dp.message(Command("start"))
async def cmd_start(message: Message):
    user_name = message.from_user.first_name or "Друг"
    user_id = message.from_user.id
    current_model = AVAILABLE_MODELS[get_user_model(user_id)]
    
    welcome_text = f"""
🤖 **Привет, {user_name}!** Бобро поржаловать!

🧠 **Текущая модель:** {current_model['display_name']}

**🆕 Новые возможности:**
• 💾 **Память разговора** - бот запоминает контекст диалога
• 🔄 **Непрерывный диалог** - можете ссылаться на предыдущие сообщения
• 📊 **Управление памятью** - команда /memory

**📋 Доступные модели:**
• 🧠 **Шнырь 24B** - Быстрая, показывает процесс мышления
• 🚀 **Шнырь 405B** - Максимально мощная для сложных задач

**⚡ Основные команды:**
/start - это сообщение
/model - выбрать модель  
/memory - управление памятью
/debug - режим отладки
/stats - статистика
/help - подробная помощь

Просто начните общение! Бот будет помнить контекст разговора 🚀
    """
    await message.answer(welcome_text)

@dp.message(Command("model"))
async def cmd_model(message: Message):
    user_id = message.from_user.id
    current_model = AVAILABLE_MODELS[get_user_model(user_id)]
    
    text = f"""
🤖 **Выбор модели AI**

**Текущая модель:** {current_model['display_name']}

**🧠 Шнырь 24B (Быстрый и глубокий)**
• ⚡ Скорость: ~1-5 сек
• 🎯 Особенность: Показывает процесс мышления
• 👍 Лучше для: Большинства задач, быстрые рассуждения
• 💾 Поддержка памяти: Полная

**🚀 Шнырь 405B (Самый сильный)**  
• ⚡ Скорость: ~2-30 сек
• 🎯 Особенность: Высочайшее качество ответов
• 👍 Лучше для: Сложных задач, глубокого анализа
• 💾 Поддержка памяти: Полная

**💡 Память работает с обеими моделями!** Бот запоминает ваш разговор независимо от выбранной модели.

Выберите модель:
    """
    await message.answer(text, reply_markup=create_model_keyboard())

@dp.callback_query(lambda c: c.data.startswith('model_'))
async def process_model_selection(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    model_key = callback_query.data.split('_')[1]
    
    if model_key in AVAILABLE_MODELS:
        set_user_model(user_id, model_key)
        model_info = AVAILABLE_MODELS[model_key]
        
        extra_info = ""
        if model_key == "hermes405b":
            extra_info = "\n\n**⚠️ Внимание:** Эта модель значительно медленнее, но предоставляет ответы высочайшего качества."
        
        text = f"""
✅ **Модель изменена!**

**Выбрана:** {model_info['display_name']}
**Описание:** {model_info['description']}
**Поддержка памяти:** ✅ Полная{extra_info}

💾 **Ваша память сохранена** - новая модель получит доступ ко всему контексту разговора!

Продолжайте общение! 🚀
        """
        
        await callback_query.message.edit_text(text)
        await callback_query.answer(f"✅ Переключено на {model_info['display_name']}")
    else:
        await callback_query.answer("❌ Неизвестная модель")

@dp.callback_query(lambda c: c.data == 'cancel')
async def process_cancel(callback_query: CallbackQuery):
    await callback_query.message.delete()
    await callback_query.answer("Отменено")

@dp.message(Command("debug"))
async def cmd_debug(message: Message):
    user_id = message.from_user.id
    current_debug = get_user_debug_mode(user_id)
    new_debug = not current_debug
    set_user_debug_mode(user_id, new_debug)
    
    if new_debug:
        text = """
🔧 **Режим отладки ВКЛЮЧЕН**

Теперь для модели Шнырь 24B будут показываться теги `<think>` с процессом размышления модели.

💡 Это поможет понять, как модель анализирует ваши запросы и приходит к ответам.

⚠️ На модель Шнырь 405B режим отладки не влияет.
        """
    else:
        text = """
🔧 **Режим отладки ВЫКЛЮЧЕН**

Теги `<think>` больше не будут показываться. Вы будете видеть только финальные ответы модели.
        """
    
    await message.answer(text)

@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = """
📚 **Подробная помощь**

🆕 **Память и контекст:**
• Бот запоминает до 100 последних сообщений
• Может ссылаться на предыдущие части разговора
• Память очищается через 5 часов бездействия
• Команда /memory для управления памятью

🎯 **Возможности:**
• Решение задач с пошаговым анализом
• Программирование на любых языках  
• Креативный контент (стихи, рассказы)
• Объяснение сложных концепций
• Анализ данных и выводы
• Помощь с учебными заданиями

🤖 **Модели с памятью:**
• **Шнырь 24B** - Быстрый, показывает мышление
• **Шнырь 405B** - Максимальная мощность

💡 **Примеры использования памяти:**
• "Продолжи рассказ, который ты начал"
• "А что насчет того алгоритма, который мы обсуждали?"
• "Можешь исправить код из предыдущего ответа?"

🔧 **Команды:**
/start - Главное меню
/model - Выбор модели
/memory - Управление памятью  
/debug - Режим отладки
/stats - Статистика
/help - Эта справка

⚙️ **Лимиты:**
• 1 запрос в 3 секунды
• До 100 сообщений в памяти
• Контекст ~8000 токенов
    """
    await message.answer(help_text)

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    user_id = message.from_user.id
    count = user_stats.get(user_id, 0)
    current_model_key = get_user_model(user_id)
    current_model = AVAILABLE_MODELS[current_model_key]
    debug_mode = get_user_debug_mode(user_id)
    memory_stats = get_memory_stats(user_id)
    last_request_time = user_last_request.get(user_id, 0)
    last_request_str = "Никогда" if last_request_time == 0 else datetime.fromtimestamp(last_request_time).strftime("%H:%M:%S")
    
   # Продолжение кода с того места, где он оборвался в функции cmd_stats:

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
• Режим отладки: {'🟢 Включен' if debug_mode else '🔴 Выключен'}

💾 **Память:**
• Сообщений в памяти: {memory_stats["count"]}/{MEMORY_SETTINGS["max_messages_per_user"]}
• Статус: {'🟢 Активна' if memory_stats["count"] > 0 else '🔴 Пуста'}
• Время существования: {str(memory_stats["newest"] - memory_stats["oldest"]).split('.')[0] if memory_stats["count"] > 0 else 'N/A'}

⚡ **О боте:**
• Версия: 4.0 (Memory Edition)
• API: Nous Research
• Память: {MEMORY_SETTINGS["max_messages_per_user"]} сообщений, {MEMORY_SETTINGS["context_cleanup_hours"]}ч
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
    
    # Получаем статистику памяти для логирования
    memory_stats = get_memory_stats(user_id)
    
    logger.info(f"📝 Запрос #{user_stats[user_id]} от {user_name} (ID: {user_id}, модель: {model_info['display_name']}, память: {memory_stats['count']} сообщений): {user_message[:50]}...")
    
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    # Статусные сообщения с учетом памяти
    status_msg = None
    context_info = f" (с контекстом {memory_stats['count']} сообщений)" if memory_stats['count'] > 0 else ""
    
    if user_model == "hermes405b":
        status_msg = await message.answer(f"""
🚀 **Модель {model_info['display_name']}** приняла ваш запрос в работу{context_info}.

Это самая мощная модель, поэтому ей нужно больше времени на глубокий анализ с учетом всего контекста разговора.

⏱ **Ожидаемое время ответа: 2-30 сек.**
☕️ Можете пока сделать себе чай. Качественный ответ с учетом всей истории диалога стоит того, чтобы подождать!
        """)
    elif len(user_message) > 300 or memory_stats['count'] > 5:
        status_msg = await message.answer(f"🧠 **{model_info['display_name']}** анализирует ваш запрос{context_info}...\n⏱ Ожидайте до 1 минуты")
    
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
        
        # Добавляем префикс с информацией о модели и памяти
        memory_indicator = ""
        if memory_stats['count'] > 0:
            memory_indicator = f" 💾 ({memory_stats['count']} в памяти)"
        
        model_prefix = f"🤖 **Ответ от {model_info['display_name']}:**{memory_indicator}\n\n"
        final_response = model_prefix + final_response
        
        # Разбиение длинных сообщений
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
                await asyncio.sleep(0.5)
        else:
            await message.answer(final_response)
            
        logger.info(f"✅ Ответ отправлен пользователю {user_name} (модель: {model_info['display_name']}, длина: {len(final_response)} символов, память: {memory_stats['count']+2} сообщений)")
        
    except Exception as e:
        logger.error(f"💥 Ошибка при обработке сообщения от {user_name}: {str(e)}")
        if status_msg:
            try:
                await status_msg.delete()
            except:
                pass
        await message.answer("❌ Произошла ошибка при обработке вашего сообщения. Попробуйте позже.")

async def main():
    logger.info("🚀 Запуск Telegram бота с Nous Research API (Версия 4.0, Memory Edition)...")
    logger.info(f"Bot Token: {BOT_TOKEN[:10]}...")
    logger.info(f"API Key: {NOUS_API_KEY[:10]}...")
    
    logger.info("🧠 Настройки памяти:")
    logger.info(f"   • Максимум сообщений на пользователя: {MEMORY_SETTINGS['max_messages_per_user']}")
    logger.info(f"   • Автоочистка через: {MEMORY_SETTINGS['context_cleanup_hours']} часов")
    logger.info(f"   • Лимит контекста: {MEMORY_SETTINGS['max_context_tokens']} токенов")
    
    logger.info("✅ Настройки моделей с поддержкой памяти:")
    logger.info(f"   • Hermes 405B: max_tokens={AVAILABLE_MODELS['hermes405b']['max_tokens']}, temperature={AVAILABLE_MODELS['hermes405b']['temperature']}")
    logger.info(f"   • DeepHermes 24B: max_tokens={AVAILABLE_MODELS['deephermes']['max_tokens']}, temperature={AVAILABLE_MODELS['deephermes']['temperature']}")
    logger.info("   • Поддержка контекста включена для всех моделей.")
    
    try:
        bot_info = await bot.get_me()
        logger.info(f"✅ Бот успешно авторизован: @{bot_info.username}")
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("🔄 Webhook удален, начинаем поллинг...")
        
        # Запуск периодической очистки памяти
        asyncio.create_task(periodic_cleanup())
        
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске бота: {str(e)}")
        raise

async def periodic_cleanup():
    """Периодическая очистка старых контекстов"""
    while True:
        try:
            await asyncio.sleep(3600)  # Проверка каждый час
            cleanup_old_contexts()
            logger.info("🧹 Выполнена периодическая очистка памяти")
        except Exception as e:
            logger.error(f"❌ Ошибка при периодической очистке: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {str(e)}")
