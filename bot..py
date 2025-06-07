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

# Доступные модели - ОПТИМИЗИРОВАННЫЕ НАСТРОЙКИ
AVAILABLE_MODELS = {
    "deephermes": {
        "name": "DeepHermes-3-Mistral-24B-Preview",
        "display_name": "🧠 DeepHermes 24B (Быстрая)",
        "context": "32k",
        "description": "Быстрая модель с глубоким мышлением",
        "system_prompt": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.",
        "max_tokens": 1024,
        "temperature": 0.7,
        "timeout": 120.0
    },
    "hermes405b": {
        "name": "Hermes-3-Llama-3.1-405B",
        "display_name": "🚀 Hermes 405B (Мощная)",
        "context": "32k", 
        "description": "Самая мощная модель для сложных задач",
        # УПРОЩЕННЫЙ системный промпт для ускорения
        "system_prompt": "You are Hermes 3, a helpful AI assistant. Provide clear, concise, and accurate responses.",
        # УМЕНЬШЕННЫЙ max_tokens для быстрого ответа
        "max_tokens": 512,
        # ПОНИЖЕННАЯ температура для стабильности
        "temperature": 0.3,
        # УВЕЛИЧЕННЫЙ таймаут
        "timeout": 200.0
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

def optimize_message_for_405b(message: str) -> str:
    """Оптимизирует сообщение для модели 405B"""
    # Для 405B сокращаем длинные сообщения сразу
    if len(message) > 300:
        return message[:250] + "... (сообщение сокращено для ускорения обработки)"
    return message

async def call_nous_api(user_message: str, user_id: int, retry_count: int = 0) -> str:
    """
    ОПТИМИЗИРОВАННАЯ функция отправки запроса к API Nous Research
    """
    headers = {
        "Authorization": f"Bearer {NOUS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Получаем настройки пользователя
    user_model = get_user_model(user_id)
    model_config = AVAILABLE_MODELS[user_model]
    
    # СПЕЦИАЛЬНАЯ ОПТИМИЗАЦИЯ ДЛЯ 405B
    if user_model == "hermes405b":
        user_message = optimize_message_for_405b(user_message)
        # Дополнительное сокращение при повторной попытке
        if retry_count > 0:
            user_message = user_message[:150] + "... (упрощенный запрос)"
    elif len(user_message) > 500 and retry_count > 0:
        # Для других моделей сокращаем только при повторе
        user_message = user_message[:400] + "... (сообщение сокращено)"
    
    # Системный промпт - упрощаем при повторной попытке
    system_prompt = model_config["system_prompt"]
    if retry_count > 0:
        system_prompt = "You are a helpful AI assistant. Answer briefly and clearly."
    
    # ОПТИМИЗИРОВАННЫЙ payload для 405B
    max_tokens = model_config["max_tokens"]
    temperature = model_config["temperature"]
    
    # Дополнительные ограничения при повторной попытке
    if retry_count > 0:
        max_tokens = min(max_tokens, 256)
        temperature = 0.1  # Максимально детерминированный ответ
    
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
        # ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ СТАБИЛЬНОСТИ
        "top_p": 0.9 if retry_count == 0 else 0.5,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    try:
        # АДАПТИВНЫЕ ТАЙМАУТЫ
        timeout_read = model_config["timeout"]
        if retry_count > 0:
            timeout_read = min(timeout_read, 60.0)  # Сокращаем при повторе
            
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
                    # Пробуем с более простым запросом
                    return await call_nous_api("Ответь кратко на вопрос: " + user_message[:100], user_id, retry_count + 1)
                return "❌ Ошибка в формате запроса. Попробуйте переформулировать вопрос."
                
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return f"❌ Ошибка сервера API ({response.status_code}). Попробуйте позже."
                
    except httpx.TimeoutException:
        logger.error(f"⏰ Timeout при обращении к {model_config['display_name']} (попытка {retry_count + 1})")
        
        if retry_count < 1:  # Одна повторная попытка
            logger.info("🔄 Повторная попытка с упрощенным запросом...")
            return await call_nous_api(user_message, user_id, retry_count + 1)
            
        # Специальное сообщение для 405B
        if user_model == "hermes405b":
            return f"⏰ **{model_config['display_name']} перегружена**\n\nМодель слишком долго обрабатывает запрос. Попробуйте:\n• Задать более простой/короткий вопрос\n• Переключиться на DeepHermes 24B (/model)\n• Повторить запрос через минуту"
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
    
    # Убираем теги <think> для обычных пользователей (оставляем только финальный ответ)
    think_pattern = r'<think>.*?</think>'
    cleaned = re.sub(think_pattern, '', response, flags=re.DOTALL)
    
    return cleaned.strip()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """
    Обработчик команды /start
    """
    user_name = message.from_user.first_name or "Друг"
    user_id = message.from_user.id
    current_model = AVAILABLE_MODELS[get_user_model(user_id)]
    
    welcome_text = f"""
🤖 **Привет, {user_name}!** Добро пожаловать в AI-бота на базе Nous Research!

🧠 **Текущая модель:** {current_model['display_name']}

**📋 Модели:**
• 🧠 **DeepHermes 24B** - Быстрая, глубокое мышление
• 🚀 **Hermes 405B** - Мощная, для сложных задач (оптимизирована)

**💡 Возможности:**
• Математика, физика, программирование
• Анализ и рассуждения  
• Творческие задания
• Помощь с учебой и работой

**⚡ Команды:**
/start - это сообщение
/model - выбрать модель
/debug - режим отладки
/stats - статистика  
/help - подробная помощь

**🔧 Оптимизации для 405B:**
✅ Сокращенные ответы для скорости
✅ Улучшенная обработка таймаутов
✅ Автоматическая оптимизация запросов

Начинайте общение! 🚀
    """
    await message.answer(welcome_text)

@dp.message(Command("model"))
async def cmd_model(message: Message):
    """
    Обработчик команды выбора модели
    """
    user_id = message.from_user.id
    current_model = AVAILABLE_MODELS[get_user_model(user_id)]
    
    text = f"""
🤖 **Выбор модели AI**

**Текущая модель:** {current_model['display_name']}

**🧠 DeepHermes 24B (Быстрая)**
• ⚡ Скорость: Высокая (~30-60 сек)
• 🧠 Особенность: Показывает процесс мышления в <think>
• 💭 Лучше для: Рассуждения, анализ, большинство задач
• 🎯 Токены: до 1024

**🚀 Hermes 405B (Мощная) - ОПТИМИЗИРОВАНА**
• ⚡ Скорость: Средняя (~60-120 сек)
• 🎯 Особенность: Максимальное качество ответов
• 💭 Лучше для: Сложные вопросы, креатив, анализ
• 🎯 Токены: до 512 (оптимизировано)
• ✅ Автосокращение длинных запросов

**💡 Совет:** Начните с DeepHermes, переключайтесь на 405B для особо сложных задач.

Выберите модель:
    """
    
    await message.answer(text, reply_markup=create_model_keyboard())

@dp.callback_query(lambda c: c.data.startswith('model_'))
async def process_model_selection(callback_query: CallbackQuery):
    """
    Обработчик выбора модели
    """
    user_id = callback_query.from_user.id
    model_key = callback_query.data.split('_')[1]
    
    if model_key in AVAILABLE_MODELS:
        set_user_model(user_id, model_key)
        model_info = AVAILABLE_MODELS[model_key]
        
        # Специальное сообщение для 405B
        extra_info = ""
        if model_key == "hermes405b":
            extra_info = "\n\n⚡ **Оптимизация включена:**\n• Ускоренная обработка\n• Сокращение длинных запросов\n• Улучшенная стабильность"
        
        text = f"""
✅ **Модель изменена!**

**Выбрана:** {model_info['display_name']}
**Описание:** {model_info['description']}
**Контекст:** {model_info['context']} токенов
**Максимум токенов:** {model_info['max_tokens']}{extra_info}

Теперь можете задавать вопросы! 🚀
        """
        
        await callback_query.message.edit_text(text)
        await callback_query.answer(f"✅ Переключено на {model_info['display_name']}")
    else:
        await callback_query.answer("❌ Неизвестная модель")

@dp.callback_query(lambda c: c.data == 'cancel')
async def process_cancel(callback_query: CallbackQuery):
    """
    Обработчик отмены
    """
    await callback_query.message.delete()
    await callback_query.answer("Отменено")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """
    Обработчик команды /help
    """
    help_text = """
📚 **Подробная помощь**

🎯 **Возможности:**
• Решение задач с пошаговым анализом
• Программирование на любых языках
• Креативный контент (стихи, рассказы)
• Объяснение сложных концепций
• Анализ данных и выводы
• Помощь с учебными заданиями

🤖 **Модели (ОПТИМИЗИРОВАННЫЕ):**
• **DeepHermes 24B** - Быстрая, показывает мышление
• **Hermes 405B** - Мощная, оптимизирована для скорости

💡 **Примеры запросов:**
• "Объясни квантовую физику простыми словами"
• "Напиши функцию сортировки на Python"  
• "Создай план маркетинговой кампании"
• "Реши: 2x² + 5x - 3 = 0"
• "Короткий рассказ про космос"

🔧 **Команды:**
/start - Главное меню
/model - Выбор модели
/debug - Режим отладки (показать <think>)
/stats - Статистика запросов
/help - Эта справка

⚙️ **Лимиты и оптимизации:**
• Rate limit: 1 запрос в 3 секунды
• 405B: автосокращение длинных запросов
• Адаптивные таймауты
• Повторные попытки при ошибках

🔍 **Режим отладки:**
Команда /debug покажет процесс размышлений DeepHermes в тегах <think>

💡 **Советы для 405B:**
• Задавайте конкретные, четкие вопросы
• Избегайте очень длинных текстов
• При долгом ожидании попробуйте DeepHermes
    """
    await message.answer(help_text)

@dp.message(Command("premium"))
async def cmd_premium(message: Message):
    """
    Информация о премиум возможностях
    """
    premium_text = """
💎 **Информация о боте**

🚀 **Доступно сейчас (ОПТИМИЗИРОВАНО):**
• 2 мощные AI модели с улучшениями
• Режим глубокого мышления
• Оптимизированная работа 405B модели
• Адаптивные таймауты и повторы
• Статистика использования

⭐ **Оптимизации версии 2.0:**
✅ Ускоренная обработка для 405B
✅ Автосокращение длинных запросов  
✅ Улучшенная обработка ошибок
✅ Адаптивные настройки температуры
✅ Интеллектуальные повторные попытки

💰 **Текущие лимиты API:**
• 100 запросов/мин (общие для всех)
• Rate limit: 1 запрос в 3 сек на пользователя
• DeepHermes: ~30-60 сек ответ
• Hermes 405B: ~60-120 сек ответ (оптимизировано)

📊 **Планы развития:**
• Память диалогов между сообщениями
• Работа с изображениями и документами
• Персональные настройки AI
• Расширенная аналитика

📧 **Обратная связь:**
Сообщайте о проблемах и предложениях!
    """
    await message.answer(premium_text)

@dp.message(Command("debug"))
async def cmd_debug(message: Message):
    """
    Переключение режима отладки для показа тегов размышлений
    """
    user_id = message.from_user.id
    current_debug = get_user_debug_mode(user_id)
    new_debug = not current_debug
    set_user_debug_mode(user_id, new_debug)
    
    if new_debug:
        await message.answer("""
🔍 **Режим отладки включен**

Теперь вы будете видеть процесс размышлений AI в тегах <think>

⚠️ **Важно:** Работает только с DeepHermes моделью
🚀 **Hermes 405B** не показывает теги размышлений для ускорения работы
        """)
    else:
        await message.answer("🔍 **Режим отладки выключен**\nТеперь вы будете видеть только финальные ответы")

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    """
    Показать статистику пользователя
    """
    user_id = message.from_user.id
    count = user_stats.get(user_id, 0)
    current_model = AVAILABLE_MODELS[get_user_model(user_id)]
    debug_mode = get_user_debug_mode(user_id)
    
    # Статистика времени
    last_request_time = user_last_request.get(user_id, 0)
    last_request_str = "Никогда" if last_request_time == 0 else datetime.fromtimestamp(last_request_time).strftime("%H:%M:%S")
    
    stats_text = f"""
📊 **Ваша статистика:**

👤 **Пользователь:**
• ID: {user_id}
• Всего запросов: {count}
• Последний запрос: {last_request_str}

🤖 **Настройки:**
• Модель: {current_model['display_name']}
• Max токенов: {current_model['max_tokens']}
• Температура: {current_model['temperature']}
• Режим отладки: {'🟢 Включен' if debug_mode else '🔴 Выключен'}

⚡ **Оптимизации:**
• Rate limit: 1 запрос в 3 сек
• Таймаут: {current_model['timeout']}s
• Статус: 🟢 Активен

🚀 **О боте:**
• Версия: 2.0 (оптимизированная)
• Доступно моделей: {len(AVAILABLE_MODELS)}
• API: Nous Research
    """
    await message.answer(stats_text)

@dp.message()
async def handle_message(message: Message):
    """
    ОПТИМИЗИРОВАННЫЙ обработчик всех текстовых сообщений
    """
    user_message = message.text
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "Пользователь"
    
    # Проверяем rate limiting
    if not check_rate_limit(user_id):
        remaining_time = get_time_until_next_request(user_id)
        await message.answer(f"⏳ Подождите {remaining_time} сек. до следующего запроса")
        return
    
    # Обновляем статистику
    user_stats[user_id] = user_stats.get(user_id, 0) + 1
    
    # Получаем настройки пользователя
    user_model = get_user_model(user_id)
    model_info = AVAILABLE_MODELS[user_model]
    debug_mode = get_user_debug_mode(user_id)
    
    # Логируем запрос
    logger.info(f"📝 Запрос #{user_stats[user_id]} от {user_name} (ID: {user_id}, модель: {model_info['display_name']}): {user_message[:50]}...")
    
    # Отправляем индикатор "печатает"
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    # АДАПТИВНЫЕ сообщения о времени ожидания
    status_msg = None
    if user_model == "hermes405b":
        status_msg = await message.answer(f"""
🚀 **{model_info['display_name']}** обрабатывает запрос...

⏱ **Ожидаемое время:** 1-2 минуты
🔧 **Оптимизации:** Включены  
💡 **Совет:** Для быстрых ответов используйте DeepHermes (/model)
        """)
    elif len(user_message) > 200:
        status_msg = await message.answer(f"🧠 **{model_info['display_name']}** анализирует ваш запрос...\n⏱ Ожидайте до 1 минуты")
    
    try:
        # Получаем ответ от AI
        ai_response = await call_nous_api(user_message, user_id)
        
        # Проверяем на ошибки API лимитов
        if "rate limit" in ai_response.lower() or "429" in ai_response or "превышен лимит" in ai_response.lower():
            if status_msg:
                try:
                    await status_msg.delete()
                except:
                    pass
            await message.answer(ai_response)
            return
        
        # Удаляем статусное сообщение
        if status_msg:
            try:
                await status_msg.delete()  
            except:
                pass
        
        # Проверяем, нужно ли показывать процесс размышлений  
        if debug_mode and user_model == "deephermes":
            # В режиме отладки показываем полный ответ с тегами <think>
            final_response = ai_response
        else:
            # В обычном режиме скрываем теги размышлений
            final_response = clean_ai_response(ai_response)
        
        # Добавляем информацию о модели в начало ответа
        model_prefix = f"🤖 **{model_info['display_name']}:**\n\n"
        final_response = model_prefix + final_response
        
        # Telegram имеет лимит на длину сообщений (4096 символов)
        if len(final_response) > 4000:
            # Разбиваем длинный ответ на части
            parts = []
            for i in range(0, len(final_response), 4000):
                parts.append(final_response[i:i+4000])
            
            for i, part in enumerate(parts):
                if i == 0:
                    await message.answer(f"📝 **Ответ (часть {i+1}/{len(parts)}):**\n\n{part}")
                else:
                    await message.answer(f"📝 **Часть {i+1}/{len(parts)}:**\n\n{part}")
        else:
            await message.answer(final_response)
            
        logger.info(f"✅ Ответ отправлен пользователю {user_name} (модель: {model_info['display_name']}, длина: {len(final_response)} символов)")
        
    except Exception as e:
        logger.error(f"💥 Ошибка при обработке сообщения от {user_name}: {str(e)}")
        # Удаляем статусное сообщение если оно было
        if status_msg:
            try:
                await status_msg.delete()
            except:
                pass
        await message.answer("❌ Произошла ошибка при обработке вашего сообщения. Попробуйте позже или задайте более короткий вопрос.")

async def main():
    """
    Главная функция для запуска бота
    """
    logger.info("🚀 Запуск Telegram бота с Nous Research API (Оптимизированная версия 2.0)...")
    logger.info(f"Bot Token: {BOT_TOKEN[:10]}...")
    logger.info(f"API Key: {NOUS_API_KEY[:10]}...")
    logger.info(f"Доступные модели: {', '.join([model['display_name'] for model in AVAILABLE_MODELS.values()])}")
    
    # Показываем оптимизации
    logger.info("✅ Оптимизации включены:")
    logger.info("   • Hermes 405B: max_tokens=512, temperature=0.3")
    logger.info("   • Автосокращение длинных запросов для 405B")
    logger.info("   • Адаптивные таймауты и повторные попытки")
    logger.info("   • Улучшенная обработка ошибок")
    
    try:
        # Получаем информацию о боте
        bot_info = await bot.get_me()
        logger.info(f"✅ Бот успешно авторизован: @{bot_info.username}")
        
        # Удаляем webhook (если был установлен)
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("🔄 Webhook удален, начинаем поллинг...")
        
        # Запускаем поллинг
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
