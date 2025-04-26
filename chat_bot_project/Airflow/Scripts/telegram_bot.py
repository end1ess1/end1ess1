import asyncio
import functools
import os
import sys
import traceback
from datetime import datetime
from dotenv import load_dotenv
from pprint import pprint
import redis
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import subprocess

from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from rich.traceback import install
from pymilvus.model.reranker import CrossEncoderRerankFunction

from connection import connect_to_databases

load_dotenv()
install(show_locals=True)
sys.path.append(os.getenv("LIBS_PATH"))
from log_lib import Log, ChatHistory
from milvus_lib import MilvusDBClient
from model_lib import Model


ROUTER = Router()


# Обработка исключений и логирование в БДшки
def handle_errors(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            traceback.print_exc()

    return wrapper


@handle_errors
async def log_message(
    user_id: int,
    first_name: str,
    last_name: str,
    username: str,
    chat_id: str,
    question: str,
    answer: str,
    question_date: datetime,
    answer_date: datetime,
    language_code: str = "ru",
    feedback: str = "None",
):
    __dicting__ = {
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "chat_id": chat_id,
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "question_date": question_date,
        "answer_date": answer_date,
        "language_code": language_code,
        "model_version": os.getenv("MODEL_VERSION"),
        "log_table_name": os.getenv("LOG_TABLE_NAME"),
    }

    ChatHistory.load_logs(__dicting__)
    logging.success("Inserting logs success")


@handle_errors
@ROUTER.message(Command("start"))
async def send_welcome(message: Message):
    logging.log("Приветствие бота")
    await message.answer("Привет! Я бот-помощник Гисик. Задавай любой вопрос!")


async def get_llm_message(chat_id, question, user_id, first_name, last_name, username):
    question_date = datetime.now()

    # Поищи сначала в редисе:
    r = redis.Redis(host="localhost", port=6379, db=3)

    final_answer = r.hget(question, "answer")
    if final_answer:
        logging.log("Вопрос пользователя найден в кеше")
        print("Вопрос пользователя найден в кеше")

        await BOT.send_message(text=final_answer.decode("utf-8"), chat_id=chat_id)

        print(f"Ответ: {final_answer.decode("utf-8")}")

        answer_date = datetime.now()

        data_logs = {
            "question": question,
            "answer": final_answer.decode("utf-8"),
            "question_date": question_date,
            "answer_date": answer_date,
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "chat_id": chat_id,
        }

        await log_message(**data_logs)

        logging.success("Модель ответила на вопрос пользователя")

        return

    agent_answer = model.get_answer(
        question, os.getenv("PROMPT_AGENT"), os.getenv("MESSAGE_AGENT")
    )

    if agent_answer == "Да":
        logging.log("Вопрос пользователя относится к вузу")
        print("Вопрос пользователя относится к вузу")
        # answers = client.search_answer(question=message.text, reranker=reranker)

        answers = client.search_answer(question)
        logging.log("Получили похожие тексты из БД")

        if answers:
            final_answer = model.get_answer(
                question,
                os.getenv("PROMPT_MAIN"),
                os.getenv("MESSAGE_MAIN"),
                "\n".join(
                    answer["text"] for answer in answers[:3] if answer.get("text")
                ),
            )
        else:
            final_answer = "Данные по вашему вопросу, к сожалению, не найдены.\nПопробуйте переформулировать вопрос или задать другой."

        await BOT.send_message(text=final_answer, chat_id=chat_id)

        result = []
        for i, answer in enumerate(answers[:3], start=1):
            result.append(
                f"""
                Результат #{i}:
                Текст: {answer.get('text', 'Нет данных')}
                Раздел: {answer.get('section', 'Нет данных')}
                Статья: {answer.get('article', 'Нет данных')}
                Схожесть: {1 - answer.get('distance', 1):.2%}
                """
            )

        result.append(f"----\n\nОтвет: {final_answer}")

        pprint("".join(result))

        answer_date = datetime.now()

        data_logs = {
            "question": question,
            "answer": final_answer,
            "question_date": question_date,
            "answer_date": answer_date,
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "chat_id": chat_id,
        }

        user_logs[chat_id] = data_logs

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Да", callback_data="feedback_yes")],
                [InlineKeyboardButton(text="Нет", callback_data="feedback_no")],
            ]
        )
        await BOT.send_message(
            text="Был ли Вам полезен мой ответ?",
            reply_markup=keyboard,
            chat_id=chat_id,
        )

    else:
        logging.log("Вопрос пользователя не относится к вузу")
        print("Вопрос пользователя не относится к вузу")
        final_answer = model.get_answer(
            question, os.getenv("PROMPT_COMMON"), os.getenv("MESSAGE_COMMON")
        )

        await BOT.send_message(text=final_answer, chat_id=chat_id)

        print(f"Ответ: {final_answer}")

        answer_date = datetime.now()

        data_logs = {
            "question": question,
            "answer": final_answer,
            "question_date": question_date,
            "answer_date": answer_date,
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "chat_id": chat_id,
        }

        await log_message(**data_logs)

        logging.success("Модель ответила на вопрос пользователя")


@handle_errors
@ROUTER.message(F.text)
async def handle_message(message: Message):
    print(f"Вопрос пользователя: {message.text}")

    user = message.from_user

    await get_llm_message(
        message.chat.id,
        message.text,
        user.id,
        user.first_name,
        user.last_name,
        user.username,
    )


async def convert_ogg_to_wav(ogg_path, wav_path):
    try:
        voice = AudioSegment.from_file(ogg_path, format="ogg")
        voice.export(wav_path, format="wav")
        print(f"Файл успешно конвертирован: {ogg_path} -> {wav_path}")
        return True
    except Exception as e:
        print(f"Ошибка при конвертации через pydub: {e}")
        try:
            command = f"ffmpeg -i {ogg_path} -ac 1 -ar 16000 {wav_path} -y"
            subprocess.call(command, shell=True)
            print(f"Файл конвертирован через ffmpeg: {ogg_path} -> {wav_path}")
            return True
        except Exception as e:
            print(f"Ошибка конвертации файла через ffmpeg: {e}")
            return False


def recognize_speech(wav_file_path, language="ru-RU"):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)
            print("Распознавание речи...")
            text = recognizer.recognize_google(audio_data, language=language)
            return text
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError as e:
        return f"Ошибка сервиса распознавания: {e}"
    except Exception as e:
        return f"Ошибка: {e}"


@ROUTER.message(F.voice)
async def handle_voice(message: Message):
    print(f"Получено голосовое сообщение от {message.from_user.first_name}")

    try:
        if not os.path.exists("voice_files"):
            os.makedirs("voice_files")

        file_id = message.voice.file_id
        file = await BOT.get_file(file_id)
        file_path = file.file_path

        ogg_path = f"voice_files/voice_{file_id}.ogg"
        await BOT.download_file(file_path, ogg_path)
        print(f"Голосовое сообщение сохранено: {ogg_path}")

        wav_path = f"voice_files/voice_{file_id}.wav"

        await message.answer("Обрабатываю ваше голосовое сообщение...")

        if await convert_ogg_to_wav(ogg_path, wav_path):
            recognized_text = recognize_speech(wav_path)

            await message.answer(f"Ваше сообщение: {recognized_text}")
            print(f"Распознанный текст: {recognized_text}")

            user = message.from_user

            await get_llm_message(
                message.chat.id,
                recognized_text,
                user.id,
                user.first_name,
                user.last_name,
                user.username,
            )
        else:
            await message.answer("Не удалось обработать голосовое сообщение")

        # Удаляем временные файлы
        try:
            os.remove(ogg_path)
            os.remove(wav_path)
            print("Временные файлы удалены")
        except Exception as e:
            print(f"Не удалось удалить временные файлы: {e}")

    except Exception as e:
        print(f"Ошибка при обработке голосового сообщения: {e}")
        await message.answer(
            f"Произошла ошибка при обработке голосового сообщения: {e}"
        )


@ROUTER.callback_query(lambda c: c.data.startswith("feedback_"))
async def handle_feedback(callback_query: types.CallbackQuery):
    feedback = (
        "Спасибо за обратную связь! Если будут еще вопросы - я с удовольствием на них отвечу!"
        if callback_query.data == "feedback_yes"
        else "Спасибо за обратную связь! Я уже пополняю базу знаний!"
    )

    user_logs[callback_query.from_user.id].update(
        {"feedback": "Да" if callback_query.data == "feedback_yes" else "Нет"}
    )

    await callback_query.message.answer(feedback)
    await callback_query.answer()
    await log_message(**user_logs[callback_query.from_user.id])
    logging.success("Модель ответила на вопрос пользователя")


async def main():

    DP.include_router(ROUTER)
    await DP.start_polling(BOT)


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    SCRIPT_NAME = "telegram_bot.py"
    user_logs = {}

    logging = Log(*connect_to_databases(), SCRIPT_NAME)
    ChatHistory = ChatHistory(*connect_to_databases(), SCRIPT_NAME)
    logging.success("Подключение к БД успешно!")

    model = Model()
    logging.success("Инициализация модели успешно!")

    client = MilvusDBClient(LibLog=logging)
    client.connect()
    client.create_collection(name="MiigaikDocsInfo", dimension=3072)

    reranker = CrossEncoderRerankFunction(
        model_name=os.getenv("RERANKER_MODEL"),
        device="cpu",
    )

    BOT = Bot(token=os.getenv("API_TOKEN"))
    DP = Dispatcher()

    logging.success("Инициализация Бота успешна")

    asyncio.run(main())
