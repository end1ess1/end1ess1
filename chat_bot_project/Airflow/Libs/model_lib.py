from dataclasses import dataclass, fields
import argparse
import requests
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class MetaClass(type):
    def __new__(cls, name, bases, dct):
        old = super().__new__(cls, name, bases, dct)
        return dataclass(old)


class ModelArgs(metaclass=MetaClass):
    __description__ = {
        "llama_server_path": "Путь до сервера",
        "model_path": "Путь до модели",
        "ngl": "Number of GPU Layers - кол-во слоев",
        "context_size": "Контекстное окно, которое модель учитывает  при генерации ответа",
        "host": "Порт",
        "port": "Хост",
    }

    __mapping__ = {int: int, str: str, dict: str, list: str, bool: bool}

    llama_server_path: str
    model_path: str
    ngl: str
    context_size: str
    host: str
    port: str

    @classmethod
    def parse_base_args(cls):
        parser = argparse.ArgumentParser("Параметры для модели")
        for field in fields(cls):
            parser.add_argument(
                f"--{field.name}", type=field.type, help=cls.__description__[field.name]
            )
        return parser.parse_args()


class Model(metaclass=MetaClass):
    model_type: str = "notLocal"

    def _normalize_embedding(self, embedding):
        norm_embedding = embedding / np.linalg.norm(embedding)
        return norm_embedding

    def get_embedding(self, text: str) -> list:
        if self.model_type == "local":
            response = requests.post(self.local_url, json={"content": text})
            embedding = response.json()[0].get("embedding")[0]
            norm_embedding = self._normalize_embedding(embedding)

            return norm_embedding

        else:
            client = OpenAI(
                api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL")
            )
            embedding = client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL"), input=text, encoding_format="float"
            )
            embedding = embedding.data[0].embedding
            norm_embedding = self._normalize_embedding(embedding)

            return embedding

    def get_answer(
        self,
        question,
        prompt_start,
        message_prep,
        document: str = None,
    ):
        if self.model_type == "local":
            base_url = os.getenv("LOCAL_URL")
            api_key = os.getenv("LOCAL_API_KEY")
        else:
            base_url = os.getenv("BASE_URL")
            api_key = os.getenv("API_KEY")

        client = OpenAI(base_url=base_url, api_key=api_key)

        if prompt_start == os.getenv("PROMPT_MAIN"):
            message = message_prep.format(document=document, question=question)
        else:
            message = message_prep.format(question=question)

        print(f"PROMPT: {prompt_start}")
        print(f"MESSAGE_USER: {message}")

        completion = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[
                {"role": "system", "content": prompt_start},
                {
                    "role": "user",
                    "content": message,
                },
            ],
        )

        answer = completion.choices[0].message.content

        return answer
