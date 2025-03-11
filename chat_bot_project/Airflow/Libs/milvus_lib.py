from log_lib import Log
from typing import List, Optional, Dict
from dataclasses import dataclass
from model_lib import Model
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    AnnSearchRequest,
    Function,
    FunctionType,
    RRFRanker,
    MilvusClient,
)
from pymilvus.model.reranker import CrossEncoderRerankFunction
from pymorphy3 import MorphAnalyzer


class CollectionConfig:
    """Конфиг для БД"""

    NAME: str = "Документы"
    DIMENSION: int = 3072
    DESCRIPTION: str = "Эмбеддинги"
    CONSISTENCY_LEVEL: str = "Strong"
    AUTO_ID: bool = True

    @classmethod
    def get_fields(cls, dimension: int, text_len: int = 10000) -> List[FieldSchema]:
        """Поля для коллекции"""

        return [
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=cls.AUTO_ID
            ),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=text_len,
                enable_analyzer=True,
            ),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="subsection", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=2000),
        ]


class IndexConfig:
    """Конфиг для индексов Malvus БД"""

    EMBEDDING_INDEX_NAME = "dense_embedding"
    INDEX_TYPE = "IVF_FLAT"
    METRIC_TYPE = "IP"
    INDEX_PARAMS = {"nlist": 128}


@dataclass
class DocumentData:
    """Дата класс документа"""

    text: str
    dense: List[float]
    section: str
    subsection: str
    keywords: str


class MilvusDBClient:
    """Инициализация Милвус и операции с БД"""

    def __init__(self, LibLog: Log):
        self.logging: Log = LibLog
        self._connection_alias = "default"
        self.collection: Optional[Collection] = None
        self._is_connected = False

    def connect(self, host: str = "localhost", port: str = "19530") -> None:
        """Коннекшн"""
        try:
            connections.connect(alias=self._connection_alias, host=host, port=port)
            self._is_connected = True
            self.client = MilvusClient(
                uri="http://localhost:19530", token="root:Milvus"
            )
            self.logging.log("Подключение к Milvus успешно!")
        except Exception as e:
            self.logging.error(f"Ошибка: {e}")
            raise

    def _validate_connection(self) -> None:
        """Проверка на коннекшн"""
        if not self._is_connected:
            raise ConnectionError("Нет коннекшена к Милвусу")

    def create_collection(
        self, name: str, dimension: int, text_len: int = 10000
    ) -> None:
        """Создание коллекции"""
        self._validate_connection()

        self.dimension = dimension
        self.text_len = text_len

        if utility.has_collection(name):
            self.collection = Collection(name)
            self.logging.log(f"Коллекция уже существует: {name}")

        else:
            schema = CollectionSchema(
                fields=CollectionConfig.get_fields(self.dimension, self.text_len),
                description=CollectionConfig.DESCRIPTION,
            )

            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )

            schema.add_function(bm25_function)

            self.collection = Collection(
                name=name,
                schema=schema,
                consistency_level=CollectionConfig.CONSISTENCY_LEVEL,
            )
            self.logging.log(f"Коллекция {name} успешно создана!")

    def create_index(self) -> None:
        """Cоздание индекса"""
        self._validate_connection()

        if not self.collection:
            raise ValueError("Коллекция не инициализирована")

        if self.collection.has_index():
            self.logging.log("Индекс уже существует")
        else:

            index_params = {
                "index_type": IndexConfig.INDEX_TYPE,
                "metric_type": IndexConfig.METRIC_TYPE,
                "params": IndexConfig.INDEX_PARAMS,
            }

            self.collection.create_index(
                field_name="dense",
                index_params=index_params,
                index_name=IndexConfig.EMBEDDING_INDEX_NAME,
            )

            index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {},
            }

            self.collection.create_index(
                field_name="sparse", index_params=index_params, index_name="sparse_emb"
            )

            self.collection.load()
            self.logging.log("Индекс успешно создан")

    def insert_document(self, document: DocumentData) -> None:
        """
        Вставка дока в коллекцию
        """
        self._validate_connection()

        if not self.collection:
            raise ValueError("Коллекция не инициализирована")

        entities = [
            [document.dense],
            [document.text],
            [document.section],
            [document.subsection],
            [document.keywords],
        ]

        try:
            self.collection.insert(entities)
            self.logging.log("Док успешно вставлен в коллекцию")
        except Exception as e:
            self.logging.error(f"Ошибка вставки: {e}")
            raise

    def _rerank_docs(
        self,
        reranker: CrossEncoderRerankFunction,
        search_results: List[Dict[str, str]],
        question: str,
    ):

        reranked_results = reranker(
            query=question, documents=[c["text"] for c in search_results], top_k=3
        )

        final_results = []
        for reranked in reranked_results:
            for c in search_results:
                if c["text"] == reranked.text:
                    final_results.append({**c, "score": reranked.score})
                    break

        return final_results

    def _lemmatize(self, text):
        words = text.split()
        lemmas = [MorphAnalyzer().parse(word)[0].normal_form for word in words]

        return lemmas

    def search_answer(
        self,
        question: str,
        model_type: str = "notLocal",
        reranker: CrossEncoderRerankFunction = None,
    ):
        embedding = Model(model_type=model_type).get_embedding(question)

        if not embedding or len(embedding) != self.dimension:
            self.logging.warning(f"Некорректный эмбеддинг для вопроса: {question}")
            return []

        if not self.collection:
            raise ValueError("Коллекция не инициализирована")

        try:
            # Полнотекстовый поиск
            req1 = AnnSearchRequest(
                data=[question],
                anns_field="sparse",
                param={"params": {"drop_ratio_search": 0.2}, "metric_type": "BM25"},
                limit=2,
            )

            # Векторное сходство
            req2 = AnnSearchRequest(
                data=[embedding],
                anns_field="dense",
                param={"metric_type": "IP", "params": {"nprobe": 100}},
                limit=2,
            )

            # Гибридный поиск
            results = self.client.hybrid_search(
                collection_name="MiigaikDocsInfo",
                reqs=[req1, req2],
                ranker=RRFRanker(60),
                limit=3,
                output_fields=["text", "section", "keywords"],
            )

            search_results = [
                {
                    "text": hit["entity"].get("text"),
                    "section": hit["entity"].get("section"),
                    "keywords": hit["entity"].get("keywords"),
                    "distance": hit["distance"],
                }
                for hit in results[0]
            ]

            if reranker:
                return self._rerank_docs(reranker, search_results, question)
            else:
                return search_results

        except Exception as e:
            self.logging.warning(f"Ошибка поиска: {str(e)}")
            return []

    def close(self) -> None:
        """Закрыли коннекшн"""
        connections.disconnect(self._connection_alias)
        self._is_connected = False
        self.logging.log("Коннекш закрыт")
