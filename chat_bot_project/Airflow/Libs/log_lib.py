from dataclasses import dataclass
import psycopg2.extensions
from datetime import datetime
import redis.client
from typing import Dict, Union


class MetaClass(type):
    def __new__(cls, name, bases, dct):
        old = super().__new__(cls, name, bases, dct)
        return dataclass(old)


class ChatHistoryConfig(metaclass=MetaClass):
    user_id: str
    first_name: str
    last_name: str
    username: str
    chat_id: str
    question: str
    answer: str
    feedback: str
    question_date: datetime
    answer_date: datetime
    language_code: str
    model_version: str
    log_table_name: str


class RedisConfig:
    @staticmethod
    def chat_history_entry(dict_config: Dict[str, Union[str, datetime]]):
        c = ChatHistoryConfig(**dict_config)

        return [
            str(c.question),
            {
                # "user_id": str(c.user_id),
                # "first_name": str(c.first_name),
                # "last_name": str(c.last_name),
                # "username": str(c.username),
                # "chat_id": str(c.chat_id),
                # "question": str(c.question),
                "answer": str(c.answer),
                # "feedback": str(c.feedback),
                # "question_length": str(len(c.question)),
                # "answer_length": str(len(c.answer)),
                # "response_time_s": str(
                #     (c.answer_date - c.question_date).total_seconds()
                # ),
                # "language_code": str(c.language_code),
                # "question_date": str(c.question_date),
                # "answer_date": str(c.answer_date),
            },
        ]

    @staticmethod
    def logs_entry(level: str, comment: str):
        return [
            "last_log",
            {
                "processed_dttm": str(
                    datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
                ),
                "level": level,
                "comment": comment,
            },
        ]


class PostgreConfig:
    @staticmethod
    def logs_table_init(log_table_name: str):
        COMMENTS = {
            "id": "Уникальный идентификатор записи",
            "script_name": "Имя, обязательно для заполнения",
            "processed_dttm": "Дата и время, обязательно для заполнения",
            "level": "Уровень логирования (например: log, error, warning)",
            "comment": "Комментарий (может быть NULL)",
        }

        SQL_SAMPLE = f"""CREATE TABLE IF NOT EXISTS {log_table_name} (
                            id SERIAL PRIMARY KEY,
                            script_name VARCHAR(50) NOT NULL,
                            processed_dttm VARCHAR(50) NOT NULL,
                            level VARCHAR(10) NOT NULL,
                            comment TEXT);
                    """
        return COMMENTS, SQL_SAMPLE

    @staticmethod
    def logs_insert_sql(
        log_table_name: str, script_name: str, level: str, comment: str
    ):
        SQL_SAMPLE = f"""
                        INSERT INTO {log_table_name} (
                            script_name,
                            processed_dttm,
                            level,
                            comment
                            )
                        VALUES (
                            '{script_name}',
                            '{str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))}',
                            '{level}',
                            '{comment}'
                        );"""
        return SQL_SAMPLE

    @staticmethod
    def chat_history_insert_sql(dict_config: Dict[str, Union[str, datetime]]):
        c = ChatHistoryConfig(**dict_config)

        SQL_SAMPLE = f"""
                INSERT INTO {c.log_table_name} (
                    user_id,
                    first_name,
                    last_name,
                    username,
                    chat_id,
                    question,
                    answer,
                    feedback,
                    question_length,
                    answer_length,
                    response_time_s,
                    language_code,
                    question_date,
                    answer_date,
                    model_version
                )
                VALUES (
                    '{c.user_id}',
                    '{str(c.first_name)}',
                    '{str(c.last_name)}',
                    '{str(c.username)}',
                    '{c.chat_id}',
                    '{str(c.question)}',
                    '{str(c.answer)}',
                    '{str(c.feedback)}',
                    '{str(len(c.question))}',
                    '{str(len(c.answer))}',
                    '{(c.answer_date-c.question_date).total_seconds()}',
                    '{str(c.language_code)}',
                    '{c.question_date}',
                    '{c.answer_date}',
                    '{str(c.model_version)}'
                );
            """
        return SQL_SAMPLE


class Log(metaclass=MetaClass):
    postgresql_conn: psycopg2.extensions.connection
    redis_conn: redis.client.Redis
    script_name: str
    log_table_name: str = "scripts_logs"

    def __post_init__(self) -> None:
        self._setup_table()

    def _setup_table(self) -> None:
        with self.postgresql_conn.cursor() as cur:

            comments, sql_init_table = PostgreConfig.logs_table_init(
                self.log_table_name
            )

            cur.execute(sql_init_table)

            for col in comments:
                cur.execute(
                    f"COMMENT ON COLUMN {self.log_table_name}.{col} IS '{comments[col]}'"
                )

            cur.execute(
                f"COMMENT ON TABLE {self.log_table_name} IS 'Таблица для логирования скриптов'"
            )

            self.postgresql_conn.commit()

    def _get_config(self, comment, level):
        sql_insert_logs = PostgreConfig.logs_insert_sql(
            self.log_table_name, self.script_name, level, comment
        )
        redis_mapping = RedisConfig.logs_entry(comment, level)

        return sql_insert_logs, redis_mapping

    def _redis_caching_expire(self, mapping):
        self.redis_conn.hset(name=mapping[0], mapping=mapping[1])
        self.redis_conn.expire(name=mapping[0], time=86400)

    def insert(self, sql_sample: str, redis_mapping: list) -> None:
        with self.postgresql_conn.cursor() as cur:
            cur.execute(sql_sample)

            self.postgresql_conn.commit()

        self._redis_caching_expire(redis_mapping)

    def error(self, comment: str, level: str = "error") -> None:
        self.insert(*self._get_config(comment, level))

    def warning(self, comment: str, level: str = "warning") -> None:
        self.insert(*self._get_config(comment, level))

    def log(self, comment: str, level: str = "log") -> None:
        self.insert(*self._get_config(comment, level))

    def success(self, comment: str, level: str = "success") -> None:
        self.insert(*self._get_config(comment, level))

    def finish(self, comment: str = "Завершение работы", level: str = "finish") -> None:
        self.insert(*self._get_config(comment, level))
        self.postgresql_conn.close()
        self.redis_conn.close()


class ChatHistory(Log):
    def _get_config(self, dict_config):
        sql_insert_logs = PostgreConfig.chat_history_insert_sql(dict_config)
        redis_mapping = RedisConfig.chat_history_entry(dict_config)

        return sql_insert_logs, redis_mapping

    def load_logs(self, dict_config):
        self.insert(*self._get_config(dict_config))


class LogRetriever(metaclass=MetaClass):
    postgresql_conn: psycopg2.extensions.connection
    redis_conn: redis.client.Redis

    def get_logs(self, key: str):
        logs = self.redis_conn.hgetall(key)

        if logs:
            return {k.decode(): v.decode() for k, v in logs.items()}
        else:
            with self.postgresql_conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {Log.log_table_name} WHERE key = {key}")
                logs = cur.fetchall()
            return logs or "Логи не найдены."
