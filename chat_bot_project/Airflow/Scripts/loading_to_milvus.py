import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from pymorphy3 import MorphAnalyzer
from dotenv import load_dotenv
from rich.traceback import install
from tqdm import tqdm

load_dotenv()
install(show_locals=True)
sys.path.append(os.getenv("LIBS_PATH_LOCAL"))

from connection import connect_to_databases
from milvus_lib import MilvusDBClient, DocumentData
from model_lib import Model
from log_lib import Log


def _get_embedded_doc(data: Dict[str, str]) -> DocumentData:
    doc_data = DocumentData(
        text=data["text"],
        dense=Model().get_embedding(data["text"]),
        section=(
            data["metadata"]["section"]
            if data["metadata"]["section"] is not None
            else "None"
        ),
        subsection=(
            data["metadata"]["subsection"]
            if data["metadata"]["subsection"] is not None
            else "None"
        ),
        keywords=(
            data["metadata"]["keywords"]
            if data["metadata"]["keywords"] is not None
            else "None"
        ),
    )
    return doc_data


def get_docs_list(folder: str) -> List[Dict[str, str]]:
    data = []

    for file in os.listdir(folder)[:]:
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as ff:
                data.extend(json.load(ff))

    return data


def load_to_db_and_upd_progress_bar(
    client: MilvusDBClient, doc: Dict[str, str], pbar: tqdm
) -> None:
    emb_doc = _get_embedded_doc(doc)
    client.insert_document(emb_doc)
    pbar.update(1)


def load_to_milvus(
    collection_name: str,
    dimension: str,
    script_name: str,
    docs: List[Dict[str, str]],
    text_len: int = 10000,
) -> None:
    client = None
    LibLog = Log(*connect_to_databases(), script_name=script_name)

    try:
        client = MilvusDBClient(LibLog=LibLog)
        client.connect()
        client.create_collection(
            name=collection_name, dimension=dimension, text_len=text_len
        )
        client.create_index()

        with tqdm(
            total=len(docs), desc="Loading docs to Milvus Database", position=0
        ) as overall_pbar:
            with ThreadPoolExecutor(max_workers=5) as io_executor:
                io_executor.map(
                    lambda doc: load_to_db_and_upd_progress_bar(
                        client, doc, overall_pbar
                    ),
                    docs,
                )

    except Exception as e:
        LibLog.error(f"Error occurred: {e}", exc_info=True)
    finally:
        if client:
            client.close()


def main():
    FOLDER = os.getenv("DOCS_FOLDER_LOCAL")
    DOCS = get_docs_list(FOLDER)
    EMBEDDING_DIMENSION = len(Model().get_embedding("get len embedding"))

    load_to_milvus(
        collection_name="MiigaikDocsInfo",
        dimension=EMBEDDING_DIMENSION,
        script_name="loading_to_milvus.py",
        docs=DOCS,
    )


if __name__ == "__main__":
    main()
