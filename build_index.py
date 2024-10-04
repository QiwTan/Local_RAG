# build_index.py

import os
import json
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config_SQLite_FAISS_RAG as config

def create_sqlite_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_title TEXT,
            chunk_index INTEGER,
            chunk_length INTEGER,
            chunk_content TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def process_and_build_index(
    json_folder_path, 
    db_path, 
    index_path, 
    batch_size,
    embedding_model_name,
    device
):
    conn, cursor = create_sqlite_db(db_path)
    embedding_dim = None
    index = None
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    chunked_files = [f for f in os.listdir(json_folder_path) if f.endswith("_chunked.json")]
    total_files = len(chunked_files)
    print(f"Total number of files to process: {total_files}")
    total_processed_records = 0
    batch_texts = []
    batch_data_items = []

    for file in tqdm(chunked_files, desc="Processing JSON files"):
        file_path = os.path.join(json_folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data_list = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {file}")
                continue
            for data in data_list:
                if 'chunk_content' not in data:
                    continue
                batch_texts.append(data['chunk_content'])
                batch_data_items.append(data)
                total_processed_records += 1
                if len(batch_texts) >= batch_size:
                    embeddings = embedding_model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                    if embeddings is None or len(embeddings) == 0:
                        print("No embeddings returned for batch")
                        batch_texts = []
                        batch_data_items = []
                        continue
                    embeddings = np.array(embeddings, dtype='float32')
                    data_to_insert = [
                        (
                            data_item.get("main_title", ""),
                            data_item.get("chunk_index", 0),
                            data_item.get("chunk_length", 0),
                            data_item['chunk_content']
                        )
                        for data_item in batch_data_items
                    ]
                    cursor.executemany('''
                        INSERT INTO documents (main_title, chunk_index, chunk_length, chunk_content)
                        VALUES (?, ?, ?, ?)
                    ''', data_to_insert)
                    conn.commit()
                    first_row_id = cursor.execute('SELECT last_insert_rowid()').fetchone()[0] - len(data_to_insert) + 1
                    ids = np.arange(first_row_id, first_row_id + len(data_to_insert), dtype='int64')
                    if embedding_dim is None:
                        embedding_dim = embeddings.shape[1]
                        index_flat = faiss.IndexFlatL2(embedding_dim)
                        index = faiss.IndexIDMap(index_flat)
                    index.add_with_ids(embeddings, ids)
                    print(f'Processed {total_processed_records} records')
                    batch_texts = []
                    batch_data_items = []

    if batch_texts:
        embeddings = embedding_model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        if embeddings is not None and len(embeddings) > 0:
            embeddings = np.array(embeddings, dtype='float32')
            data_to_insert = [
                (
                    data_item.get("main_title", ""),
                    data_item.get("chunk_index", 0),
                    data_item.get("chunk_length", 0),
                    data_item['chunk_content']
                )
                for data_item in batch_data_items
            ]
            cursor.executemany('''
                INSERT INTO documents (main_title, chunk_index, chunk_length, chunk_content)
                VALUES (?, ?, ?, ?)
            ''', data_to_insert)
            conn.commit()
            first_row_id = cursor.execute('SELECT last_insert_rowid()').fetchone()[0] - len(data_to_insert) + 1
            ids = np.arange(first_row_id, first_row_id + len(data_to_insert), dtype='int64')
            if embedding_dim is None:
                embedding_dim = embeddings.shape[1]
                index_flat = faiss.IndexFlatL2(embedding_dim)
                index = faiss.IndexIDMap(index_flat)
            index.add_with_ids(embeddings, ids)
            print(f'Processed {total_processed_records} records (completed)')
    conn.close()
    faiss.write_index(index, index_path)
    print(f'FAISS index saved to {index_path}')

if __name__ == "__main__":
    process_and_build_index(
        json_folder_path=config.json_folder_path,
        db_path=config.db_path,
        index_path=config.index_path,
        batch_size=config.batch_size,
        embedding_model_name=config.embedding_model_name,
        device=config.device
    )