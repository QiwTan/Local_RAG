# search_index.py

import numpy as np
import sqlite3
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import config_SQLite_FAISS_RAG as config

def search_similar_chunks(query_embedding, index_path, db_path, top_k=5):
    index = faiss.read_index(index_path)
    query_vector = np.array([query_embedding]).astype('float32')
    distances, ids = index.search(query_vector, top_k)
    results = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for doc_id in tqdm(ids[0], desc="Retrieving data"):
        doc_id = int(doc_id)
        cursor.execute('SELECT main_title, chunk_index, chunk_length, chunk_content FROM documents WHERE id=?', (doc_id,))
        result = cursor.fetchone()
        if result:
            results.append({
                'main_title': result[0],
                'chunk_index': result[1],
                'chunk_length': result[2],
                'chunk_content': result[3]
            })
    conn.close()
    return results

if __name__ == "__main__":
    embedding_model = SentenceTransformer(config.embedding_model_name, device=config.device)
    query = config.query
    query_embedding = embedding_model.encode(query)
    results = search_similar_chunks(query_embedding, config.index_path, config.db_path, top_k=config.top_k)
    if results:
        for idx, result in enumerate(results, start=1):
            print(f"Result {idx}:")
            print(f"Title: {result['main_title']}")
            print(f"Content: {result['chunk_content']}")
            print("-" * 80)
    else:
        print("No matching results found.")