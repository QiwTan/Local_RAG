# config.py

# Path settings
input_folder = "/Users/tanqiwen/Documents/SQLite-FAISS-RAG"      # Folder containing PDF files
output_folder = "/Users/tanqiwen/Documents/SQLite-FAISS-RAG"    # Folder to save processed JSON files

json_folder_path = output_folder  # Using the output folder from the previous step
db_path = "chunks.db"             # Path to the SQLite database
index_path = "faiss.index"        # Path to the FAISS index

# Text segmentation parameters
chunk_size = 1200
chunk_overlap = 400

# Embedding model parameters
embedding_model_name = 'BAAI/bge-base-en-v1.5'  # Or other models
device = 'mps'  # Options: 'cpu', 'cuda', 'mps', etc.

# Batch size
batch_size = 500  # Adjust according to the GPU memory capacity

# Query parameters
query = "PaLM"
top_k = 5
