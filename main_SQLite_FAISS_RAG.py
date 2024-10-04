# main.py

import subprocess

print("Step 1: Preprocessing PDFs...")
subprocess.run(["python3", "preprocess_pdfs.py"])

print("Step 2: Building FAISS index...")
subprocess.run(["python3", "build_index.py"])

print("Step 3: Searching the index...")
subprocess.run(["python3", "search_index.py"])