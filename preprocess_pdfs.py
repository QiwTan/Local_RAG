# preprocess_pdfs.py

import os
import json
import re
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config_SQLite_FAISS_RAG as config

# 使用配置文件中的参数
input_folder = config.input_folder
output_folder = config.output_folder
chunk_size = config.chunk_size
chunk_overlap = config.chunk_overlap

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 数据清洗函数
def clean_text(text):
    # 1. 合并由于换行和连字符分隔的单词
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 2. 将换行符替换为空格
    text = text.replace('\n', ' ')
    # 3. 删除引用（如 [1]、(1) 等）
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    # 4. 删除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 删除 "References" 之后的内容
def remove_references(text):
    reference_keywords = ['references', 'bibliography']
    pattern = re.compile(r'|'.join(reference_keywords), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        text = text[:match.start()]
    return text

# 提取标题和小标题
def extract_titles(raw_text):
    main_title = raw_text.split("\n", 1)[0].strip()
    section_pattern = re.compile(r"^(?:\d+\.?|\d+\.\d+|\b[A-Z]+\b)\s+[A-Za-z]", re.MULTILINE)
    section_titles = section_pattern.findall(raw_text)
    return main_title, section_titles

# 定义文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)

# 获取文件夹中的所有 PDF 文件
pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]

# 使用 tqdm 显示进度条
for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
    file_path = os.path.join(input_folder, pdf_file)
    loader_py = PyPDFLoader(file_path)
    pages_py = loader_py.load()
    pages_content_py = [page.page_content for page in pages_py]
    raw_text = "\n".join(pages_content_py)
    main_title, section_titles = extract_titles(raw_text)
    text_no_references = remove_references(raw_text)
    chunks = text_splitter.split_text(text_no_references)
    cleaned_chunks = [clean_text(chunk) for chunk in chunks]
    chunked_data = [
        {
            "main_title": main_title,
            "chunk_index": idx + 1,
            "chunk_length": len(chunk),
            "chunk_content": chunk
        }
        for idx, chunk in enumerate(cleaned_chunks)
    ]
    output_json_path_py = os.path.join(output_folder, pdf_file.replace(".pdf", "_chunked.json"))
    with open(output_json_path_py, "w", encoding="utf-8") as json_file:
        json.dump(chunked_data, json_file, ensure_ascii=False, indent=4)