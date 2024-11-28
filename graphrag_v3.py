from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
import os
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext, load_index_from_storage
from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from pyvis.network import Network
import IPython
import openai
import torch
from llama_index.readers.file import PyMuPDFReader
import pdfplumber
from llama_index.core.schema import Document
import re
from docx import Document as DocxDocument
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from config import OPENAI_API_KEY1,API_BASE

# 设置 OpenAI API
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY1
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = API_BASE

# 配置 GPU/CPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm = OpenAI(temperature=0, model="gpt-4o", device=device)

# 文件读取函数
def read_docx(file_path):
    docx = DocxDocument(file_path)
    content = []
    for paragraph in docx.paragraphs:
        text = paragraph.text.strip()
        if text:
            content.append(text)
    for table in docx.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                content.append(" | ".join(row_text))
    return content

def preprocess_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text_with_overlap(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def read_pdf_with_pymupdf(file_path):
    documents = []
    pdf = fitz.open(file_path)
    for i, page in enumerate(pdf):
        text = page.get_text("text")
        if text:
            processed_text = preprocess_text(text)
            documents.append(Document(text=processed_text, extra_info={"source": f"Page {i + 1}"}))
    pdf.close()
    return documents

def process_file(file_path, file_type):
    if file_type == "docx":
        docx_content = read_docx(file_path)
        if not docx_content:
            raise ValueError("未从文件中读取到任何内容，请检查文件格式或内容是否为空。")
        documents = [
            Document(text=preprocess_text(text), extra_info={"source": f"Content {i + 1}"})
            for i, text in enumerate(docx_content)
        ]
    elif file_type == "pdf":
        documents = read_pdf_with_pymupdf(file_path)
    elif file_type == "txt":
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return documents

# 定义知识提取模板
def mingchao():
    CUSTOM_KG_TRIPLET_EXTRACT_TMPL = (
        "从给定文本中提取与明朝历史相关的知识三元组。"
        "每个三元组应以 (head, relation, tail) 及其各自的类型形式出现。\n"
        "---------------------\n"
        "初始本体论：\n"
        "实体类型：{allowed_entity_types}\n"
        "关系类型：{allowed_relation_types}\n"
        "\n"
        "以这些类型为起点，但根据上下文需要引入新类型。\n"
        "\n"
        "指南：\n"
        "- 以 JSON 格式输出：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
        "- 使用最完整的实体形式（例如，使用 '朱元璋' 而不是 '朱'）\n"
        "- 保持实体简洁（最多 3-5 个词）\n"
        "- 将复杂短语分解为多个三元组\n"
        "- 确保知识图谱连贯且易于理解\n"
        "- 专注于明朝的关键人物、事件、政策、战役、制度等方面\n"
        "- 包括人物的出生、登基、去世、战役的起因、过程、结果、政策的制定与影响等\n"
        "---------------------\n"
        "文本：{text}\n"
        "输出：\n"
    )
    CUSTOM_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
        CUSTOM_KG_TRIPLET_EXTRACT_TMPL,
        prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
    )
    allowed_entity_types = [
        "PERSON", "EVENT", "POLITICAL_FIGURE", "MILITARY_LEADER",
        "DYNASTY", "BATTLE", "CAPITAL", "POLICY", "INSTITUTION",
        "CULTURAL_FIGURE", "ECONOMIC_POLICY", "SOCIAL_SYSTEM"
    ]
    allowed_relation_types = [
        "FOUNDER", "MEMBER", "GENERAL", "CAPTURED_BY", "SUCCEEDED_BY",
        "PARTICIPANT", "RESULTED_IN", "IMPLEMENTED", "INFLUENCED_BY",
        "BORN_IN", "DIED_IN", "ASCENDED_THRONE", "STARTED_REIGN",
        "ENDED_REIGN", "LED_BATTLE", "DEFEATED"
    ]
    return allowed_entity_types, allowed_relation_types, CUSTOM_KG_TRIPLET_EXTRACT_PROMPT

# 过滤特定人物的三元组
def filter_triplets_by_person(triplets, person_name):
    return [
        triplet for triplet in triplets
        if person_name in triplet["head"] or person_name in triplet["tail"]
    ]

# 文件处理和知识图谱构建
file_path = "/home/share/shucshqyfzyxgsi/home/lishuguang/my_graphrag/data/明朝那些事儿.pdf"
documents = process_file(file_path, "pdf")

chunk_size = 512
overlap = 50
chunked_documents = []
for doc in documents:
    text_chunks = chunk_text_with_overlap(doc.text, chunk_size, overlap)
    for chunk in text_chunks:
        chunked_documents.append(Document(text=chunk, extra_info=doc.extra_info))

# 加载模板
entity_types, relation_types, CUSTOM_KG_TRIPLET_EXTRACT_PROMPT = mingchao()
person_name = "朱元璋"

# 创建存储上下文
PERSIST_DIR_PERSON = f"{person_name}_knowledge_graph"
HTML_FILE_PERSON = f"{person_name}_knowledge_graph.html"
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 构建针对特定人物的知识图谱
index = PropertyGraphIndex.from_documents(
    chunked_documents,
    max_triplets_per_chunk=10,
    storage_context=storage_context,
    show_progress=True,
    include_embeddings=False,
    kg_triple_extract_template=CUSTOM_KG_TRIPLET_EXTRACT_PROMPT,
    allowed_entity_types=entity_types,
    allowed_relation_types=relation_types,
    triplet_filter_fn=lambda triplets: filter_triplets_by_person(triplets, person_name)
)

# 存储知识图谱
if not os.path.exists(PERSIST_DIR_PERSON):
    os.makedirs(PERSIST_DIR_PERSON)
storage_context.persist(persist_dir=PERSIST_DIR_PERSON)
index.property_graph_store.save_networkx_graph(name=HTML_FILE_PERSON)

# 查询特定人物的知识
query_engine = index.as_query_engine(llm=llm, include_text=False)
response = query_engine.query("朱元璋当皇帝前都做过什么")
print(response)