
import os
import asyncio
import logging
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from science_graph import generate_knowledge_graph, load_knowledge_graph,get_response


app = FastAPI()
# 全局缓存知识图谱索引
knowledge_graphs = {}

async def save_uploaded_file(upload_file: UploadFile, destination: str):
    """
    保存上传的文件到指定路径
    """
    try:
        with open(destination, "wb") as buffer:
            buffer.write(await upload_file.read())
        return destination
    except Exception as e:
        logging.error(f"保存文件失败: {e}")
        return None

async def generate_index(file_path, file_type, dir_name, storage_dir):
    """
    异步生成知识图谱索引
    """
    if dir_name in knowledge_graphs:
        # 如果缓存中已有索引，直接返回
        logging.info(f"知识图谱 {dir_name} 已在缓存中")

    if not os.path.exists(os.path.join(storage_dir, "index")):
        # 如果图谱不存在，则生成
        logging.info("开始生成知识图谱...")
        index = await asyncio.to_thread(generate_knowledge_graph, file_path, file_type, dir_name, storage_dir)
        logging.info("图谱生成完成！")
    # 将索引添加到缓存
    knowledge_graphs[dir_name] = index
    logging.info(f"新知识图谱 {dir_name} 已加载到内存")

async def query_knowledge_graph(index, queries):
    """
    异步查询知识图谱
    """
    response = await asyncio.to_thread(get_response, index,queries)
    return response

@app.on_event("startup")
async def load_existing_knowledge_graphs():
    """
    启动时加载已有的知识图谱到内存中
    """
    base_dir = os.getcwd()
    for folder in os.listdir(base_dir):
        storage_dir = os.path.join(base_dir, folder)
        if os.path.isdir(storage_dir) and os.path.exists(os.path.join(storage_dir, "index")):
            dir_name = folder.replace("_knowledge_graph", "")
            logging.info(f"加载已有知识图谱: {dir_name}")
            index = await asyncio.to_thread(load_knowledge_graph, storage_dir)
            knowledge_graphs[dir_name] = index

@app.post("/upload")
async def upload_and_process(file: UploadFile = File(...), file_type: str = Form(...), dir_name: str = Form(...)):
    """
    接收上传的文件并生成知识图谱
    """
    try:
        # 保存文件
        storage_dir = os.path.join(os.getcwd(), f"{dir_name}_knowledge_graph/")
        os.makedirs(storage_dir, exist_ok=True)
        file_path = os.path.join(storage_dir, file.filename)

        saved_file_path = await save_uploaded_file(file, file_path)
        if not saved_file_path:
            return {"status": "error", "message": "文件保存失败"}

        # 加载或生成索引
        await generate_index(saved_file_path, file_type, dir_name, storage_dir)

        return {"status": "success", "message": "文件处理完成并已加载到内存", "index_path": storage_dir}

    except Exception as e:
        logging.error(f"上传文件处理失败: {e}")
        return {"status": "error", "message": "文件处理失败"}

@app.post("/query")
async def query_graph(question: str = Form(...), graph_name: str = Form(...)):
    """
    查询已生成的知识图谱
    """
    try:
        if graph_name not in knowledge_graphs:
            return {"status": "error", "message": "未查询到相关知识图谱，请先上传文件生成知识图谱"}

        # 查询索引
        index = knowledge_graphs[graph_name]
        pre_prompt = (
            "根据自身能力和检索到的知识尽可能详细的回复下述问题，且回复要满足："
            "回答的准确性、回答的完整性、回答的逻辑性、回答的语言表达清晰性，这四个要求，"
            "仅需输出回答就好了，不需要额外的输出。不要生成 LaTeX 语法。下面是问题："
        )
        response = await query_knowledge_graph(index, pre_prompt + question)

        return {"status": "success", "response": str(response)}

    except Exception as e:
        logging.error(f"查询失败: {e}")
        return {"status": "error", "message": "查询失败"}

if __name__ == "__main__":

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("high_concurrency_knowledge_graph.log"),
            logging.StreamHandler()
        ]
    )
    uvicorn.run(app, host="0.0.0.0", port=8001)
