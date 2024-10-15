import os 
os.environ["OPENAI_API_KEY"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser 
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

from llama_index.core import StorageContext

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from IPython.display import Markdown, display


def load_file():
    ######加载数据
    # 直接读取整个文件夹中的文件
    # documents = SimpleDirectoryReader("./data").load_data()

    # # 直接从文本转换
    # text_list = [text1, text2, ...]
    # documents = [Document(t) for t in text_list]

    # 将文件加载为文档
    reader = SimpleDirectoryReader(
        input_files = ["红楼梦.txt"]
    )
    documents = reader.load_data()

    return documents

def create_nodes(documents):
    ###### 创建节点
    # 将文档解析未节点
    # 最简单的节点构建方式
    # 初始化文档解析器
    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20) 
    # 将文档解析为节点 
    nodes = parser.get_nodes_from_documents(documents)

    """也可以直接手动创建节点，这样就可以直接跳过加载文档那一步"""
    return nodes

def create_graph_index(documents):

    print("创建图索引")

    # llm = OpenAI(temperature=0, model="text-davinci-002")
    # Settings.llm = llm
    # Settings.chunk_size = 512

    # # 创建并存储图索引
    # graph_store = SimpleGraphStore()
    # storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # 使用本地编码模型模型
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="/home/kuaipan/model/bgelarge/bge-large-zh-v1.5"
    )
    # storage_context=storage_context,
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=2,
        embed_model=Settings.embed_model
    )
    index.root_index.storage_context.persist(persist_dir=graph_index_dict)

def create_index(nodes,documents):
    
    # # rebuild storage context
    # storage_context = StorageContext.from_defaults(persist_dir="./data")
    # # load index
    # index = load_index_from_storage(storage_context)

    ###### 创建index，需要模型
    # # 从node创建
    # index = VectorStoreIndex(nodes)

    # # 直接从文档创建，这样就可以直接跳过创建node的步骤
    # from llama_index.core import VectorStoreIndex
    # index = VectorStoreIndex.from_documents(documents)

    # 从文档创建，自定义编码模型进行index创建
    # # 使用openai模型
    # Settings.embed_model = OpenAIEmbedding()

    # 使用本地编码模型模型
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="/home/kuaipan/model/bgelarge/bge-large-zh-v1.5"
    )
    # per-index
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
    # 将index存储到磁盘
    index.storage_context.persist(persist_dir=index_dict)


    # # 多个index复用node
    # from gpt_index.docstore import SimpleDocumentStore
    # docstore = SimpleDocumentStore()
    # docstore.add_documents(nodes)
    # index1 = GPTSimpleVectorIndex(nodes, docstore=docstore)
    # index2 = GPTListIndex(nodes, docstore=docstore)


    # # 将文档插入index
    # from llama_index import GPTSimpleVectorIndex
    # index = GPTSimpleVectorIndex([])
    # for doc in documents:
    #     index.insert(doc)

    return index

def graph_response(index):
    query_engine = index.as_query_engine(
        include_text=False, response_mode="tree_summarize"
    )
    response = query_engine.query(
        "Tell me more about Interleaf",
    )
    print(response)


def respones(index):
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "林黛玉是谁"
    )
    print(response)

def main():
    graph_response(graph_index)

if __name__ == "__main__":
    index_dict = "./data/index/"
    graph_index_dict = "./data/graph_index"
    documents = load_file()
    # nodes = create_nodes(documents)
    # index = create_index(nodes,documents)
    graph_index = create_graph_index(documents)
    
    main()


