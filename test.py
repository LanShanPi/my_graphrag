import os 
# os.environ["OPENAI_API_KEY"] = ""

from llama_index.core import SimpleDirectoryReader

from llama_index.core import SimpleNodeParser 
# 直接读取整个文件夹中的文件
# documents = SimpleDirectoryReader("./data").load_data()

# 将文件加载未文档
reader = SimpleDirectoryReader(
    input_files = ["红楼梦.txt"]
)
documents = reader.load_data()

# 将文档解析未节点
# 初始化文档解析器
parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20) 

# 将文档解析为节点 
nodes = parser.get_nodes_from_documents(documents)
