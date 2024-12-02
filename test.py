import os
from llama_index.core import StorageContext, PropertyGraphIndex
import networkx as nx
from pyvis.network import Network
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext, load_index_from_storage
from config import OPENAI_API_KEY1,API_BASE
from prompt.prompt import CUSTOM_KG_TRIPLET_EXTRACT_TMPL
import openai



# 设置 OpenAI API
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY1
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = API_BASE



def filter_subgraph_by_person(index, person_name):
    """
    从现有索引中过滤出特定人物的知识图谱子图
    :param index: PropertyGraphIndex 对象
    :param person_name: 历史人物名称
    :return: 与人物相关的三元组列表
    """
    # 获取所有知识三元组
    triplets = index.property_graph_store.get_all_triplets()

    # 筛选与指定人物相关的三元组
    filtered_triplets = [
        triplet for triplet in triplets
        if person_name in triplet['head'] or person_name in triplet['tail']
    ]
    return filtered_triplets

def create_graph(triplets):
    """
    根据三元组列表创建 networkx 图
    :param triplets: 知识三元组列表
    :return: networkx 图对象
    """
    graph = nx.DiGraph()  # 使用有向图
    for triplet in triplets:
        head, relation, tail = triplet['head'], triplet['relation'], triplet['tail']
        graph.add_node(head)
        graph.add_node(tail)
        graph.add_edge(head, tail, label=relation)
    return graph

def visualize_graph(graph, output_file):
    """
    将 networkx 图可视化为 HTML 文件
    :param graph: networkx 图对象
    :param output_file: 输出的 HTML 文件路径
    """
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(graph)  # 从 networkx 图加载
    for edge in graph.edges(data=True):
        net.edges[net.get_edge_index(edge[0], edge[1])]['label'] = edge[2]['label']
    net.show(output_file)

def process_person_graph(storage_dir, person_name, output_dir):
    """
    处理某个人物的知识图谱并生成 HTML 可视化文件
    :param storage_dir: 存储索引的目录
    :param person_name: 历史人物名称
    :param output_dir: 输出目录
    """
    # 加载存储上下文和索引
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)

    # 过滤出与目标人物相关的三元组
    filtered_triplets = filter_subgraph_by_person(index, person_name)

    # 创建知识图谱
    graph = create_graph(filtered_triplets)

    # 可视化并保存 HTML
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{person_name}_graph.html")
    visualize_graph(graph, output_file)
    print(f"知识图谱 HTML 文件已保存到: {output_file}")

# 示例调用
storage_dir = "/home/share/shucshqyfzyxgsi/home/lishuguang/my_graphrag/zhuyuanzhang_knowledge_graph/index"  # 替换为索引存储目录
person_name = "朱棣"
output_dir = "./output"

process_person_graph(storage_dir, person_name, output_dir)