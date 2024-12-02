import asyncio
from llama_index.core import StorageContext, PropertyGraphIndex


async def load_index(storage_dir, person_name=None, compress=False, use_subgraph=False):
    """
    加速加载索引的函数，支持异步加载、按需加载和压缩选项。
    :param storage_dir: 存储索引的目录
    :param person_name: 如果指定，则加载与特定人物相关的子图索引
    :param compress: 是否启用压缩以加速存储读取
    :param use_subgraph: 是否仅加载子图
    :return: 加载好的 PropertyGraphIndex 对象
    """
    loop = asyncio.get_event_loop()
    persist_dir = storage_dir
    if person_name and use_subgraph:
        persist_dir = f"{storage_dir}/{person_name}_index"  # 指定人物子图存储目录

    # 创建存储上下文，支持压缩
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir, compress=compress)

    # 异步加载索引
    print(f"Loading index from: {persist_dir}, Compress: {compress}")
    index = await loop.run_in_executor(None, PropertyGraphIndex.load_from_storage, storage_context)

    if not index or not hasattr(index, 'property_graph_store'):
        raise ValueError("Failed to load PropertyGraphIndex or property_graph_store is not initialized.")

    print("Index loaded successfully.")

    # 如果未指定子图但需要过滤
    if person_name and not use_subgraph:
        print(f"Filtering subgraph by person: {person_name}")
        all_triplets = index.property_graph_store.get_all_triplets()

        if not all_triplets:
            raise ValueError("No triplets found in property_graph_store.")

        filtered_triplets = [
            triplet for triplet in all_triplets
            if person_name in triplet['head'] or person_name in triplet['tail']
        ]
        print(f"Filtered {len(filtered_triplets)} triplets related to {person_name}.")

        subgraph = PropertyGraphIndex.from_triplets(
            triplets=filtered_triplets, storage_context=storage_context
        )
        return subgraph

    return index


async def main():
    storage_dir = "/home/share/shucshqyfzyxgsi/home/lishuguang/my_graphrag/zhuyuanzhang_knowledge_graph/index"
    person_name = "朱元璋"

    try:
        # 加载整个索引
        print("Loading full index...")
        index = await load_index(storage_dir)

        # 检查索引加载成功
        if not index or not hasattr(index, 'property_graph_store'):
            raise ValueError("Failed to load PropertyGraphIndex.")

        print("Full index loaded successfully.")

        # 打印全部三元组数量
        if hasattr(index.property_graph_store, 'get_all_triplets'):
            triplets = index.property_graph_store.get_all_triplets()
            print(f"Total triplets in full index: {len(triplets)}")
        else:
            raise AttributeError("property_graph_store does not have get_all_triplets method.")

        # 加载特定人物的子图索引
        print(f"Loading subgraph index for {person_name}...")
        subgraph_index = await load_index(storage_dir, person_name=person_name, use_subgraph=True)

        # 打印子图加载结果
        if subgraph_index:
            print(f"Subgraph index for {person_name} loaded successfully.")
        else:
            print("Failed to load subgraph index.")

    except Exception as e:
        print(f"Error: {e}")


# 运行主程序
asyncio.run(main())