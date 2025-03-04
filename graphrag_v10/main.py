from qa import ask_question
import logging
from pipelineconfig import PipelineConfig
from data_process import (
    process_file,
    setup_storage_dir,
    check_subfolder_exists
)
from knowledge_graph_process import (
    generate_knowledge_graph,
    load_knowledge_graph
)
from qa import ask_question


def my_symbolic_fn(question: str) -> str:
    # 这里举个例子：做 SPARQL 查询, 或 Owlready2, 或 Sympy
    # 只是返回固定文本演示
    # 你可以把 question 解析，执行 RDF/OWL/Sympy，然后拼接字符串
    return "我是符号推理结果: [推理内容示例...]"

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    Pipe_config = PipelineConfig(
        enable_symbolic_reasoning=True,  # 是否启用“符号推理”模式
        reasoning_mode="owl",           # owl / sympy / prolog / ...
        chunk_size=128,
        overlap=10,
        max_triplets_per_chunk=10,
        debug=True
    )

    file_path = "/nfs/hongzhili/my_graphrag/data/高中物理测试题3.pdf"
    file_type = "pdf"
    dir_name = "highschool_physics"

    storage_dir = setup_storage_dir(dir_name)

    # 如果还没生成KG，就先生成，否则直接加载
    if not check_subfolder_exists(storage_dir, "index"):
        docs = process_file(file_path, file_type, Pipe_config)
        index = generate_knowledge_graph(docs, dir_name, storage_dir, Pipe_config)
    else:
        index = load_knowledge_graph(storage_dir)

    # 交互式问答
    while True:
        user_query = input("请输入你的问题(输入 'over' 结束): ")
        if user_query.lower() == "over":
            break

        # 使用 ask_question, 并把你的 my_symbolic_fn 当参数传进去
        answer = ask_question(index, user_query, config, symbolic_fn=my_symbolic_fn)
        print("\n[Final Answer]:\n", answer, "\n")

if __name__ == "__main__":
    main()