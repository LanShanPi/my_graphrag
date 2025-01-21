# qa.py

import logging

from llama_index.llms.openai import OpenAI as LlamaOpenAI

# ========== 1) 判断是否需要符号推理的函数 ==========

def need_symbolic_reasoning(question: str) -> bool:
    """
    简易示例：若问题中包含典型的逻辑/条件/推断词，则判定需要符号推理。
    你也可以改成更复杂的正则或LLM分类器。
    """
    logic_keywords = [
        "if", "then", "implies", "imply", "when", 
        "假设", "若", "则", "推断", "推理", "定理", "证明"
    ]
    q_lower = question.lower()
    for kw in logic_keywords:
        if kw in q_lower:
            return True
    return False


# ========== 2) 普通问答函数 ==========

def get_response_v1(index, user_query, config):
    """
    最简单的问答逻辑：基于 LlamaIndex 检索+回答，不做任何符号推理。
    """
    llm = LlamaOpenAI(
        temperature=0,
        model="gpt-4o",
        api_key="",     # 请填入真实API key
        base_url="",
        timeout=600
    )

    query_engine = index.as_query_engine(
        llm=llm,
        include_text=True,
        response_mode="tree_summarize",
        similarity_top_k=5
    )

    response = query_engine.query(user_query)
    return response


# ========== 3) 符号推理问答函数(示例) ==========

def get_response_with_symbolic(index, user_query, config, symbolic_fn=None):
    """
    演示：先做普通问答，再结合符号推理结果做二次回答。
    - symbolic_fn: 一个可选的函数, 接收(user_query)并返回推理结果字符串,
      例如 SPARQL 查询结果, OWL 推理结论, Sympy 逻辑表达式等。

    你也可以在这里直接调用:
      - SPARQL / RDF
      - OWLready2 同步推理
      - Sympy/Prolog
    """
    llm = LlamaOpenAI(
        temperature=0,
        model="gpt-4o",
        api_key="",
        base_url="",
        timeout=600
    )

    query_engine = index.as_query_engine(
        llm=llm,
        include_text=True,
        response_mode="tree_summarize",
        similarity_top_k=5
    )

    # 第一步：先得到初步回答
    initial_answer = query_engine.query(user_query)

    # 第二步：符号推理(若有传入 symbolic_fn)
    if symbolic_fn is not None:
        reasoning_result = symbolic_fn(user_query)
        # 可以将推理结果与初步回答作二次合并
        refine_query = f"""
        我从符号推理中得到以下结果:
        {reasoning_result}

        之前的回答是:
        {initial_answer}

        请结合上述推理结果和回答，再进行补充或修正:
        """
        refined_answer = query_engine.query(refine_query)
        return refined_answer
    
    # 若没有 symbolic_fn，直接返回初步回答
    return initial_answer


# ========== 4) 一个统一的问答入口函数 ==========

def ask_question(index, user_query, config, symbolic_fn=None):
    """
    根据:
      1. config.enable_symbolic_reasoning
      2. 用户问题是否含逻辑/推断词(need_symbolic_reasoning)
    判断要不要走符号推理流程。
    """
    # 先简单判断
    if config.enable_symbolic_reasoning and need_symbolic_reasoning(user_query):
        # 如果系统和问题都指示要用符号推理，就调用 get_response_with_symbolic
        logging.info("[QA] Using symbolic reasoning pipeline.")
        return get_response_with_symbolic(index, user_query, config, symbolic_fn)
    else:
        # 否则就是普通问答
        logging.info("[QA] Using normal QA pipeline.")
        return get_response_v1(index, user_query, config)