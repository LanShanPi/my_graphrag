
from llama_index.core import  StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai import OpenAI
import openai
import torch
from config import OPENAI_API_KEY2 as OPENAI_API_KEY
from config import API_BASE3 as API_BASE
from config import API_BASE2
from functools import lru_cache
from openai import OpenAI as local_openai
from prompt.response_prompt import mingchao_person
import os
import re
# 设置 OpenAI API
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = API_BASE

# 配置 GPU/CPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm = OpenAI(temperature=0, model="gpt-4o", device=device)


def load_knowledge_graph(storage_dir):
    # 加载已存储的索引
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir+"/"+"index")
    index = load_index_from_storage(storage_context)
    return index


def get_response(index, noun):
    client = local_openai(  
        api_key=OPENAI_API_KEY,
        base_url=API_BASE2 
    )
    completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "根据你对中国历史古籍的理解，回答问题"},
                {"role": "user", "content": f"{noun}有哪些别的称呼，请尽可能全面的罗列，显示格式为：&名字1&名字2...名字n&"}
            ]
        )
    nouns = completion.choices[0].message.content.split("&")[1:-1]
    query_engine = index.as_query_engine(llm=llm, include_text=True,response_mode="tree_summarize",similarity_top_k=5)
    response = query_engine.query(mingchao_person.format(noun=noun,nouns=nouns))
    return response

if __name__ == "__main__":
    index = load_knowledge_graph("/home/share/shucshqyfzyxgsi/home/lishuguang/my_graphrag/mingchao4_knowledge_graph")
    response = get_response(index,"朱元璋")
    print(response)
