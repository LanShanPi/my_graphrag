
from config import OPENAI_API_KEY1 as OPENAI_API_KEY
from config import API_BASE2

from openai import OpenAI
client = OpenAI(  
    api_key=OPENAI_API_KEY, # 此处使用openai官方key，怎么获取本文不讨论
    base_url=API_BASE2 # "https://api.openai.com/v1" # 既然是直接访问官方站点，这个变量可以不用声明
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "根据你对中国历史古籍的理解，回答问题"},
        {"role": "user","content": "王熙凤有哪些别名，显示格式为：&名字1&名字2...名字n&"}
    ]
)

print(completion.choices[0].message.content.split("&")[1:-1])