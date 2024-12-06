
from config import OPENAI_API_KEY1 as OPENAI_API_KEY
from config import API_BASE2

from openai import OpenAI
client = OpenAI(  
    api_key=OPENAI_API_KEY, 
    base_url=API_BASE2
     )

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "根据你对中国历史古籍的理解，回答问题"},
        {"role": "user","content": "王熙凤有哪些别名，显示格式为：&名字1&名字2...名字n&"}
    ]
)

print(completion.choices[0].message.content.split("&")[1:-1])