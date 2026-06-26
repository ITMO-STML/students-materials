import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("CLOUDRU_API_KEY"),
    base_url=os.getenv("CLOUDRU_BASE_URL")
)

response = client.chat.completions.create(
    model="openai/gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": "Answer only OK"
        }
    ],
    temperature=0
)

print(response.choices[0].message.content)