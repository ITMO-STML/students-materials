import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from openai import InternalServerError, RateLimitError, APIConnectionError


class LLMClient:
    def __init__(self, model_name: str):
        load_dotenv()

        self.model_name = model_name

        self.client = OpenAI(
            api_key=os.getenv("CLOUDRU_API_KEY"),
            base_url=os.getenv("CLOUDRU_BASE_URL"),
            timeout=60
        )

    def generate(self, prompt: str) -> str:
        max_retries = 5

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0
                )

                return response.choices[0].message.content

            except (InternalServerError, RateLimitError, APIConnectionError) as e:
                wait_seconds = attempt * 10

                print(f"LLM error on attempt {attempt}/{max_retries}: {e}")
                print(f"Waiting {wait_seconds} seconds...")

                time.sleep(wait_seconds)

        raise RuntimeError("LLM request failed after several retries")