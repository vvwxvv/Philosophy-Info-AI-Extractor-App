import os
import requests
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

class OpenAIChatClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str ="deepseek-ai/DeepSeek-V3",
        system_prompt: str = "You are a helpful assistant.",
        base_url: str = "https://api.siliconflow.cn/v1/chat/completions",
        temperature: float = 0.7,
        max_tokens: int = 300
    ):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set SILICONFLOW_API_KEY in .env or pass it explicitly.")
        self.model = model
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask(self, user_message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = history[:] if history else []
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"].strip()
            return reply
        except requests.RequestException as e:
            print(f"❌ API request failed: {e}")
            return "Error: Unable to get response from AI."
        except KeyError:
            print("❌ Unexpected API response format:", response.text)
            return "Error: Unexpected response format."

    def with_system_prompt(self, new_prompt: str) -> 'OpenAIChatClient':
        """Clone the client with a new system prompt."""
        return OpenAIChatClient(
            api_key=self.api_key,
            model=self.model,
            system_prompt=new_prompt,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


# ✅ 示例用法
if __name__ == "__main__":
    chat_client = OpenAIChatClient()

    question = "Who are you?"
    reply = chat_client.ask(question)
    print("🤖 AI:", reply)

    # 可复用不同的 prompt
    custom_client = chat_client.with_system_prompt("你是一位哲学家 AI。")
    philosophical_reply = custom_client.ask("意识是什么？")
    print("🧠 哲学AI:", philosophical_reply)
