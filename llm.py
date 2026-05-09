# src/llm.py

import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenRouter-compatible client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1"
)


def encode_image_to_base64(image_path: str) -> str:
    """
    Load an image from disk and return a base64 string.
    Qwen‑3‑Vision accepts base64 images in the messages API.
    """
    img_bytes = Path(image_path).read_bytes()
    return base64.b64encode(img_bytes).decode("utf-8")


def run_qwen_vision(prompt: str, image_path: str) -> str:
    """
    Send the prompt + image to Qwen‑3‑Vision via OpenRouter.
    Returns the model's text output.
    """

    image_b64 = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model="qwen/qwen-2.5-vision-instruct",   # OpenRouter model name
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=800,
        temperature=0.2,
    )

    return response.choices[0].message["content"]
