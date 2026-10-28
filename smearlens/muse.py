import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

ENDPOINT = "https://inference.baseten.co/v1/chat/completions"
MODEL = "meta-models/muse-spark-open-1026"


def ask_image(image: bytes, prompt: str, max_tokens: int = 4000, retries: int = 3) -> tuple[str, int]:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"},
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            response = json.load(urllib.request.urlopen(request, timeout=180))
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    text = response["choices"][0]["message"].get("content") or ""
    return text.strip(), response.get("usage", {}).get("total_tokens", 0)


def parse_label(text: str, classes: tuple[str, ...]) -> str:
    match = re.search(r"LABEL:\s*\**\s*([a-z_]+)", text, re.IGNORECASE)
    if not match:
        return "unparsed"
    label = match.group(1).lower()
    return label if label in classes else f"other:{label}"
