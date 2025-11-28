import requests

payload = {
      "text": "Hello, I am happy today",
      "emotions": ["Cheerful", "Warm"],
      "percentages": ["77%", "23%"],
      "top_k": 1
}

res = requests.post(
    "http://localhost:7860/generate",
    json=payload
)

print(res.text)
