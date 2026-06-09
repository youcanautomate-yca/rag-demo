import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"

def generate_embeddings(chunks, model_name):

    embeddings = []
    for chunk in chunks:
        response = requests.post(OLLAMA_URL, 
                                 json={"model": model_name, "prompt": chunk})
        data = response.json()
        if "embedding" not in data:
            raise ValueError(f"Embedding not found in response: {data}")
        embeddings.append(data["embedding"])
    return embeddings