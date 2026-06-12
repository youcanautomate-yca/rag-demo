
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_ollama(question: str, context: str, config) -> str:
    
    prompt = f"""

You are a helpful HR assistant that answers questions based on the following context:

context: {context}
question: {question}

please make sure to answer the question based on the context provided. If the context does not contain the answer, 
please say "I don't know"."""
    
    response = requests.post(OLLAMA_URL, 
                             json={"model": config["ollama"]["llm_model"], 
                                   "prompt": prompt,
                                   "stream": False})
    data = response.json()
    if "response" not in data:
        raise ValueError(f"Response not found in OLLAMA response: {data}")
    return data["response"]