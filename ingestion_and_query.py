from qdrant_client import QdrantClient
from rag_utils.ask_ollama import ask_ollama
from rag_utils.embedding import generate_embeddings
from rag_utils.fetch_data import load_pdf
from rag_utils.preprocessing import preprocess
from rag_utils.chunking import chunk_text
import yaml

from rag_utils.vectorstore import  setup_collection, ingest_chunks
PDF_PATH = "hr_policy_detailed_5_pages.pdf"
    

def load_config():
  
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():

    config = load_config()
    
    # Step 1 - Load PDF and extract text
    text = load_pdf(PDF_PATH)
    # print(text)

    # Step 2 - Preprocess the text
    cleaned_text = preprocess(text)
    # print(cleaned_text)

    print("Chunking...")
    chunks = chunk_text(cleaned_text)
    print("Total chunks:", len(chunks))
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---\n")
        print(chunk)

    # Step 3 - Generate embeddings
    embeddings = generate_embeddings(chunks, config["ollama"]["embedding_model"])
    print("Embeddings generated for all chunks.", embeddings[0][:10])  # Print the first 10 dimensions of the first embedding for verification
  

    # Step 4 - Setup Qdrant collection and ingest data
    print("Connecting to Qdrant...")
    client = QdrantClient(
        host=config["qdrant"]["host"], 
        port=config["qdrant"]["port"]
    )

    setup_collection(
        client, 
        config["qdrant"]["collection_name"],
          config["qdrant"]["vector_size"])
    
    print("Ingesting vectors into Qdrant...")
    ingest_chunks(
        client, 
        config["qdrant"]["collection_name"],
        chunks, 
        embeddings)
    
    print("Data ingestion complete.")

    # Step 5 - Querying the collection  
    question = input("\nEnter your question about the document: ")
    query_vector = generate_embeddings([question], config["ollama"]["embedding_model"])

    top_matches = client.query_points(
        collection_name=config["qdrant"]["collection_name"],
        query=query_vector[0],
        limit=3)
    
    texts = []
    for hit in top_matches.points:
        print(f"\n--- Retrieved Chunk (Score: {hit.score:.4f}) ---\n")
        texts.append(hit.payload["text"])
        print(hit.payload["text"])

    context = "\n\n".join(texts)

    # Step 6 - Generate answer by asking Ollama
    answer = ask_ollama(question, context, config)
    print("\n--- Answer ---\n")
    print(answer)

main()
