import argparse
import os
import pandas as pd
from vertexai.preview.language_models import TextEmbeddingModel
from google.cloud import aiplatform

from transformers import pipeline
from google import genai
from google.genai import types
from vertexai.preview.generative_models import GenerativeModel, Part,GenerationConfig



def load_metadata_from_gcs(bucket_name, file_path):
    """
    Load metadata from a JSON Lines (JSONL) file stored in GCS.

    Args:
        bucket_name (str): Google Cloud Storage bucket name.
        file_path (str): Path to the JSONL file in the bucket.

    Returns:
        dict: A dictionary where keys are chunk IDs and values are their metadata.
    """
    from google.cloud import storage
    import json

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Read content of the JSONL file
    metadata_jsonl = blob.download_as_text()

    # Parse JSONL content line-by-line
    metadata = {}
    for line in metadata_jsonl.strip().split("\n"):
        record = json.loads(line)  # Parse each line (a JSON object)
        metadata[record["id"]] = record["metadata"]  # Store "id" as key and "metadata" as value

    return metadata

def get_metadata_for_chunk(metadata, chunk_id):
    """
    Retrieve metadata for the given chunk ID.

    Args:
        metadata (dict): Loaded metadata.
        chunk_id (str): Chunk ID.

    Returns:
        str: Text content of the chunk.
    """
    return metadata.get(chunk_id, "⚠️ No metadata available for this chunk ID!")

import re

def clean_chunk_text(raw_text):
    """
    Clean the raw text to ensure it is well-structured and formatted.

    Args:
        raw_text (str): Raw text of the chunk.

    Returns:
        str: Cleaned content comprising the first few sentences (up to 3).
    """
    # Step 1: Remove excessive whitespace, newlines, and leading/trailing spaces
    cleaned_text = raw_text.strip().replace("\n", " ")  # Remove newlines
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Replace multiple spaces with a single space

    # Step 2: Split text into individual sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cleaned_text)  # Split on periods, question marks

    # Step 3: Filter out empty or overly short sentences (optional, if needed)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]  # Remove empty strings

    # Step 4: Return the first few sentences (e.g., first 3 for conciseness)
    return " ".join(sentences) if sentences else "Error: No valid sentences found in chunk."

from vertexai.preview.language_models import TextGenerationModel

from vertexai.preview.language_models import TextGenerationModel

from vertexai.generative_models import GenerativeModel

def generate_answer(query, cleaned_chunk_text, model_name="gemini-1.5-flash"):
    """
    Generate an answer using the generative model, based on the query and cleaned context.

    Args:
        query (str): The user's query.
        cleaned_chunk_text (str): Relevant context extracted from the chunk.
        model_name (str): The generative model name.

    Returns:
        str: Generated answer.
    """
    prompt = f"""
    Instruction: You are an assistant that provides answers strictly based on the provided context extracted from a PDF file. 
                 Preserve original formatting, and do not speculate beyond the PDF content.
    
    Context extracted from the PDF:
    {cleaned_chunk_text}
    
    Question:{query}
    Answer (strictly using only the provided context and preserving all formatting such as bullet points):
     """



    # Initialize the Gemini model (generative model)
  #  model = GenerativeModel(model_name)
    model = GenerativeModel(
        model_name,
        generation_config=GenerationConfig(
            temperature=0.01,
            top_p=0.8,
            top_k=40,
            max_output_tokens=5000
        )
    )
    
    print(cleaned_chunk_text)

    # Start a fresh chat session
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    
    return response.text.strip()




def chunk_text(query, chunk_size=100):
    """
    Divide the input query into chunks of a specified size.

    Args:
        query (str): The input query text.
        chunk_size (int): The maximum number of words per chunk.

    Returns:
        list: List of text chunks.
    """
    words = query.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def save_to_csv(chunks, output_csv_path):
    """
    Save text chunks to a CSV file.

    Args:
        chunks (list): List of text chunks.
        output_csv_path (str): Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df = pd.DataFrame({"chunk": chunks})
    df.to_csv(output_csv_path, index=False)
    print(f"Chunks saved to: {output_csv_path}")


def generate_embedding(chunk, embedding_model_name="text-embedding-005"):
    """
    Generate an embedding for a single chunk.

    Args:
        chunk (str): The text chunk.
        embedding_model_name (str): VertexAI model for generating embeddings.

    Returns:
        list: Embedding vector.
    """
    model = TextEmbeddingModel.from_pretrained(embedding_model_name)
    embedding = model.get_embeddings([chunk])[0]
    print(embedding.values)
    return embedding.values


def query_matching_engine(project_id, region, index_endpoint_name, deployed_index_id, query_embedding, top_k):
    """
    Query the Matching Engine to find similar items for the query embedding.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Vertex AI region.
        index_endpoint_name (str): Name of the Matching Engine endpoint.
        deployed_index_id (str): ID of the deployed index.
        query_embedding (list): The query embedding vector.
        top_k (int): The number of nearest neighbors to retrieve.

    Returns:
        list: Nearest neighbor results.
    """
    aiplatform.init(project=project_id, location=region)
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name)
    response = index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_embedding],
        num_neighbors=top_k,
        return_full_datapoint=False
    )
    

    results = []
    print(f"Response: {response}")
    if not response or  len(response[0]) == 0:
        print("⚠️ No matching neighbors found.")
        return results
    

    for neighbor in response[0]:
        results.append({
            "id": neighbor.id,
            "distance": neighbor.distance,
           # "metadata": neighbor.metadata,
        })
    
    return results

def get_latest_deployed_index_id(project_id, region, index_endpoint_name):
    """
    Retrieve the latest deployed index ID for the specified endpoint.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Vertex AI region.
        index_endpoint_name (str): Name of the Matching Engine endpoint.

    Returns:
        str: Latest deployed index ID.

    Raises:
        ValueError: If no indexes are deployed to the specified endpoint.
    """
    aiplatform.init(project=project_id, location=region)

    # Retrieve the index endpoint
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name)

    # Check if there are deployed indexes
    if index_endpoint.deployed_indexes:
        deployed_id = index_endpoint.deployed_indexes[0].id
        print(f"Latest deployed index ID: {deployed_id}")
        return deployed_id
    else:
        raise ValueError(f"No indexes are currently deployed to endpoint: {index_endpoint_name}")
        
def get_index_endpoint_resource_name(project_id, region, display_name):
    """
    Retrieve the full resource name of the index endpoint by its display name.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Vertex AI region.
        display_name (str): The display name of the index endpoint.

    Returns:
        str: Full resource name of the index endpoint.

    Raises:
        ValueError: If no matching endpoint is found with the given display name.
    """
    aiplatform.init(project=project_id, location=region)

    # List all endpoints in the region for the project
    endpoints = aiplatform.MatchingEngineIndexEndpoint.list()

    for endpoint in endpoints:
        if endpoint.display_name == display_name:
            print(f"Found endpoint '{display_name}' with resource name: {endpoint.resource_name}")
            return endpoint.resource_name

    raise ValueError(f"No index endpoint found with display name: {display_name}")


def handle_query(project_id, region, index_endpoint_name, deployed_index_id, query, chunk_size=100, top_k=5):
    """
    End-to-end function: chunk query, store chunks in CSV, generate embeddings,
    and query the Matching Engine for results.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Vertex AI region.
        index_endpoint_name (str): Matching Engine endpoint name.
        deployed_index_id (str): Deployed index ID.
        query (str): The query string provided by the user.
        chunk_size (int): The maximum number of words per chunk.
        top_k (int): The number of nearest neighbors to retrieve.
    """
    # Step 1: Retrieve the latest deployed index ID if not provided
    if not deployed_index_id:
        print("Deployed index ID not provided. Retrieving dynamically...")
        deployed_index_id = get_latest_deployed_index_id(project_id, region, index_endpoint_name)

    # Step 2: Chunk the input query
    print("Chunking the query...")
    chunks = chunk_text(query, chunk_size=chunk_size)

    # Step 3: Save chunks to CSV
    output_csv_path = "data/chunk_result.csv"
    save_to_csv(chunks, output_csv_path)

    # Step 4: Generate embedding for the full query by aggregating chunk embeddings
    print("Generating embeddings for chunks...")
    aggregated_embedding = None
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk}")
        embedding = generate_embedding(chunk)
        # Aggregate embeddings (simple average)
        if aggregated_embedding is None:
            aggregated_embedding = embedding
        else:
            aggregated_embedding = [
                (a + b) for a, b in zip(aggregated_embedding, embedding)
            ]
    aggregated_embedding = [x / len(chunks) for x in aggregated_embedding]  # Normalize by chunk count

    # Step 5: Query Matching Engine with the aggregated embedding
    print("Querying Matching Engine for results...")
    results = query_matching_engine(
        project_id, region, index_endpoint_name, deployed_index_id, aggregated_embedding, top_k
    )
    
    metadata = load_metadata_from_gcs(bucket_name="cloud-ai-platform-4292e48c-7499-4d31-8059-7ec80f0d493c", file_path="batch_root/embeddings.json")
    print("metadata loaded")
    print(metadata)
    # Step 6: Display results
    print("\n--- Matching Engine Results ---")
    if results:
        for result in results:
            chunk_id = result['id']
            raw_chunk_text = get_metadata_for_chunk(metadata, chunk_id)

            # Extract the "text" field from the metadata dictionary
            if isinstance(raw_chunk_text, dict) and "text" in raw_chunk_text:
                raw_text = raw_chunk_text["text"]  # Retrieve the text field
                cleaned_chunk_text = clean_chunk_text(raw_text)
                # Generate the final answer
                
                prompt = f"""
Instruction: Based on the provided context, explain the answer to the question comprehensively, highlighting specific features or points from the context.
Context: {cleaned_chunk_text}
Question: {query}
Answer:
"""
                #final_answer = generate_answer_hf(prompt, max_length=200, temperature=0.8, num_beams=5)
                final_answer = generate_answer(query, cleaned_chunk_text)
                
                # Output results
                print(f"Chunk ID: {chunk_id}, Distance: {result['distance']:.4f}")
                print(f"Generated Answer: {final_answer}\n")
            else:
                print(f"⚠️ No 'text' field found for chunk ID: {chunk_id}. Skipping...")
    else:
        print("⚠️ No neighbors found!")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk the query, generate embeddings, and query the Matching Engine.")
    parser.add_argument("--project_id", type=str, required=True, help="Google Cloud Project ID.")
    parser.add_argument("--region", type=str, required=True, help="Google Cloud Region (e.g., 'us-central1').")
    parser.add_argument("--index_endpoint_name", type=str, required=True, help="Name for the Matching Engine endpoint.")
    parser.add_argument("--deployed_index_id", type=str, required=False, default=None, help="ID for the deployed index.")
    parser.add_argument("--query", type=str, required=True, help="Query string for searching the Matching Engine.")
    parser.add_argument("--chunk_size", type=int, help="Max number of words per chunk.")
    parser.add_argument("--top_k", type=int, help="Number of nearest neighbors to retrieve.")
    args = parser.parse_args()
    
    index_endpoint_name = get_index_endpoint_resource_name(
        project_id=args.project_id,
        region=args.region,
        display_name=args.index_endpoint_name,
    )
   
   
    handle_query(
        project_id=args.project_id,
        region=args.region,
        index_endpoint_name=index_endpoint_name,
        deployed_index_id=args.deployed_index_id,  # Optional: Retrieve dynamically if not provided
        query=args.query,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
    )