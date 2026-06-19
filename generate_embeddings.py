import os
import json
import pandas as pd
from google.cloud import storage, aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from google.api_core.exceptions import AlreadyExists, FailedPrecondition
import time
import datetime

def load_chunks(csv_path):
    """Load text chunks from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df["chunk"].tolist()

def generate_embeddings(chunks, embedding_model_name="text-embedding-005"):
    """Generate embeddings for text chunks using Vertex AI embedding model."""
    model = TextEmbeddingModel.from_pretrained(embedding_model_name)
    embeddings = [model.get_embeddings([chunk])[0] for chunk in chunks]
    return embeddings

def save_embeddings_to_jsonl(chunks, embeddings, output_path):
    """Save embeddings and corresponding chunks to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for i, embedding in enumerate(embeddings):
            record = {
                "id": f"chunk_{i}",
                "embedding": embedding.values,
                "metadata": {"text": chunks[i]}
            }
            f.write(json.dumps(record) + "\n")
    print(f"Embeddings saved locally at: {output_path}")

def upload_to_gcs(local_path, bucket_name, gcs_blob_path):
    """Upload a local file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_path)
    gcs_uri = f"gs://{bucket_name}/{os.path.dirname(gcs_blob_path)}"
    print(f"Uploaded to GCS: {gcs_uri}")
    return gcs_uri

def create_index(display_name, gcs_uri, dimensions):
    """Create a Matching Engine index in Vertex AI."""
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        contents_delta_uri=gcs_uri,
        dimensions=dimensions,
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
    )
    index.wait()
    print(f"Index '{display_name}' created successfully.")
    return index

def create_index_endpoint(display_name):
    """Create an index endpoint in Vertex AI."""
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=display_name,
        public_endpoint_enabled=True,
    )
    print(f"Index Endpoint '{display_name}' created successfully.")
    return index_endpoint

def is_index_deployed(index_endpoint, deployed_index_id):
    return any(d.id == deployed_index_id for d in index_endpoint.deployed_indexes)


def delete_existing_resources(index_name: str, endpoint_name: str):
    print("Checking for existing index and endpoint...")

    # Find existing index
    existing_indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_name}"')
    existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{endpoint_name}"')

    if existing_endpoints:
        endpoint = existing_endpoints[0]
        # Undeploy any deployed indexes
        if endpoint.deployed_indexes:
            print(f"Undeploying indexes from endpoint: {endpoint_name}")
            for deployed in endpoint.deployed_indexes:
                deployed_id = deployed.id
                print(f"Undeploying index with ID: {deployed_id}")
                endpoint.undeploy_index(deployed_index_id=deployed_id)
        print(f"Deleting endpoint: {endpoint_name}")
        endpoint.delete()

    if existing_indexes:
        index = existing_indexes[0]
        print(f"Deleting existing index: {index_name}")
        index.delete()

        
        
def undeploy_if_exists(index_endpoint, deployed_index_id):
    deployed_indexes = index_endpoint.deployed_indexes
    for deployed in deployed_indexes:
        if deployed.id == deployed_index_id:
            print(f"Undeploying existing deployed index with ID: {deployed_index_id}")
            index_endpoint.undeploy_index(deployed_index_id=deployed_index_id)
            return True
    return False

def main(csv_path, project_id, region, bucket_name, output_folder, index_name, deployed_index_id, recreate_index=False):
    aiplatform.init(project=project_id, location=region)

    endpoint_name = f"{index_name}-endpoint"

    chunks = load_chunks(csv_path)
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = generate_embeddings(chunks)

    local_jsonl_path = os.path.join(output_folder, "embeddings.json")
    save_embeddings_to_jsonl(chunks, embeddings, local_jsonl_path)

    gcs_blob_path = "batch_root/embeddings.json"
    gcs_uri = upload_to_gcs(local_jsonl_path, bucket_name, gcs_blob_path)
    print(gcs_uri)
    # Step 1: Handle recreate flag
    if recreate_index:
        delete_existing_resources(index_name, endpoint_name)
        index = create_index(index_name, gcs_uri, dimensions=len(embeddings[0].values))
        index_endpoint = create_index_endpoint(endpoint_name)
    else:
        # Step 2: Reuse existing index or create
        existing_indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_name}"')
        if existing_indexes:
            print(f"Index '{index_name}' already exists. Skipping creation.")
            index = existing_indexes[0]
        else:
            index = create_index(index_name, gcs_uri, dimensions=len(embeddings[0].values))

        # Step 3: Reuse existing endpoint or create
        existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{endpoint_name}"')
        if existing_endpoints:
            print(f"Endpoint '{endpoint_name}' already exists. Skipping creation.")
            index_endpoint = existing_endpoints[0]
        else:
            index_endpoint = create_index_endpoint(endpoint_name)
    
    # Step 4: Check deployment
    if is_index_deployed(index_endpoint, deployed_index_id):
        print(f"Index is already deployed with ID '{deployed_index_id}'. Skipping deployment.")
    else:
        print(f"Index not deployed yet. Deploying index with ID '{deployed_index_id}'...")
        undeploy_if_exists(index_endpoint, deployed_index_id)
       

        # Add timestamp only when recreating
        if args.recreate_index:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            deployed_index_id = f"{args.deployed_index_id}_{timestamp}"
        else:
            deployed_index_id = args.deployed_index_id
        index_endpoint.deploy_index(
            deployed_index_id=deployed_index_id,
            index=index,
            min_replica_count=1,
            max_replica_count=1
        )
        print("Deployed index successfully.")
    


    print("Index deployment process completed.")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings and deploy to Vertex AI Matching Engine.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing text chunks.")
    parser.add_argument("--project_id", type=str, required=True, help="Google Cloud Project ID.")
    parser.add_argument("--region", type=str, required=True, help="Google Cloud Region (e.g., 'us-central1').")
    parser.add_argument("--bucket_name", type=str, required=True, help="GCS bucket name for storing embeddings.")
    parser.add_argument("--output_folder", type=str, required=True, help="Local folder to save embeddings JSONL.")
    parser.add_argument("--index_name", type=str, required=True, help="Name for the Matching Engine index.")
    parser.add_argument("--deployed_index_id", type=str, required=True, help="ID for the deployed index.")
    parser.add_argument("--recreate_index", action="store_true", help="Force delete and recreate the index and endpoint.")
    args = parser.parse_args()

    main(
        args.csv_path,
        args.project_id,
        args.region,
        args.bucket_name,
        args.output_folder,
        args.index_name,
        args.deployed_index_id,
        args.recreate_index,
    )