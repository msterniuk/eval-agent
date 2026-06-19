import os
import pandas as pd
from PyPDF2 import PdfReader
import io  # For GCS bytes wrapper

def list_pdf_files(source_type, source_path, folder_prefix=""):
    """
    List all PDF files from the specified source path.
    - If local: Lists all .pdf files in the given source_path.
    - If GCS: Lists all .pdf files under the bucket and optional folder (using prefix).
    """
    if source_type == "local":
        # List all .pdf files in the local directory
        return [os.path.join(source_path, file) for file in os.listdir(source_path) if file.endswith(".pdf")]
    elif source_type == "gcs":
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(source_path)
        blobs = bucket.list_blobs(prefix=folder_prefix)  # Only list files under the given folder prefix
        return [blob.name for blob in blobs if blob.name.endswith(".pdf")]
    else:
        raise ValueError("source_type must be one of ['local', 'gcs']")

def load_pdf_content(source_type, source_path, file_name):
    """
    Extract text content from a PDF file.
    """
    if source_type == "local":
        pdf_path = os.path.join(source_path, file_name)
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()  # Extract text from each page
        return text
    elif source_type == "gcs":
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(source_path)
        blob = bucket.blob(file_name)

        # Download the PDF as bytes and wrap it in a file-like buffer
        pdf_content = blob.download_as_bytes()  # Raw bytes from GCS
        pdf_file_stream = io.BytesIO(pdf_content)  # Wrap bytes in a file-like object

        # Pass the stream to PdfReader
        reader = PdfReader(pdf_file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text()  # Extract text from each page
        return text
    else:
        raise ValueError("source_type must be one of ['local', 'gcs']")

def preprocess_text(text, chunk_size=300):
    """
    Normalize text and break it into smaller chunks.
    """
    # Normalize text by removing unnecessary characters
    text = text.replace("\n", " ").strip()  # Replace newlines with spaces
    text = " ".join(text.split())  # Remove extra spaces

    # Split the text into smaller chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return [text]

def save_preprocessed_corpus(output_folder, preprocessed_corpus):
    """
    Save preprocessed corpus as a CSV file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "processed_corpus.csv")
    
    # Save as a DataFrame
    pd.DataFrame(preprocessed_corpus).to_csv(output_file, index=False)
    print(f"Corpus has been saved at {output_file}")

def main(source_type, source_path, output_folder, chunk_size=300, folder_prefix=""):
    """
    Main driver function to process PDF files from local or GCS paths into chunks.
    """
    # Get list of PDF files
    file_list = list_pdf_files(source_type, source_path, folder_prefix)
    print(f"Found {len(file_list)} PDF files in '{source_path}/{folder_prefix}'")

    # Load and preprocess each file
    preprocessed_corpus = []
    for file_name in file_list:
        print(f"Processing file: {file_name}")
        content = load_pdf_content(source_type, source_path, file_name)
        chunks = preprocess_text(content, chunk_size)
        for chunk in chunks:
            preprocessed_corpus.append({"file_name": file_name, "chunk": chunk})

    # Save the preprocessed data as a CSV
    save_preprocessed_corpus(output_folder, preprocessed_corpus)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and process PDF files into text chunks.")
    parser.add_argument("--source_type", type=str, required=True, help="Source type: 'local' or 'gcs'")
    parser.add_argument("--source_path", type=str, required=True, help="Path to local folder or GCS bucket")
    parser.add_argument("--output_folder", type=str, required=True, help="Local folder to save the output CSV file")
    parser.add_argument("--chunk_size", type=int, default=300, help="Character size for each text chunk")
    parser.add_argument("--folder_prefix", type=str, default="", help="Prefix or folder path within the GCS bucket")
    args = parser.parse_args()

    main(args.source_type, args.source_path, args.output_folder, args.chunk_size, args.folder_prefix)