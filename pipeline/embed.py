import pathlib
import re
from sentence_transformers import SentenceTransformer
import chromadb

EXTRACTED_FOLDER = pathlib.Path("extracted")
CHROMA_STORE_FOLDER = "chroma_store"
COLLECTION_NAME = "tmi_quotes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EXPECTED_RECORD_COUNT = 30


# Connects to the local persistent ChromaDB store and returns the collection
def get_chroma_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_FOLDER)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        return collection
    except Exception as error:
        print(f"Error: could not connect to ChromaDB at '{CHROMA_STORE_FOLDER}': {error}")
        raise


# Reads the full text content of a single .txt file
def read_text_file(file_path):
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as error:
        print(f"Error: could not read file '{file_path.name}': {error}")
        return None


# Encodes text to an embedding vector using the local sentence-transformers model
def embed_text(embedding_model, text):
    try:
        return embedding_model.encode(text).tolist()
    except Exception as error:
        print(f"Error: embedding failed: {error}")
        return None


# Extracts the outcome (win or loss) from the document text
def parse_outcome(document_text):
    if "OUTCOME: WON" in document_text:
        return "won"
    if "OUTCOME: LOST" in document_text:
        return "lost"
    return None


# Extracts the site location from the document text
def parse_location(document_text):
    match = re.search(r"Site Location:\s*(.+)", document_text)
    if match:
        return match.group(1).strip()
    return None


# Extracts the cooling load in kW from the application line in the document text
def parse_cooling_load(document_text):
    match = re.search(r"Application:.*?(\d[\d,]*)\s*kW", document_text)
    if match:
        return match.group(1).replace(",", "") + " kW"
    return None


# Builds a metadata dictionary for a document, including only fields that were successfully parsed
def build_metadata(filename, document_text):
    metadata = {"filename": filename}
    outcome = parse_outcome(document_text)
    location = parse_location(document_text)
    cooling_load = parse_cooling_load(document_text)
    if outcome:
        metadata["outcome"] = outcome
    if location:
        metadata["location"] = location
    if cooling_load:
        metadata["cooling_load"] = cooling_load
    return metadata


# Embeds a single document and stores it in the ChromaDB collection
def embed_and_store_document(txt_file, collection, embedding_model):
    document_text = read_text_file(txt_file)
    if document_text is None:
        return False

    embedding_vector = embed_text(embedding_model, document_text)
    if embedding_vector is None:
        return False

    document_id = txt_file.stem
    metadata = build_metadata(txt_file.name, document_text)

    try:
        collection.add(
            ids=[document_id],
            embeddings=[embedding_vector],
            documents=[document_text],
            metadatas=[metadata]
        )
        print(f"Stored: {txt_file.name} | outcome={metadata.get('outcome')} | location={metadata.get('location')} | cooling_load={metadata.get('cooling_load')}")
        return True
    except Exception as error:
        print(f"Error: could not store '{txt_file.name}' in ChromaDB: {error}")
        return False


# Loops through all .txt files in the extracted folder and embeds each one
def embed_all_documents(collection, embedding_model):
    txt_files = sorted(EXTRACTED_FOLDER.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{EXTRACTED_FOLDER}'")
        return

    success_count = 0
    failure_count = 0

    for txt_file in txt_files:
        succeeded = embed_and_store_document(txt_file, collection, embedding_model)
        if succeeded:
            success_count += 1
        else:
            failure_count += 1

    print(f"\nEmbedding complete: {success_count} stored successfully, {failure_count} failed.")


# Entry point: checks for existing records, then embeds and stores all documents
def main():
    collection = get_chroma_collection()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    existing_count = collection.count()
    if existing_count >= EXPECTED_RECORD_COUNT:
        print(f"Collection '{COLLECTION_NAME}' already contains {existing_count} records. Nothing to do.")
        return

    embed_all_documents(collection, embedding_model)

    final_count = collection.count()
    print(f"Total records in collection: {final_count}")

    if final_count < EXPECTED_RECORD_COUNT:
        print(f"Warning: expected {EXPECTED_RECORD_COUNT} records but found {final_count}.")


if __name__ == "__main__":
    main()
