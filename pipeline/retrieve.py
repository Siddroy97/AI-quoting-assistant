from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_STORE_FOLDER = "chroma_store"
COLLECTION_NAME = "tmi_quotes"
RULES_COLLECTION_NAME = "engineering_rules"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NUM_RESULTS = 3


# Loads the sentence-transformer model and returns it
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# Connects to the local ChromaDB store and returns the collection
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_FOLDER)
    return chroma_client.get_collection(name=COLLECTION_NAME)


# Builds a single descriptive query string from the structured RFQ inputs
def build_query_string(facility_size, cooling_load, config_type, location, redundancy_tier, timeline):
    return (
        f"Data centre cooling project in {location}. "
        f"{cooling_load} cooling load. "
        f"{config_type} configuration. "
        f"{redundancy_tier} redundancy. "
        f"{timeline} timeline. "
        f"{facility_size} facility."
    )


# Converts a ChromaDB L2 distance to a cosine similarity percentage
# Valid because all-MiniLM-L6-v2 produces unit-normalised vectors
def l2_distance_to_similarity_percentage(l2_distance):
    cosine_similarity = 1 - (l2_distance ** 2) / 2
    clamped = max(0.0, min(1.0, cosine_similarity))
    return round(clamped * 100, 1)


# Formats a single ChromaDB result row into the standard return dict
def format_result(document_text, metadata, l2_distance):
    return {
        "filename": metadata.get("filename"),
        "text": document_text,
        "similarity": l2_distance_to_similarity_percentage(l2_distance),
        "metadata": metadata,
    }


# Queries the ChromaDB collection and returns the top 3 nearest neighbours as formatted dicts
def query_collection(collection, embedding_model, query_string):
    query_vector = embedding_model.encode(query_string).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=NUM_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        format_result(document_text, metadata, distance)
        for document_text, metadata, distance in zip(documents, metadatas, distances)
    ]


# Queries the engineering_rules ChromaDB collection and returns the top matching rules as strings
def retrieve_rules(query_text, top_k=3):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_FOLDER)
        collection = chroma_client.get_collection(name=RULES_COLLECTION_NAME)
    except Exception:
        return []
    embedding_model = load_embedding_model()
    query_vector = embedding_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents"],
    )
    return results["documents"][0]


# Retrieves the top 3 most similar historical quotes for the given RFQ inputs
def retrieve(facility_size, cooling_load, config_type, location, redundancy_tier, timeline):
    embedding_model = load_embedding_model()
    collection = get_chroma_collection()
    query_string = build_query_string(
        facility_size, cooling_load, config_type, location, redundancy_tier, timeline
    )
    return query_collection(collection, embedding_model, query_string)
