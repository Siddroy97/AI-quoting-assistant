import re
import sys
import os
import json
import pathlib
import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env from project root regardless of where the script is launched from
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

EXTRACTED_FOLDER = pathlib.Path("extracted")
RULES_OUTPUT_FILE = pathlib.Path("engineering_rules.json")
CHROMA_STORE_FOLDER = "chroma_store"
RULES_COLLECTION_NAME = "engineering_rules"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"


# Extracts the engineering notes and critical design decision sections from one document
def extract_engineering_content(document_text):
    eng_notes = re.search(
        r"(ENGINEERING NOTES.*?)(?=\n\d+\.\s+[A-Z]|OUTCOME:|$)",
        document_text,
        re.DOTALL | re.IGNORECASE,
    )
    cdd = re.search(
        r"(CRITICAL DESIGN DECISION.*?)(?=\n\d+\.\s+[A-Z]|OUTCOME:|$)",
        document_text,
        re.DOTALL | re.IGNORECASE,
    )
    parts = []
    if eng_notes:
        parts.append(eng_notes.group(1).strip())
    if cdd:
        parts.append(cdd.group(1).strip())
    return "\n\n".join(parts) if parts else ""


# Calls Claude to extract a JSON array of engineering rules from one document's engineering content
def extract_rules_from_document(anthropic_client, engineering_content, filename):
    prompt = (
        "You are extracting reusable engineering rules from a TMI Climate Solutions quote document.\n\n"
        "Read the engineering notes and critical design decision sections below, then output a JSON array "
        "of standalone engineering rules. Each rule must be a single sentence that could apply to future "
        "quotes with similar parameters. Focus on rules about: equipment sizing, location-specific "
        "adjustments, redundancy logic, material choices, pricing patterns, and schedule assumptions.\n\n"
        "Output only a JSON array of strings. No other text.\n\n"
        f"SOURCE DOCUMENT: {filename}\n\n"
        f"{engineering_content}"
    )
    try:
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
        return json.loads(raw_text)
    except Exception as error:
        print(f"  Warning: could not extract rules from {filename}: {error}")
        return []


# Calls Claude to deduplicate a flat list of rules and return a condensed unique set
def deduplicate_rules(anthropic_client, all_rules):
    rules_text = "\n".join(f"- {rule}" for rule in all_rules)
    prompt = (
        "You are consolidating a list of engineering rules extracted from multiple TMI Climate Solutions "
        "quote documents. Many rules are duplicates or near-duplicates.\n\n"
        "Remove exact duplicates and merge near-duplicates into a single best-phrased version. "
        "Keep rules that are genuinely distinct. Output only a JSON array of strings. No other text.\n\n"
        f"INPUT RULES:\n{rules_text}"
    )
    try:
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
        return json.loads(raw_text)
    except Exception as error:
        print(f"  Warning: deduplication failed: {error}")
        return all_rules


# Writes the deduplicated rules list to engineering_rules.json in the project root
def save_rules_to_json(rules):
    try:
        RULES_OUTPUT_FILE.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved {len(rules)} rules to {RULES_OUTPUT_FILE}")
    except Exception as error:
        print(f"Error: could not write rules file: {error}")
        sys.exit(1)


# Embeds the rules list and stores them in the ChromaDB engineering_rules collection
def embed_rules_into_chroma(rules):
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_FOLDER)
        try:
            chroma_client.delete_collection(name=RULES_COLLECTION_NAME)
        except Exception:
            pass
        collection = chroma_client.create_collection(name=RULES_COLLECTION_NAME)
        vectors = embedding_model.encode(rules).tolist()
        collection.add(
            ids=[f"rule_{index}" for index in range(len(rules))],
            documents=rules,
            embeddings=vectors,
        )
        print(f"  Embedded {len(rules)} rules into ChromaDB collection '{RULES_COLLECTION_NAME}'")
    except Exception as error:
        print(f"Error: could not embed rules into ChromaDB: {error}")
        sys.exit(1)


# Entry point: reads extracted docs, extracts rules, deduplicates, saves JSON and embeds in ChromaDB
def main():
    extracted_files = sorted(EXTRACTED_FOLDER.glob("*.txt"))
    if not extracted_files:
        print(f"Error: no .txt files found in {EXTRACTED_FOLDER}")
        sys.exit(1)

    anthropic_client = anthropic.Anthropic()
    all_rules = []

    print(f"Step 1/4  Extracting rules from {len(extracted_files)} documents...")
    for txt_file in extracted_files:
        try:
            document_text = txt_file.read_text(encoding="utf-8")
        except Exception as error:
            print(f"  Warning: could not read {txt_file.name}: {error}")
            continue
        engineering_content = extract_engineering_content(document_text)
        if not engineering_content:
            print(f"  Skipping {txt_file.name} — no engineering content found")
            continue
        rules = extract_rules_from_document(anthropic_client, engineering_content, txt_file.name)
        print(f"  {txt_file.name}: {len(rules)} rules extracted")
        all_rules.extend(rules)

    print(f"\nStep 2/4  Deduplicating {len(all_rules)} raw rules...")
    unique_rules = deduplicate_rules(anthropic_client, all_rules)
    print(f"  Reduced to {len(unique_rules)} unique rules")

    print("\nStep 3/4  Saving rules to JSON...")
    save_rules_to_json(unique_rules)

    print("\nStep 4/4  Embedding rules into ChromaDB...")
    embed_rules_into_chroma(unique_rules)

    print("\nDone.")


if __name__ == "__main__":
    main()
