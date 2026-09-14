import json
import re
from pathlib import Path


INPUT_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 600
OVERLAP = 100


def clean_text(text):
    """Clean unnecessary whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_references_section(text):
    """
    Detect whether a page has entered the references/bibliography section.
    """
    text_lower = text.lower().strip()

    patterns = [
        r"^references$",
        r"^references\s*$",
        r"^bibliography$",
        r"^literature cited$",
        r"^reference list$",
    ]

    for pattern in patterns:
        if re.search(pattern, text_lower, re.MULTILINE):
            return True

    return False


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Create overlapping chunks from text."""
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def process_paper(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    paper_id = json_file.stem

    all_chunks = []
    references_found = False

    for page in pages:
        page_number = page["page"]
        raw_text = page["text"]

        if not raw_text.strip():
            continue

        # Check whether this page begins the references section
        if is_references_section(raw_text):
            print(f"  References detected on page {page_number}.")
            references_found = True
            break

        text = clean_text(raw_text)

        chunks = create_chunks(text)

        for index, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_p{page_number}_c{index}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "page": page_number,
                "text": chunk
            })

    return all_chunks


def main():
    json_files = [
        f for f in INPUT_DIR.glob("*.json")
        if f.name != "chunks.jsonl"
    ]

    print(f"Found {len(json_files)} papers.\n")

    all_chunks = []

    for json_file in json_files:
        print(f"Processing: {json_file.name}")

        chunks = process_paper(json_file)
        all_chunks.extend(chunks)

        print(f"  Created {len(chunks)} chunks.\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Finished.")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()