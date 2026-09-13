import fitz
from pathlib import Path


PAPERS_DIR = Path("data/papers")
OUTPUT_DIR = Path("data/processed")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text
        })

    doc.close()

    return pages


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")

        pages = extract_text_from_pdf(pdf)

        output_file = OUTPUT_DIR / f"{pdf.stem}.json"

        import json

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()