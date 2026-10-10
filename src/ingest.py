"""Build the FAISS index from the PM-USP CSSS guidelines PDF.
Run: python -m src.ingest
"""
import sys
import json
import re
from pathlib import Path
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding="utf-8")

from src.config import PDF_PATH, INDEX_DIR, EMBED_MODEL

# Section patterns - flexible to match actual PDF format
SECTION_PATTERNS = [
    (r'1\.?\s+Objective', "1", "Objective"),
    (r'2\.?\s+Scope', "2", "Scope"),
    (r'3\.?\s+Allocation\s+of\s+Scholarship', "3", "Allocation of Scholarship"),
    (r'4\.?\s+Eligibility\s+for\s+Scholarship', "4", "Eligibility for Scholarship"),
    (r'5\.?\s+Reservation', "5", "Reservation"),
    (r'6\.?\s+Procedure\s+for\s+Application', "6", "Procedure for Application"),
    (r'7\.?\s+Rate\s+and\s+Duration\s+of\s+Scholarship', "7", "Rate and Duration of Scholarship"),
    (r'8\.?\s+Payment\s+of\s+Scholarship', "8", "Payment of Scholarship"),
    (r'9\.?\s+For\s+Colleges/?Universities', "9", "For Colleges/Universities"),
    (r'10\.?\s+For\s+State\s+Government/?State\s+Boards', "10", "For State Government/State Boards"),
    (r'11\.?\s+Grievance\s+Redressal', "11", "Grievance Redressal")
]

SECTION_4_CLAUSES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
SECTION_6_CLAUSES = ["i", "ii", "iii", "iv", "v", "vi", "vii"]


def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF with normalized whitespace."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Normalize whitespace but keep line structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_by_sections(text):
    """Split text by section headings using regex patterns."""
    sections = {}
    matches = []

    # Find all section matches
    for pattern_str, section_num, section_title in SECTION_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        match = pattern.search(text)
        if match:
            matches.append((match.start(), section_num, section_title, match.end()))

    if not matches:
        raise ValueError("Could not find any section headings in PDF")

    # Sort by position
    matches.sort(key=lambda x: x[0])

    # Everything before first section is title
    first_start = matches[0][0]
    sections["0 Title"] = text[:first_start].strip()

    # Extract each section's text
    for i, (start, section_num, section_title, end_of_heading) in enumerate(matches):
        # Find where this section ends (next section starts or end of text)
        if i + 1 < len(matches):
            section_end = matches[i + 1][0]
        else:
            section_end = len(text)

        section_text = text[end_of_heading:section_end].strip()
        section_key = f"{section_num} {section_title}"
        sections[section_key] = section_text

    return sections


def split_section_4(text):
    """Split section 4 into individual clauses (i to ix)."""
    chunks = []

    # Pattern to find clause markers like "i.", "ii.", etc.
    clause_pattern = re.compile(r'\b(i{1,3}|iv|v|vi{0,3}|ix)\b\.?\s', re.IGNORECASE)

    matches = list(clause_pattern.finditer(text))

    if len(matches) != 9:
        print(f"WARNING: Found {len(matches)} clause markers in Section 4, expected 9")
        # Fallback: return the whole section as one chunk
        return [{
            "id": "sec_4",
            "section_number": "4",
            "section_title": "Eligibility for Scholarship",
            "clause": None,
            "text": f"Section 4 - Eligibility for Scholarship:\n\n{text}",
            "source_file": PDF_PATH.name
        }]

    for idx, match in enumerate(matches):
        clause_num = SECTION_4_CLAUSES[idx]
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        clause_text = text[start:end].strip()

        chunks.append({
            "id": f"sec_4_clause_{clause_num}",
            "section_number": "4",
            "section_title": "Eligibility for Scholarship",
            "clause": clause_num,
            "text": f"Section 4 (Eligibility), clause {clause_num}:\n\n{clause_text}",
            "source_file": PDF_PATH.name
        })

    return chunks


def split_section_6(text):
    """Split section 6 into individual clauses (i to vii)."""
    chunks = []

    clause_pattern = re.compile(r'\b(i{1,3}|iv|v|vi{0,3})\b\.?\s', re.IGNORECASE)

    matches = list(clause_pattern.finditer(text))

    if len(matches) != 7:
        print(f"WARNING: Found {len(matches)} clause markers in Section 6, expected 7")
        return [{
            "id": "sec_6",
            "section_number": "6",
            "section_title": "Procedure for Application",
            "clause": None,
            "text": f"Section 6 - Procedure for Application:\n\n{text}",
            "source_file": PDF_PATH.name
        }]

    for idx, match in enumerate(matches):
        clause_num = SECTION_6_CLAUSES[idx]
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        clause_text = text[start:end].strip()

        chunks.append({
            "id": f"sec_6_clause_{clause_num}",
            "section_number": "6",
            "section_title": "Procedure for Application",
            "clause": clause_num,
            "text": f"Section 6 (Procedure for Application), clause {clause_num}:\n\n{clause_text}",
            "source_file": PDF_PATH.name
        })

    return chunks


def further_split_if_needed(chunk):
    """Split chunks longer than 1200 chars with RecursiveCharacterTextSplitter."""
    if len(chunk["text"]) <= 1200:
        return [chunk]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )

    # Extract the header and body
    lines = chunk["text"].split("\n", 1)
    header = lines[0] if len(lines) > 0 else ""
    body = lines[1] if len(lines) > 1 else chunk["text"]

    sub_texts = splitter.split_text(body)

    sub_chunks = []
    for i, sub_text in enumerate(sub_texts):
        sub_chunk = chunk.copy()
        sub_chunk["id"] = f"{chunk['id']}_part_{i+1}"
        sub_chunk["text"] = f"{header}\n\n{sub_text}"
        sub_chunks.append(sub_chunk)

    return sub_chunks


def main():
    print("=" * 60)
    print("PM-USP CSSS Guidelines Ingestion")
    print("=" * 60)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"\n1. Extracting text from {PDF_PATH.name}...")
    full_text = extract_text_from_pdf(PDF_PATH)
    print(f"   Extracted {len(full_text)} characters")

    print("\n2. Splitting by sections...")
    sections = split_by_sections(full_text)
    print(f"   Found {len(sections)} sections")

    print("\n3. Creating chunks...")
    all_chunks = []
    section_counts = {}

    for section_key, section_text in sections.items():
        # Parse section number and title
        if section_key == "0 Title":
            section_num = "0"
            section_title = "Title"
        else:
            parts = section_key.split(" ", 1)
            section_num = parts[0]
            section_title = parts[1] if len(parts) > 1 else ""

        # Special handling for sections 4 and 6
        if section_num == "4":
            chunks = split_section_4(section_text)
            if len(chunks) != 9:
                raise ValueError(f"Section 4 split produced {len(chunks)} chunks, expected 9")
        elif section_num == "6":
            chunks = split_section_6(section_text)
            if len(chunks) != 7:
                raise ValueError(f"Section 6 split produced {len(chunks)} chunks, expected 7")
        else:
            # Regular section
            chunks = [{
                "id": f"sec_{section_num}",
                "section_number": section_num,
                "section_title": section_title,
                "clause": None,
                "text": f"Section {section_num} - {section_title}:\n\n{section_text}",
                "source_file": PDF_PATH.name
            }]

        # Further split if needed
        final_chunks = []
        for chunk in chunks:
            final_chunks.extend(further_split_if_needed(chunk))

        all_chunks.extend(final_chunks)
        section_counts[section_key] = len(final_chunks)

    print("\n   Chunks per section:")
    for section, count in section_counts.items():
        print(f"   - {section}: {count}")
    print(f"\n   Total chunks: {len(all_chunks)}")

    print("\n4. Generating embeddings...")
    print(f"   Loading embedding model: {EMBED_MODEL}")

    try:
        embedder = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        print(f"\n   ERROR: Could not load embedding model '{EMBED_MODEL}'")
        print(f"   {e}")
        raise

    # For bge models, no prefix for documents, add prefix to queries only
    texts_to_embed = [chunk["text"] for chunk in all_chunks]

    print(f"   Encoding {len(texts_to_embed)} chunks...")
    embeddings = embedder.encode(
        texts_to_embed,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"   Embedding shape: {embeddings.shape}")

    print("\n5. Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
    index.add(embeddings.astype('float32'))

    print(f"   Index contains {index.ntotal} vectors")

    print("\n6. Saving index and metadata...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "index.faiss"
    chunks_path = INDEX_DIR / "chunks.json"

    faiss.write_index(index, str(index_path))

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"   Saved: {index_path}")
    print(f"   Saved: {chunks_path}")

    print("\n" + "=" * 60)
    print("✓ Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
