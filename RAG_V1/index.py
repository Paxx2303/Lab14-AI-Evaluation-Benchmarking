"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            if line.startswith("Source:"):
                metadata["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Department:"):
                metadata["department"] = line.replace("Department:", "").strip()
            elif line.startswith("Effective Date:"):
                metadata["effective_date"] = line.replace("Effective Date:", "").strip()
            elif line.startswith("Access:"):
                metadata["access"] = line.replace("Access:", "").strip()
            elif line.startswith("==="):
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or line.isupper():
                continue
            else:
                content_lines.append(line)
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)
    # Normalize: tối đa 2 dòng trống liên tiếp, strip trailing spaces mỗi dòng
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = "\n".join(l.rstrip() for l in cleaned_text.split("\n"))

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.
    - Split theo heading "=== ... ===" trước
    - Nếu section quá dài thì split tiếp theo paragraph với overlap
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Split theo pattern "=== ... ==="
    # re.split với capturing group giữ lại delimiter trong list kết quả
    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            # Flush section hiện tại trước khi bắt đầu section mới
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Cập nhật tên section mới (bỏ dấu "===" hai đầu)
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Flush section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Split text thành chunks theo paragraph, với overlap giữa các chunk.

    Chiến lược:
    1. Split text thành paragraphs (tách tại dòng trống "\n\n")
    2. Ghép các paragraphs lại cho đến khi gần đủ chunk_chars
    3. Khi đủ size: đóng chunk, lấy overlap từ đoạn cuối sang chunk mới
    """
    # Nếu toàn bộ section vừa một chunk → trả về ngay
    if len(text) <= chunk_chars:
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_paras: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Nếu thêm paragraph này vào sẽ vượt giới hạn → flush chunk hiện tại
        if current_len + para_len > chunk_chars and current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append({
                "text": chunk_text,
                "metadata": {**base_metadata, "section": section},
            })

            # Tính overlap: lấy các paragraph cuối sao cho tổng ≤ overlap_chars
            overlap_paras: List[str] = []
            overlap_len = 0
            for p in reversed(current_paras):
                if overlap_len + len(p) <= overlap_chars:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break

            current_paras = overlap_paras
            current_len = overlap_len

        current_paras.append(para)
        current_len += para_len

    # Flush phần còn lại
    if current_paras:
        chunks.append({
            "text": "\n\n".join(current_paras),
            "metadata": {**base_metadata, "section": section},
        })

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# =============================================================================

# Cache model Sentence Transformers để không load lại nhiều lần
_st_model = None

def get_embedding(text: str) -> List[float]:
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        # Option A — OpenAI Embeddings
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    else:
        # Option B — Sentence Transformers (local, không cần API key)
        global _st_model
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            print("  [Info] Load Sentence Transformers model (lần đầu)...")
            _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return _st_model.encode(text).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store vào ChromaDB.
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo ChromaDB persistent client (tương thích 1.x)
    client = chromadb.PersistentClient(path=str(db_dir))

    # ChromaDB 1.x bắt buộc truyền embedding_function rõ ràng.
    # Dùng class rỗng để bypass vì embedding thật được inject trong upsert().
    from chromadb.api.types import EmbeddingFunction, Embeddings, Documents

    class _PassthroughEF(EmbeddingFunction):
        def __init__(self): pass
        def __call__(self, input: Documents) -> Embeddings:
            raise RuntimeError("Should not be called — embeddings provided directly")

    collection = client.get_or_create_collection(
        name="rag_lab",
        embedding_function=_PassthroughEF(),
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"\n  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        # Preprocess
        doc = preprocess_document(raw_text, str(filepath))
        print(f"    Metadata: {doc['metadata']}")

        # Chunk
        chunks = chunk_document(doc)
        print(f"    → {len(chunks)} chunks")

        # Embed và upsert từng chunk vào ChromaDB
        ids        = []
        embeddings = []
        documents  = []
        metadatas  = []

        for i, chunk in enumerate(chunks):
            chunk_id  = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        # Batch upsert (hiệu quả hơn upsert từng cái)
        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        total_chunks += len(chunks)
        print(f"    ✓ Upserted {len(chunks)} chunks vào ChromaDB")

    print(f"\n{'='*50}")
    print(f"Hoàn thành! Tổng số chunks đã index: {total_chunks}")
    print(f"Vector store lưu tại: {db_dir}")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source:         {meta.get('source', 'N/A')}")
            print(f"  Section:        {meta.get('section', 'N/A')}")
            print(f"  Department:     {meta.get('department', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Access:         {meta.get('access', 'N/A')}")
            print(f"  Text preview:   {doc[:150]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        total = len(results["metadatas"])
        print(f"\nTổng chunks: {total}")

        departments   = {}
        sources       = {}
        missing_date  = 0
        missing_source = 0

        for meta in results["metadatas"]:
            # Đếm theo department
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1

            # Đếm theo source file
            src = Path(meta.get("source", "unknown")).name
            sources[src] = sources.get(src, 0) + 1

            # Kiểm tra missing fields
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1
            if meta.get("source") in ("unknown", "", None):
                missing_source += 1

        print("\nPhân bố theo department:")
        for dept, count in sorted(departments.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {dept:<30} {count:>4} chunks  ({pct:.1f}%)")

        print("\nPhân bố theo source file:")
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {src:<40} {count:>4} chunks")

        print(f"\nChunks thiếu effective_date : {missing_date}/{total}")
        print(f"Chunks thiếu source         : {missing_source}/{total}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build index
    print("\n--- Build Full Index ---")
    build_index()

    # Bước 4: Kiểm tra index
    print("\n--- Kiểm tra Index ---")
    list_chunks(n=5)
    inspect_metadata_coverage()

    print("\n✓ Sprint 1 hoàn thành!")