"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# =============================================================================
# HELPER: Chuẩn hoá kết quả ChromaDB → list[dict]
# ChromaDB trả về nested lists, cần flatten thành list[dict] thống nhất
# =============================================================================

def _parse_chroma_results(results: Dict) -> List[Dict[str, Any]]:
    """
    Chuyển kết quả ChromaDB (nested list) thành list[dict] chuẩn:
      [{"text": ..., "metadata": ..., "score": ...}, ...]

    ChromaDB distances dùng cosine space → score = 1 - distance
    """
    chunks = []
    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        chunks.append({
            "text":     doc,
            "metadata": meta,
            "score":    round(1.0 - dist, 4),   # cosine similarity
        })
    return chunks


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Returns:
        List[dict] với keys: text, metadata, score
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR
    from chromadb.api.types import EmbeddingFunction, Embeddings, Documents

    class _PassthroughEF(EmbeddingFunction):
        def __init__(self): pass
        def __call__(self, input: Documents) -> Embeddings:
            raise RuntimeError("Should not be called — embeddings provided directly")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(
            name="rag_lab",
            embedding_function=_PassthroughEF(),
        )
    except Exception:
        raise RuntimeError(
            "Collection 'rag_lab' chưa tồn tại.\n"
            "Hãy chạy: python index.py  để build index trước."
        )

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "metadata": meta,
            "score": 1 - dist  # cosine similarity
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
    

    # Flatten ChromaDB nested-list format → list[dict]
    return _parse_chroma_results(results)


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa
    """
    import chromadb
    from rank_bm25 import BM25Okapi
    from index import CHROMA_DB_DIR
    from chromadb.api.types import EmbeddingFunction, Embeddings, Documents

    class _PassthroughEF(EmbeddingFunction):
        def __init__(self): pass
        def __call__(self, input: Documents) -> Embeddings:
            raise RuntimeError("Should not be called — embeddings provided directly")

    # Load toàn bộ chunks từ ChromaDB để build BM25 index
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(
            name="rag_lab",
            embedding_function=_PassthroughEF(),
        )
    except Exception:
        raise RuntimeError(
            "Collection 'rag_lab' chưa tồn tại.\n"
            "Hãy chạy: python index.py  để build index trước."
        )
    all_data  = collection.get(include=["documents", "metadatas"])
    all_docs  = all_data["documents"]
    all_metas = all_data["metadatas"]

    # Build BM25
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25             = BM25Okapi(tokenized_corpus)
    tokenized_query  = query.lower().split()
    scores           = bm25.get_scores(tokenized_query)

    # Lấy top_k indices có score cao nhất
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    max_score   = max(scores) if max(scores) > 0 else 1.0   # tránh chia 0

    return [
        {
            "text":     all_docs[i],
            "metadata": all_metas[i],
            "score":    round(float(scores[i]) / max_score, 4),  # normalize 0-1
        }
        for i in top_indices
    ]


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    RRF_score(doc) = dense_weight  * 1/(60 + dense_rank)
                   + sparse_weight * 1/(60 + sparse_rank)

    Phù hợp khi corpus lẫn ngôn ngữ tự nhiên lẫn tên riêng / mã lỗi / điều khoản.
    """
    RRF_K = 60  # hằng số RRF tiêu chuẩn

    dense_results  = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    # Dùng text làm key để merge (text là unique identifier của chunk)
    rrf_scores: Dict[str, float] = {}
    chunk_map:  Dict[str, Dict]  = {}

    for rank, chunk in enumerate(dense_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight * (1.0 / (RRF_K + rank))
        chunk_map[key]  = chunk

    for rank, chunk in enumerate(sparse_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight * (1.0 / (RRF_K + rank))
        chunk_map[key]  = chunk

    # Sort theo RRF score giảm dần
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

    return [
        {**chunk_map[k], "score": round(rrf_scores[k], 6)}
        for k in sorted_keys
    ]


# =============================================================================
# RERANK (Sprint 3 alternative)
# =============================================================================

# Cache cross-encoder để không load lại nhiều lần
_cross_encoder = None

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.

    Funnel logic: Search rộng (top-10) → Rerank → Select (top-3)
    Dùng khi: dense/hybrid trả về nhiều noise, muốn chắc chắn chunk vào prompt.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("  [Info] Load CrossEncoder model (lần đầu)...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs  = [[query, chunk["text"]] for chunk in candidates]
    scores = _cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    # Ghi lại rerank score để so sánh sau
    return [
        {**chunk, "rerank_score": round(float(score), 4)}
        for chunk, score in ranked[:top_k]
    ]


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion":     Thêm từ đồng nghĩa, alias, tên cũ
      - "decomposition": Tách query phức tạp thành 2-3 sub-queries
      - "hyde":          Sinh câu trả lời giả để embed thay query
    """
    import json

    STRATEGY_PROMPTS = {
        "expansion": (
            f"Given the query: '{query}'\n"
            "Generate 2-3 alternative phrasings or related Vietnamese terms "
            "that could help find relevant documents.\n"
            "Output ONLY a JSON array of strings, no explanation."
        ),
        "decomposition": (
            f"Break down this complex query into 2-3 simpler sub-queries: '{query}'\n"
            "Output ONLY a JSON array of strings, no explanation."
        ),
        "hyde": (
            f"Write a short, factual answer (2-3 sentences) that would ideally answer "
            f"the question: '{query}'\n"
            "Output ONLY a JSON array with one string (the hypothetical answer)."
        ),
    }

    if strategy not in STRATEGY_PROMPTS:
        raise ValueError(f"strategy phải là: {list(STRATEGY_PROMPTS.keys())}")

    raw = call_llm(STRATEGY_PROMPTS[strategy])
    try:
        # Strip markdown fences nếu có
        clean = raw.strip().strip("```json").strip("```").strip()
        queries = json.loads(clean)
        if isinstance(queries, list) and queries:
            return [query] + [str(q) for q in queries]  # luôn giữ query gốc
    except Exception:
        pass  # fallback về query gốc nếu parse lỗi

    return [query]


# =============================================================================
# GENERATION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói chunks thành context block cho prompt.
    Mỗi chunk có số [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta    = chunk.get("metadata", {})
        source  = meta.get("source", "unknown")
        section = meta.get("section", "")
        score   = chunk.get("score", 0)
        text    = chunk.get("text", "")

        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.4f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Grounded prompt theo 4 quy tắc:
    1. Evidence-only  2. Abstain  3. Citation  4. Short & clear
    """
    return f"""Your are an internal IT Helpdesk assistant. Answer only from the retrieved context below.
If the context is insufficient to answer the question, say that you cannot find the information in the document, show the revelant information you have found so far and finally ask for IT Helpdesk support.
Cite the source number in brackets like [1] when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.
    Ưu tiên OpenAI nếu có OPENAI_API_KEY, fallback sang Gemini.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        # Option A — OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,      # ổn định, dễ đánh giá
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    elif google_key:
        # Option B — Google Gemini
        import google.generativeai as genai
        genai.configure(api_key=google_key)
        model    = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    else:
        raise EnvironmentError(
            "Không tìm thấy API key. "
            "Hãy set OPENAI_API_KEY hoặc GOOGLE_API_KEY trong file .env"
        )


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    query_transform: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → (transform) → retrieve → (rerank) → generate.

    Args:
        query:            Câu hỏi
        retrieval_mode:   "dense" | "sparse" | "hybrid"
        top_k_search:     Số chunk lấy từ vector store
        top_k_select:     Số chunk đưa vào prompt (sau rerank/select)
        use_rerank:       Có dùng cross-encoder rerank không
        query_transform:  None | "expansion" | "decomposition" | "hyde"
        verbose:          In thêm thông tin debug

    Returns:
        Dict: answer, sources, chunks_used, query, config
    """
    config = {
        "retrieval_mode":   retrieval_mode,
        "top_k_search":     top_k_search,
        "top_k_select":     top_k_select,
        "use_rerank":       use_rerank,
        "query_transform":  query_transform,
    }

    # --- Bước 1: Query Transformation (optional) ---
    queries = [query]
    if query_transform:
        queries = transform_query(query, strategy=query_transform)
        if verbose:
            print(f"[RAG] Transformed queries: {queries}")

    # --- Bước 2: Retrieve (merge kết quả nếu có nhiều sub-queries) ---
    retrieve_fn = {
        "dense":  retrieve_dense,
        "sparse": retrieve_sparse,
        "hybrid": retrieve_hybrid,
    }.get(retrieval_mode)

    if retrieve_fn is None:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if len(queries) == 1:
        candidates = retrieve_fn(queries[0], top_k=top_k_search)
    else:
        # Nhiều sub-queries: merge và deduplicate theo text
        seen:       Dict[str, Dict] = {}
        for q in queries:
            for chunk in retrieve_fn(q, top_k=top_k_search):
                key = chunk["text"]
                # Giữ chunk có score cao hơn nếu trùng
                if key not in seen or chunk["score"] > seen[key]["score"]:
                    seen[key] = chunk
        # Sort lại theo score
        candidates = sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:top_k_search]

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.4f} | {c['metadata'].get('source', '?')} | {c['metadata'].get('section', '')}")

    # --- Bước 3: Rerank hoặc truncate ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks going into prompt")

    # --- Bước 4: Build context + prompt ---
    context_block = build_context_block(candidates)
    prompt        = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt preview:\n{prompt[:600]}...\n")

    # --- Bước 5: Generate ---
    answer = call_llm(prompt)

    # --- Bước 6: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query":       query,
        "answer":      answer,
        "sources":     sources,
        "chunks_used": candidates,
        "config":      config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.
    A/B Rule: chỉ đổi retrieval_mode, giữ nguyên các tham số còn lại.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = [
        {"label": "Baseline  — dense",          "retrieval_mode": "dense",  "use_rerank": False},
        {"label": "Variant A — hybrid",          "retrieval_mode": "hybrid", "use_rerank": False},
        {"label": "Variant B — dense + rerank",  "retrieval_mode": "dense",  "use_rerank": True},
    ]

    for cfg in strategies:
        label = cfg.pop("label")
        print(f"\n--- {label} ---")
        try:
            result = rag_answer(query, **cfg, verbose=False)
            print(f"Answer : {result['answer']}")
            print(f"Sources: {result['sources']}")
            chunks = result["chunks_used"]
            print(f"Chunks : {len(chunks)} used")
            for i, c in enumerate(chunks):
                sc = c.get("rerank_score", c.get("score", 0))
                print(f"  [{i+1}] score={sc:.4f} | {c['metadata'].get('source','?')} | {c['metadata'].get('section','')}")
        except EnvironmentError as e:
            print(f"  Config lỗi: {e}")
        except Exception as e:
            print(f"  Lỗi: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",   # không có trong docs → phải abstain
    ]

    print("\n--- Sprint 2: Baseline Dense ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer : {result['answer']}")
            print(f"Sources: {result['sources']}")
        except EnvironmentError as e:
            print(f"  {e}")
        except Exception as e:
            print(f"  Lỗi: {e}")

    print("\n--- Sprint 3: So sánh strategies ---")
    compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    compare_retrieval_strategies("ERR-403-AUTH")
