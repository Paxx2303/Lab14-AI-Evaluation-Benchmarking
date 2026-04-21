import json
import asyncio
import os
import uuid
from typing import List, Dict
from pathlib import Path


# Giả lập việc gọi LLM để tạo dữ liệu (Students will implement this)
async def generate_qa_from_text(
    text: str, num_pairs: int = 5, source: str = "unknown"
) -> List[Dict]:
    """
    TODO: Sử dụng OpenAI/Anthropic API để tạo các cặp (Question, Expected Answer, Context, expected_chunks)
    từ đoạn văn bản cho trước.
    Yêu cầu: Tạo ít nhất 1 câu hỏi 'lừa' (adversarial) hoặc cực khó.

    Returns dict with:
        - question
        - expected_answer
        - expected_chunks (list of chunk IDs that should be retrieved)
        - contexts (list of relevant text snippets, optional)
        - metadata (difficulty, type, etc.)
    """
    print(f"Generating {num_pairs} QA pairs from text of {len(text)} chars...")

    # In a real implementation, you would:
    # 1. Chunk the text using chunking_pipeline
    # 2. For each question, identify which chunks contain the answer
    # 3. Store those chunk IDs as expected_chunks

    # Placeholder: generate sample QA pairs with dummy expected_chunks
    questions = [
        "What is the main topic of this document?",
        "What are the key points discussed?",
        "What conclusions can be drawn from the content?",
        "How does this relate to evaluation systems?",
        "What are the limitations mentioned?",
    ]

    qa_pairs = []
    for i in range(min(num_pairs, len(questions))):
        # Simulate expected chunk IDs (in reality these come from actual document chunks)
        expected_chunks = [
            f"chunk_{i}_{j}" for j in range(2)
        ]  # 2 relevant chunks per question

        qa_pairs.append(
            {
                "id": str(uuid.uuid4())[:8],
                "question": questions[i],
                "expected_answer": f"Based on the document, the answer to '{questions[i]}' involves key information from the text.",
                "expected_chunks": expected_chunks,
                "expected_retrieval_ids": expected_chunks,  # Alias for compatibility
                "contexts": [f"Relevant excerpt {j}" for j in range(2)],
                "metadata": {
                    "difficulty": "medium" if i < 3 else "hard",
                    "type": "factual" if i % 2 == 0 else "analytical",
                    "source": source,
                },
            }
        )

    return qa_pairs


async def main():
    # Discover documents from data/docs to generate QA pairs from actual content
    docs_dir = Path("data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Look for any .txt or .md or .pdf files
    doc_files = []
    for ext in ["*.txt", "*.md", "*.pdf"]:
        doc_files.extend(docs_dir.glob(ext))

    all_qa_pairs = []

    if doc_files:
        print(f"Found {len(doc_files)} documents. Generating QA from each...")
        for doc_file in doc_files:
            try:
                from data.chunking_pipeline import DocumentProcessor

                processor = DocumentProcessor()
                doc_data = processor.process_document(str(doc_file))
                text = doc_data["content"]

                # Generate QA for this document's content
                qa_pairs = await generate_qa_from_text(
                    text,
                    num_pairs=10,  # 10 QA per document to reach 50+ total
                    source=doc_data["source"],
                )
                all_qa_pairs.extend(qa_pairs)
                print(f"  Generated {len(qa_pairs)} QA pairs from {doc_data['source']}")
            except Exception as e:
                print(f"  Error processing {doc_file}: {e}")
                continue
    else:
        print(
            "No documents found in data/docs. Generating synthetic QA from sample text."
        )
        # Fallback: generate from a sample text
        sample_text = """
        RAG evaluation measures retrieval and generation quality using metrics like Hit Rate, MRR, and faithfulness. 
        Challenging questions test semantic inference, negative constraints, temporal reasoning, comparative analysis, and context-dependence. 
        The V2 system uses hybrid search combining BM25 with semantic embeddings plus neural re-ranking. 
        Multi-judge consensus combines scores from GPT-4, Claude, and other models with tie-breaking logic.
        """
        all_qa_pairs = await generate_qa_from_text(
            sample_text, num_pairs=50, source="sample"
        )

    # Ensure we have at least 50 cases
    if len(all_qa_pairs) < 50:
        # Pad with more synthetic
        while len(all_qa_pairs) < 50:
            extra = await generate_qa_from_text(
                "Generic content about AI evaluation systems.",
                num_pairs=5,
                source="synthetic",
            )
            all_qa_pairs.extend(extra)

    # Save to golden set
    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Done! Saved {len(all_qa_pairs)} QA pairs to data/golden_set.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
