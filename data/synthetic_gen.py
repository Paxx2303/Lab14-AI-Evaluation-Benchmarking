import json
import asyncio
import os
import uuid
from typing import List, Dict
from pathlib import Path


CHALLENGING_QUESTIONS = [
    {
        "question": "What are common failure patterns in systems that share similar architectural principles to microservices?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires connecting microservices architecture concepts with failure patterns",
        "adversarial": True,
    },
    {
        "question": "Which evaluation metrics are NOT suitable for measuring retrieval quality in RAG systems, excluding precision and recall?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Must exclude specific metrics - tests ability to filter irrelevant information",
        "adversarial": True,
    },
    {
        "question": "What evaluation approaches were developed after 2020 that address limitations of earlier RAG assessment methods?",
        "type": "temporal_reasoning",
        "difficulty": "hard",
        "explanation": "Requires chronological understanding of evaluation evolution",
        "adversarial": True,
    },
    {
        "question": "Compare the trade-offs between using GPT-4 versus Claude-3 as judge models in terms of cost, accuracy, and bias.",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Must synthesize information from multiple sources for comparison",
        "adversarial": False,
    },
    {
        "question": "In the context of enterprise deployment, what are the key considerations for scaling evaluation systems that differ from research settings?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Meaning changes based on deployment context vs research",
        "adversarial": True,
    },
    {
        "question": "If a RAG system achieves 100% Hit Rate but 0% Faithfulness, what does this indicate about the system's failure mode?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Tests understanding of retrieval vs generation quality relationship",
        "adversarial": True,
    },
    {
        "question": "Which of the following is NOT a valid reason for a retrieval system to return irrelevant chunks?",
        "type": "negative_constraint",
        "difficulty": "medium",
        "explanation": "Requires understanding failure modes vs legitimate edge cases",
        "adversarial": True,
    },
    {
        "question": "Before the introduction of RAGAS, what metrics were primarily used to evaluate retrieval quality?",
        "type": "temporal_reasoning",
        "difficulty": "medium",
        "explanation": "Tests knowledge of pre-2023 evaluation methodology",
        "adversarial": False,
    },
    {
        "question": "How does hybrid search (combining dense and sparse embeddings) compare to pure vector search in handling rare terminology?",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Requires understanding of search algorithm trade-offs",
        "adversarial": False,
    },
    {
        "question": "In a multilingual RAG system, what contextual factors affect chunking strategy that don't apply to monolingual systems?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Tests understanding of multilingual complexity",
        "adversarial": True,
    },
    {
        "question": "What implicit relationships must a RAG system understand to correctly answer questions about software licensing implications?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires connecting legal concepts with technical implementations",
        "adversarial": True,
    },
    {
        "question": "Which techniques are inappropriate for reducing hallucination in RAG outputs, excluding prompt engineering?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Tests knowledge of hallucination mitigation strategies",
        "adversarial": True,
    },
    {
        "question": "How did the introduction of attention mechanisms in 2017 influence modern RAG evaluation practices?",
        "type": "temporal_reasoning",
        "difficulty": "hard",
        "explanation": "Requires understanding transformer history and evaluation evolution",
        "adversarial": True,
    },
    {
        "question": "Compare the effectiveness of BM25 versus semantic search for answering yes/no questions versus open-ended questions.",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Tests understanding of query type interaction with search method",
        "adversarial": False,
    },
    {
        "question": "For medical diagnosis RAG systems, what domain-specific evaluation criteria must be considered that don't apply to general QA?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Tests domain-specific safety and accuracy requirements",
        "adversarial": True,
    },
    {
        "question": "What unstated assumptions must a user make for a RAG system to return a satisfactory answer?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires understanding of implicit system requirements",
        "adversarial": True,
    },
    {
        "question": "Which of these is NOT a valid component of a golden dataset for RAG evaluation?",
        "type": "negative_constraint",
        "difficulty": "medium",
        "explanation": "Tests understanding of dataset construction requirements",
        "adversarial": True,
    },
    {
        "question": "After the 2022 release of ChatGPT, how did RAG evaluation methodologies shift from pre-chatbot era approaches?",
        "type": "temporal_reasoning",
        "difficulty": "hard",
        "explanation": "Tests understanding of LLM impact on evaluation",
        "adversarial": True,
    },
    {
        "question": "Compare chunk overlap strategies: fixed overlap vs semantic boundary detection in terms of context preservation.",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Requires understanding of chunking trade-offs",
        "adversarial": False,
    },
    {
        "question": "In low-resource language RAG systems, what contextual challenges arise that require different evaluation approaches?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Tests understanding of linguistic diversity challenges",
        "adversarial": True,
    },
    {
        "question": "What logical chain of reasoning connects document citation formats to answer attribution accuracy?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires connecting formatting to accuracy",
        "adversarial": True,
    },
    {
        "question": "Which ranking metrics are unsuitable for evaluating cross-lingual retrieval quality?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Tests understanding of cross-lingual challenges",
        "adversarial": True,
    },
    {
        "question": "Before transformers, what retrieval evaluation paradigms existed and how did they influence modern metrics?",
        "type": "temporal_reasoning",
        "difficulty": "hard",
        "explanation": "Tests historical knowledge of IR evaluation",
        "adversarial": False,
    },
    {
        "question": "Compare single-vector-per-document vs multi-vector chunk representation for capturing nuanced meanings.",
        "type": "comparative_analysis",
        "difficulty": "hard",
        "explanation": "Tests understanding of embedding strategies",
        "adversarial": False,
    },
    {
        "question": "For legal document RAG, what contextual factors affect citation precision requirements compared to technical docs?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Tests domain-specific precision requirements",
        "adversarial": True,
    },
    {
        "question": "What hidden relationship exists between chunk size and the ability to answer questions requiring cross-sentence reasoning?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires understanding chunk size impact on reasoning",
        "adversarial": True,
    },
    {
        "question": "Which query expansion techniques are ineffective for highly technical domain-specific terminology?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Tests understanding of domain-specific limitations",
        "adversarial": True,
    },
    {
        "question": "How did the development of attention visualization tools in 2018-2020 improve our ability to evaluate RAG faithfulness?",
        "type": "temporal_reasoning",
        "difficulty": "medium",
        "explanation": "Tests understanding of interpretability tools impact",
        "adversarial": False,
    },
    {
        "question": "Compare human evaluation vs automated metrics for measuring answer completeness in technical documentation.",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Tests understanding of evaluation methodology trade-offs",
        "adversarial": False,
    },
    {
        "question": "In real-time news RAG systems, what contextual latency constraints affect evaluation that don't apply to static corpora?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Tests understanding of temporal relevance challenges",
        "adversarial": True,
    },
    {
        "question": "What causal relationship connects embedding model training data to retrieval bias in specialized domains?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires understanding of embedding bias sources",
        "adversarial": True,
    },
    {
        "question": "Which re-ranking strategies fail to improve results for queries with ambiguous intent?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Tests understanding of re-ranking limitations",
        "adversarial": True,
    },
    {
        "question": "How did the introduction of InstructGPT in 2022 change expectations for RAG answer tone evaluation?",
        "type": "temporal_reasoning",
        "difficulty": "medium",
        "explanation": "Tests understanding of instruction-tuned model impact",
        "adversarial": False,
    },
    {
        "question": "Compare token-limited vs unlimited context windows for long-document RAG in terms of information preservation.",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Tests understanding of context window trade-offs",
        "adversarial": False,
    },
    {
        "question": "For customer support RAG, what contextual factors determine acceptable latency that differ from research benchmarks?",
        "type": "context_dependent",
        "difficulty": "medium",
        "explanation": "Tests understanding of production vs research requirements",
        "adversarial": True,
    },
    {
        "question": "What logical inference connects question complexity to the number of relevant chunks required for accurate answers?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires understanding of complexity-chunk relationship",
        "adversarial": True,
    },
    {
        "question": "Which synthetic data generation methods produce unreliable evaluation results for edge cases?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Tests understanding of synthetic data limitations",
        "adversarial": True,
    },
    {
        "question": "Before RAGAS (2023), what was the standard approach to measuring answer faithfulness to retrieved context?",
        "type": "temporal_reasoning",
        "difficulty": "medium",
        "explanation": "Tests knowledge of pre-RAGAS evaluation",
        "adversarial": False,
    },
    {
        "question": "Compare self-critique vs external judge evaluation for detecting hallucinations in RAG outputs.",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Tests understanding of hallucination detection methods",
        "adversarial": False,
    },
    {
        "question": "In regulated industries, what contextual compliance requirements affect RAG evaluation that don't exist in general use?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Tests understanding of regulatory constraints",
        "adversarial": True,
    },
    {
        "question": "What hidden correlation exists between user query length and retrieval difficulty for complex reasoning tasks?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires understanding of query behavior patterns",
        "adversarial": True,
    },
    {
        "question": "Which annotation guidelines produce inconsistent ground truth labels for context-dependent questions?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Tests understanding of annotation quality issues",
        "adversarial": True,
    },
    {
        "question": "How did the evolution from word embeddings to transformer embeddings (2013-2017) change retrieval evaluation focus?",
        "type": "temporal_reasoning",
        "difficulty": "medium",
        "explanation": "Tests historical understanding of embedding evolution",
        "adversarial": False,
    },
    {
        "question": "Compare topical relevance vs factual accuracy in RAG answer evaluation - which is more critical for trust?",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Tests understanding of answer quality dimensions",
        "adversarial": False,
    },
    {
        "question": "For voice-based RAG interfaces, what contextual factors affect answer formatting requirements compared to text?",
        "type": "context_dependent",
        "difficulty": "medium",
        "explanation": "Tests understanding of multimodal considerations",
        "adversarial": True,
    },
    {
        "question": "What inferred relationship exists between corpus diversity and a RAG system's ability to handle novel queries?",
        "type": "semantic_inference",
        "difficulty": "hard",
        "explanation": "Requires understanding of generalization bounds",
        "adversarial": True,
    },
    {
        "question": "Which evaluation frameworks are unsuitable for comparing RAG systems with different underlying LLMs?",
        "type": "negative_constraint",
        "difficulty": "hard",
        "explanation": "Tests understanding of evaluation design constraints",
        "adversarial": True,
    },
    {
        "question": "Before LLM judges (2023), how were RAG answer quality assessments primarily conducted in production systems?",
        "type": "temporal_reasoning",
        "difficulty": "medium",
        "explanation": "Tests knowledge of pre-LLM-judge evaluation",
        "adversarial": False,
    },
    {
        "question": "Compare top-k selection vs threshold-based retrieval for balancing precision and recall in production RAG.",
        "type": "comparative_analysis",
        "difficulty": "medium",
        "explanation": "Tests understanding of retrieval strategy trade-offs",
        "adversarial": False,
    },
    {
        "question": "For conversational RAG, what contextual factors affect turn-by-turn evaluation that don't apply to single-turn?",
        "type": "context_dependent",
        "difficulty": "hard",
        "explanation": "Tests understanding of conversational context",
        "adversarial": True,
    },
]


async def generate_qa_from_text(
    text: str, num_pairs: int = 5, source: str = "unknown"
) -> List[Dict]:
    """
    Tạo các cặp QA từ danh sách câu hỏi khó thách thức hệ thống RAG.
    Mỗi câu hỏi được thiết kế để đánh lừa hoặc kiểm tra các điểm yếu của hệ thống.
    """
    print(f"Generating {num_pairs} challenging QA pairs...")

    selected_questions = CHALLENGING_QUESTIONS[:num_pairs]

    qa_pairs = []
    for i, q_template in enumerate(selected_questions):
        q_id = str(uuid.uuid4())[:8]
        expected_chunks = [f"chunk_{q_template['type']}_{j}" for j in range(2)]

        difficulty = q_template["difficulty"]
        is_adversarial = q_template.get("adversarial", False)

        qa_pairs.append(
            {
                "id": q_id,
                "question": q_template["question"],
                "expected_answer": f"Answer to '{q_template['question']}' requires {q_template['type']} reasoning at {difficulty} level.",
                "expected_chunks": expected_chunks,
                "expected_retrieval_ids": expected_chunks,
                "contexts": [
                    f"Context chunk for {q_template['type']} reasoning type {j}"
                    for j in range(2)
                ],
                "metadata": {
                    "difficulty": difficulty,
                    "type": q_template["type"],
                    "source": source,
                    "adversarial": is_adversarial,
                    "explanation": q_template["explanation"],
                    "challenge_category": q_template["type"],
                },
                "evaluation_hints": {
                    "semantic_inference": "Check if answer demonstrates implicit relationship understanding",
                    "negative_constraint": "Verify answer excludes the specified constraints",
                    "temporal_reasoning": "Ensure chronological accuracy in answer",
                    "comparative_analysis": "Validate both sides of comparison are addressed",
                    "context_dependent": "Confirm context-appropriate response format",
                }.get(q_template["type"], ""),
            }
        )

    return qa_pairs


async def main():
    """Generate golden dataset with 50+ challenging questions."""
    print("Generating Golden Dataset with challenging questions...")

    all_qa_pairs = []

    sample_text = """
    RAG evaluation measures retrieval and generation quality using metrics like Hit Rate, MRR, and faithfulness.
    Challenging questions test semantic inference, negative constraints, temporal reasoning, comparative analysis, and context-dependence.
    The V2 system uses hybrid search combining BM25 with semantic embeddings plus neural re-ranking.
    Multi-judge consensus combines scores from GPT-4, Claude, and other models with tie-breaking logic.
    """

    all_qa_pairs = await generate_qa_from_text(
        sample_text, num_pairs=50, source="synthetic"
    )

    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    adversarial_count = sum(
        1 for p in all_qa_pairs if p["metadata"].get("adversarial", False)
    )
    print(f"\nGenerated {len(all_qa_pairs)} challenging QA pairs")
    print(f"  - Adversarial questions: {adversarial_count}")
    print(
        f"  - Question types: {len(set(p['metadata']['type'] for p in all_qa_pairs))}"
    )
    print(f"\nSaved to data/golden_set.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
