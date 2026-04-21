from typing import List, Dict, Any
import numpy as np


class RetrievalEvaluator:
    """
    Retrieval evaluator computing Hit Rate, MRR, and Precision@k
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def calculate_hit_rate(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None
    ) -> float:
        """
        Hit Rate@k: Is at least one relevant chunk in top-k?
        """
        if top_k is None:
            top_k = self.top_k
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank: 1 / rank_of_first_relevant_chunk
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_precision_at_k(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None
    ) -> float:
        """
        Precision@k: fraction of retrieved chunks that are relevant
        """
        if top_k is None:
            top_k = self.top_k
        top_retrieved = retrieved_ids[:top_k]
        if not top_retrieved:
            return 0.0
        relevant_count = sum(1 for doc_id in top_retrieved if doc_id in expected_ids)
        return relevant_count / len(top_retrieved)

    def calculate_recall_at_k(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None
    ) -> float:
        """
        Recall@k: fraction of relevant chunks that are retrieved
        """
        if top_k is None:
            top_k = self.top_k
        top_retrieved = set(retrieved_ids[:top_k])
        if not expected_ids:
            return 0.0
        retrieved_relevant = sum(
            1 for doc_id in expected_ids if doc_id in top_retrieved
        )
        return retrieved_relevant / len(expected_ids)

    def calculate_f1_at_k(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None
    ) -> float:
        """
        F1@k: harmonic mean of precision and recall at k
        """
        precision = self.calculate_precision_at_k(expected_ids, retrieved_ids, top_k)
        recall = self.calculate_recall_at_k(expected_ids, retrieved_ids, top_k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    async def score(
        self, test_case: Dict[str, Any], agent_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute RAG metrics for a single test case.
        Returns structure compatible with main.py summary aggregation:
        {
            "retrieval": {"hit_rate": ..., "mrr": ..., ...},
            "generation": {"faithfulness": ..., "relevancy": ...}
        }
        """
        expected = test_case.get(
            "expected_chunks", test_case.get("expected_retrieval_ids", [])
        )

        # Extract retrieved chunk IDs from agent response
        retrieved_chunks = agent_response.get(
            "retrieved_chunks", agent_response.get("contexts", [])
        )

        if (
            isinstance(retrieved_chunks, list)
            and retrieved_chunks
            and isinstance(retrieved_chunks[0], dict)
        ):
            retrieved_ids = [
                chunk.get("chunk_id", chunk.get("id", "")) for chunk in retrieved_chunks
            ]
        else:
            retrieved_ids = [str(c) for c in retrieved_chunks]

        hit = self.calculate_hit_rate(expected, retrieved_ids, self.top_k)
        mrr = self.calculate_mrr(expected, retrieved_ids)
        precision = self.calculate_precision_at_k(expected, retrieved_ids, self.top_k)
        recall = self.calculate_recall_at_k(expected, retrieved_ids, self.top_k)
        f1 = self.calculate_f1_at_k(expected, retrieved_ids, self.top_k)
        retrieval_quality = (hit + mrr + f1) / 3.0

        return {
            "retrieval": {
                "hit_rate": hit,
                "mrr": mrr,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_at_k": f1,
                "quality_score": retrieval_quality,
            },
            "generation": {
                # Placeholder generation metrics; would be computed by a separate judge
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "overall": 0.0,
            },
        }

    async def evaluate_batch(
        self, dataset: List[Dict], agent_responses: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval across a batch of cases.
        Returns aggregate metrics and per-case breakdown using nested metric structure.
        """
        if len(dataset) != len(agent_responses):
            raise ValueError("Dataset and responses must have same length")

        per_case = []
        for case, resp in zip(dataset, agent_responses):
            case_metrics = await self.score(case, resp)
            per_case.append(case_metrics)

        n = len(per_case)
        # Aggregate retrieval sub-metrics
        agg = {
            "avg_hit_rate": sum(m["retrieval"]["hit_rate"] for m in per_case) / n,
            "avg_mrr": sum(m["retrieval"]["mrr"] for m in per_case) / n,
            "avg_precision": sum(m["retrieval"]["precision_at_k"] for m in per_case)
            / n,
            "avg_recall": sum(m["retrieval"]["recall_at_k"] for m in per_case) / n,
            "avg_f1": sum(m["retrieval"]["f1_at_k"] for m in per_case) / n,
            "retrieval_quality": sum(m["retrieval"]["quality_score"] for m in per_case)
            / n,
        }

        return {"per_case": per_case, "aggregate": agg}

    def extract_chunk_ids(self, chunks: List[Any]) -> List[str]:
        """Helper: extract string IDs from chunk dicts or strings"""
        if not chunks:
            return []
        if isinstance(chunks[0], str):
            return chunks
        return [c.get("chunk_id", c.get("id", str(c))) for c in chunks]
