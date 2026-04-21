import asyncio
import time
from typing import List, Dict, Any


def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return value
    future = asyncio.get_event_loop().create_future()
    future.set_result(value)
    return future


def _estimate_cost_usd(tokens_used: int, model_name: str) -> float:
    model_name = (model_name or "").lower()
    rates = {
        "gpt-4o": 0.01,
        "gpt-4o-mini": 0.00015,
        "claude-3-5-sonnet": 0.003,
        "claude-3-sonnet": 0.003,
    }
    rate = 0.001 if not model_name else next((value for key, value in rates.items() if key in model_name), 0.001)
    return round((tokens_used or 0) / 1000.0 * rate, 6)


def _normalize_response(response):
    if isinstance(response, dict):
        return response
    return {
        "answer": getattr(response, "answer", ""),
        "retrieved_chunks": getattr(response, "retrieved_chunks", []),
        "contexts": getattr(response, "contexts", []),
        "metadata": getattr(response, "metadata", {}),
    }

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        raw_response = await self.agent.query(test_case["question"])
        response = _normalize_response(raw_response)
        latency = time.perf_counter() - start_time
        
        # 2. Chạy RAGAS metrics
        ragas_scores = await _maybe_await(self.evaluator.score(test_case, response))
        
        # 3. Chạy Multi-Judge
        judge_result = await _maybe_await(self.judge.evaluate_multi_judge(
            test_case["question"], 
            response["answer"], 
            test_case["expected_answer"]
        ))

        tokens_used = 0
        model_name = ""
        if isinstance(response, dict):
            metadata = response.get("metadata", {}) or {}
            tokens_used = int(metadata.get("tokens_used", 0) or 0)
            model_name = str(metadata.get("model", ""))
        elif hasattr(response, "metadata"):
            metadata = getattr(response, "metadata", {}) or {}
            tokens_used = int(metadata.get("tokens_used", 0) or 0)
            model_name = str(metadata.get("model", ""))
        
        return {
            "question": test_case["question"],
            "expected_answer": test_case.get("expected_answer", ""),
            "expected_chunks": test_case.get("expected_chunks", []),
            "test_case": test_case["question"],
            "agent_response": response.get("answer", ""),
            "retrieved_chunks": response.get("retrieved_chunks", response.get("contexts", [])),
            "latency": latency,
            "tokens_used": tokens_used,
            "estimated_cost_usd": _estimate_cost_usd(tokens_used, model_name),
            "model": model_name,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
            "metrics": {
                "retrieval": ragas_scores.get("retrieval", {}),
                "generation": ragas_scores.get("generation", {}),
            },
        }

    async def evaluate_single(self, test_case: Dict) -> Dict:
        """Backward-compatible alias used by existing tests and tooling."""
        return await self.run_single_test(test_case)

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    async def evaluate_batch(self, dataset: List[Dict], batch_size: int = 5) -> Dict[str, Any]:
        """Run a batch and return aggregate metrics useful for scorecards."""
        results = await self.run_all(dataset, batch_size=batch_size)
        total = len(results) or 1
        return {
            "results": results,
            "aggregate": {
                "avg_score": sum(item["judge"]["final_score"] for item in results) / total,
                "avg_latency": sum(item["latency"] for item in results) / total,
                "avg_cost_usd": sum(item.get("estimated_cost_usd", 0.0) for item in results) / total,
                "avg_hit_rate": sum(item["ragas"]["retrieval"]["hit_rate"] for item in results) / total,
                "avg_mrr": sum(item["ragas"]["retrieval"]["mrr"] for item in results) / total,
                "agreement_rate": sum(item["judge"]["agreement_rate"] for item in results) / total,
                "pass_rate": sum(1 for item in results if item["status"] == "pass") / total,
            },
        }
