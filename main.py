import asyncio
import json
import os
import time
from statistics import mean

from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator

def _build_summary(results, agent_version: str, baseline_version: str = "Agent_V1_Base"):
    total = len(results) or 1
    avg_score = mean(r["judge"]["final_score"] for r in results)
    avg_hit_rate = mean(r["ragas"]["retrieval"]["hit_rate"] for r in results)
    avg_mrr = mean(r["ragas"]["retrieval"]["mrr"] for r in results)
    avg_agreement = mean(r["judge"]["agreement_rate"] for r in results)
    avg_latency = mean(r["latency"] for r in results)
    avg_cost = mean(r.get("estimated_cost_usd", 0.0) for r in results)
    pass_rate = sum(1 for r in results if r["status"] == "pass") / total

    return {
        "metadata": {
            "version": agent_version,
            "baseline_version": baseline_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(avg_score, 3),
            "hit_rate": round(avg_hit_rate, 3),
            "mrr": round(avg_mrr, 3),
            "agreement_rate": round(avg_agreement, 3),
            "avg_latency": round(avg_latency, 3),
            "avg_cost_usd": round(avg_cost, 6),
            "pass_rate": round(pass_rate, 3),
        },
    }


def _release_gate(v1_summary, v2_summary):
    v1 = v1_summary["metrics"]
    v2 = v2_summary["metrics"]
    delta_score = round(v2["avg_score"] - v1["avg_score"], 3)
    delta_hit_rate = round(v2["hit_rate"] - v1["hit_rate"], 3)
    delta_agreement = round(v2["agreement_rate"] - v1["agreement_rate"], 3)
    delta_cost = round(v2["avg_cost_usd"] - v1["avg_cost_usd"], 6)
    delta_latency = round(v2["avg_latency"] - v1["avg_latency"], 3)

    approved = (
        v2["avg_score"] >= 3.5
        and v2["agreement_rate"] >= 0.7
        and v2["hit_rate"] >= 0.5
        and v2["avg_latency"] <= max(v1["avg_latency"] * 1.3, v1["avg_latency"] + 0.5)
        and v2["avg_cost_usd"] <= max(v1["avg_cost_usd"] * 1.3, v1["avg_cost_usd"] + 0.01)
    )

    return {
        "decision": "APPROVE" if approved else "BLOCK RELEASE",
        "thresholds": {
            "min_avg_score": 3.5,
            "min_agreement_rate": 0.7,
            "min_hit_rate": 0.5,
            "max_latency_multiplier": 1.3,
            "max_cost_multiplier": 1.3,
        },
        "delta": {
            "avg_score": delta_score,
            "hit_rate": delta_hit_rate,
            "agreement_rate": delta_agreement,
            "avg_cost_usd": delta_cost,
            "avg_latency": delta_latency,
        },
    }

async def run_benchmark_with_results(agent_version: str):
    print(f" Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print(" Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print(" File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), RetrievalEvaluator(), LLMJudge())
    results = await runner.run_all(dataset)

    summary = _build_summary(results, agent_version)
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print(" Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    gate = _release_gate(v1_summary, v2_summary)
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {gate['delta']['avg_score']:+.2f}")
    print(f"Hit Rate Delta: {gate['delta']['hit_rate']:+.2f}")
    print(f"Agreement Delta: {gate['delta']['agreement_rate']:+.2f}")
    print(f"Cost Delta: {gate['delta']['avg_cost_usd']:+.6f} USD")
    print(f"Latency Delta: {gate['delta']['avg_latency']:+.3f}s")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        enriched_summary = dict(v2_summary)
        enriched_summary["regression"] = {
            "baseline_version": v1_summary["metadata"]["version"],
            "gate": gate,
        }
        json.dump(enriched_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print(f" QUYẾT ĐỊNH: {gate['decision']}")

if __name__ == "__main__":
    asyncio.run(main())
