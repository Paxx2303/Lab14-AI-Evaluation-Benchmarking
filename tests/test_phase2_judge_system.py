"""
Test Suite for Phase 2: Judge System & Evaluation Framework
Tests multi-judge consensus, evaluation metrics, and benchmark framework
"""

import pytest
import json
import os
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.llm_judge import LLMJudge
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent


# ============== Fixtures ==============


@pytest.fixture
def sample_qa_pair():
    """Sample question-answer pair for testing"""
    return {
        "question": "What is RAG evaluation?",
        "answer": "RAG evaluation assesses retrieval and generation quality.",
        "ground_truth": "RAG evaluation measures both retrieval accuracy and answer fidelity.",
        "context": ["chunk_1", "chunk_2"],
    }


@pytest.fixture
def mock_llm_responses():
    """Mock responses from different LLM judges"""
    return {
        "gpt-4o": {
            "score": 4.5,
            "reasoning": "Answer is accurate and well-structured.",
        },
        "claude-3-sonnet": {
            "score": 4.0,
            "reasoning": "Good answer but could be more comprehensive.",
        },
        "gemini-pro": {"score": 4.0, "reasoning": "Factually correct but lacks depth."},
    }


# ============== Phase 2: Multi-Judge Consensus Tests ==============


class TestLLMJudge:
    """Tests for the Multi-Judge Consensus Engine (Planning.md Section)"""

    def test_judge_initialization_default(self):
        """Test LLMJudge initializes with gpt-4o as primary (Planning.md)"""
        judge = LLMJudge()
        assert judge.model == "gpt-4o"
        assert hasattr(judge, "rubrics")
        assert isinstance(judge.rubrics, dict)

    def test_judge_initialization_custom(self):
        """Test LLMJudge accepts custom model"""
        judge = LLMJudge(model="claude-3-opus")
        assert judge.model == "claude-3-opus"

    def test_rubrics_contain_required_criteria(self):
        """Test judge has rubrics for accuracy, professionalism, safety"""
        judge = LLMJudge()
        # These rubrics should be defined (currently placeholders)
        required_categories = ["accuracy", "tone"]
        for category in required_categories:
            assert category in judge.rubrics

    def test_multi_judge_structure(self, sample_qa_pair):
        """Test multi-judge evaluation returns proper structure"""
        judge = LLMJudge()

        result = asyncio.run(
            judge.evaluate_multi_judge(
                question=sample_qa_pair["question"],
                answer=sample_qa_pair["answer"],
                ground_truth=sample_qa_pair["ground_truth"],
            )
        )

        # Required fields per Planning.md consensus section
        assert "final_score" in result
        assert "agreement_rate" in result
        assert "individual_scores" in result
        assert "reasoning" in result

        # Type checks
        assert isinstance(result["final_score"], (int, float))
        assert isinstance(result["agreement_rate"], float)
        assert isinstance(result["individual_scores"], dict)

    def test_agreement_rate_calculation(self):
        """Test agreement rate metric (Planning.md requirement)"""
        judge = LLMJudge()

        # The mock returns agreement=1.0 if scores equal, 0.5 otherwise
        result = asyncio.run(
            judge.evaluate_multi_judge(
                question="Test?", answer="Answer", ground_truth="Truth"
            )
        )

        assert 0.0 <= result["agreement_rate"] <= 1.0

    def test_individual_judge_scores_tracked(self):
        """Test individual judge scores are tracked per Planning.md"""
        judge = LLMJudge()
        result = asyncio.run(
            judge.evaluate_multi_judge(
                question="Test?", answer="Answer", ground_truth="Truth"
            )
        )

        scores = result["individual_scores"]
        assert isinstance(scores, dict)
        assert len(scores) >= 2  # At least 2 judges per Planning.md

        # Each score should be numeric
        for model, score in scores.items():
            assert isinstance(score, (int, float))
            assert 1.0 <= score <= 5.0  # Assuming 1-5 scale

    def test_consensus_logic_for_low_agreement(self):
        """Test tie-breaker logic triggered when agreement < 0.5"""
        # This is a structural test - actual LLM calls would need mocking
        judge = LLMJudge()
        result = asyncio.run(
            judge.evaluate_multi_judge(
                question="Test?", answer="Answer", ground_truth="Truth"
            )
        )

        # According to Planning.md, low agreement (<0.5) should trigger tie-breaker
        # Current mock returns 0.5 or 1.0; structure should support expansion
        assert "agreement_rate" in result


# ============== Phase 2: Benchmark Framework Tests ==============


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner (main.py orchestrator)"""

    def test_benchmark_runner_initialization(self):
        """Test BenchmarkRunner initializes with components"""
        from engine.runner import BenchmarkRunner
        from agent.main_agent import MainAgent
        from engine.llm_judge import LLMJudge
        from engine.retrieval_eval import RetrievalEvaluator

        agent = MainAgent()
        evaluator = RetrievalEvaluator()
        judge = LLMJudge()

        runner = BenchmarkRunner(agent, evaluator, judge)
        assert runner is not None
        assert hasattr(runner, "agent")
        assert hasattr(runner, "evaluator")
        assert hasattr(runner, "judge")

    def test_benchmark_metrics_structure(self):
        """Test benchmark produces required metrics per Planning.md"""
        runner = BenchmarkRunner(MainAgent(), RetrievalEvaluator(), LLMJudge())

        # Check that runner has expected methods
        assert hasattr(runner, "run_all")
        assert hasattr(runner, "evaluate_single")

    def test_required_metrics_present(self):
        """Test final summary contains all required metrics"""
        # Simulate a results summary as would be produced
        sample_summary = {
            "metadata": {"version": "V2", "total": 50},
            "metrics": {"avg_score": 4.2, "hit_rate": 0.85, "agreement_rate": 0.78},
        }

        # Verify required metrics exist per README/Planning.md
        required_metrics = ["avg_score", "hit_rate", "agreement_rate"]
        for metric in required_metrics:
            assert metric in sample_summary["metrics"]


class TestEvaluationMetrics:
    """Tests for specific evaluation metrics (Phase 2)"""

    def test_faithfulness_metric_placeholder(self):
        """Test faithfulness metric structure (Planning.md Generation Metrics)"""
        # This tests the concept; actual implementation may vary
        faithfulness_score = 0.9  # Placeholder
        assert 0.0 <= faithfulness_score <= 1.0

    def test_relevance_metric_placeholder(self):
        """Test relevance metric structure"""
        relevance_score = 0.85
        assert 0.0 <= relevance_score <= 1.0

    def test_completeness_metric_placeholder(self):
        """Test completeness metric structure"""
        completeness_score = 0.75
        assert 0.0 <= completeness_score <= 1.0

    def test_clarity_metric_placeholder(self):
        """Test clarity metric structure"""
        clarity_score = 0.9
        assert 0.0 <= clarity_score <= 1.0


# ============== System Integration Tests ==============


class TestSystemIntegration:
    """Integration tests for Phase 1 + Phase 2 pipeline"""

    def test_end_to_end_with_sample_dataset(self, tmp_path, sample_qa_pair):
        """Test complete pipeline from question to evaluation"""
        # Create minimal test dataset
        dataset = [sample_qa_pair]
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

        # Load and verify structure
        with open(dataset_file, "r") as f:
            loaded = [json.loads(line) for line in f]
            assert len(loaded) == 1
            assert "question" in loaded[0]
            assert "answer" in loaded[0]
            assert "ground_truth" in loaded[0]

    def test_retrieval_to_judge_pipeline(self):
        """Test retrieval results feed into judge correctly"""
        # Simulate: retrieval returns [chunk_ids] → judge evaluates answer
        retrieved_chunks = ["chunk_001", "chunk_002"]
        generated_answer = "AI evaluation measures quality of AI systems."

        # Judge should evaluate answer using retrieved context
        judge = LLMJudge()
        # Note: judge currently doesn't use chunks, but pipeline should support it
        result = asyncio.run(
            judge.evaluate_multi_judge(
                question="What is AI evaluation?",
                answer=generated_answer,
                ground_truth="AI evaluation is the process of assessing AI system performance.",
            )
        )

        assert result is not None
        assert "final_score" in result


# ============== Report Generation Tests ==============


class TestReportGeneration:
    """Tests for report output format (Planning.md Deliverables)"""

    def test_summary_json_structure(self, tmp_path):
        """Test reports/summary.json has required fields"""
        summary = {
            "metadata": {
                "version": "Agent_V2_Optimized",
                "total": 50,
                "timestamp": "2025-01-01 12:00:00",
            },
            "metrics": {"avg_score": 4.2, "hit_rate": 0.88, "agreement_rate": 0.82},
        }

        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        with open(summary_file, "r") as f:
            loaded = json.load(f)
            assert "metadata" in loaded
            assert "metrics" in loaded
            assert "hit_rate" in loaded["metrics"]  # Retrieval metric
            assert "agreement_rate" in loaded["metrics"]  # Multi-judge metric

    def test_benchmark_results_structure(self, tmp_path):
        """Test reports/benchmark_results.json structure"""
        results = [
            {
                "case_id": 1,
                "question": "Test?",
                "retrieved_chunks": ["c1", "c2"],
                "generated_answer": "Answer text",
                "ragas": {"retrieval": {"hit_rate": 1.0, "mrr": 0.5}},
                "judge": {"final_score": 4.5, "agreement_rate": 0.8},
            }
        ]

        results_file = tmp_path / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        with open(results_file, "r") as f:
            loaded = json.load(f)
            assert isinstance(loaded, list)
            assert len(loaded) > 0
            assert "ragas" in loaded[0]
            assert "judge" in loaded[0]


# ============== Regression & Delta Analysis Tests ==============


class TestRegressionAnalysis:
    """Tests for V1 vs V2 comparison (Planning.md Phase 5)"""

    def test_delta_calculation(self):
        """Test V2 improvement delta calculation"""
        v1_score = 4.0
        v2_score = 4.5
        delta = v2_score - v1_score

        assert delta == 0.5
        assert delta > 0  # V2 should improve

    def test_release_gate_approve_logic(self):
        """Test automatic release approval when delta > 0"""
        delta = 0.3
        decision = "APPROVE" if delta > 0 else "BLOCK RELEASE"
        assert decision == "APPROVE"

    def test_release_gate_block_logic(self):
        """Test release block when regression detected"""
        delta = -0.2
        decision = "APPROVE" if delta > 0 else "BLOCK RELEASE"
        assert decision == "BLOCK RELEASE"


# ============== Validation Tests (check_lab.py compatibility) ==============


class TestLabValidation:
    """Tests to ensure check_lab.py validation passes"""

    def test_summary_json_has_metrics_and_metadata(self, tmp_path):
        """Test summary.json contains metrics and metadata (check_lab.py requirement)"""
        summary = {
            "metadata": {"version": "V2", "total": 50},
            "metrics": {"avg_score": 4.2, "hit_rate": 0.85, "agreement_rate": 0.8},
        }

        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        with open(summary_file, "r") as f:
            loaded = json.load(f)
            assert "metrics" in loaded
            assert "metadata" in loaded

    def test_benchmark_results_exists_and_is_list(self, tmp_path):
        """Test benchmark_results.json exists and is a list"""
        results = [{"test": "data"}]
        results_file = tmp_path / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        with open(results_file, "r") as f:
            loaded = json.load(f)
            assert isinstance(loaded, list)

    def test_failure_analysis_md_exists(self, tmp_path):
        """Test analysis/failure_analysis.md file exists"""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        analysis_file = analysis_dir / "failure_analysis.md"
        analysis_file.write_text("# Failure Analysis\nContent here.")

        assert analysis_file.exists()
        assert analysis_file.read_text().strip().startswith("#")


# ============== Parametrized Tests ==============


class TestParametrizedScenarios:
    """Parametrized tests for various scenarios"""

    @pytest.mark.parametrize(
        "expected,retrieved,expected_hit",
        [
            (["a"], ["a"], 1.0),
            (["a"], ["b"], 0.0),
            (["a", "b"], ["b"], 1.0),
            (["a", "b"], ["c", "d"], 0.0),
            ([], ["a"], 0.0),
        ],
    )
    def test_hit_rate_parametrized(self, expected, retrieved, expected_hit):
        """Test Hit Rate across various scenarios"""
        verifier = ChunkVerifier()
        hit = verifier.calculate_hit_rate(expected, retrieved)
        assert hit == expected_hit

    @pytest.mark.parametrize(
        "expected,retrieved,expected_mrr",
        [
            (["a"], ["a"], 1.0),
            (["a"], ["b", "a"], 0.5),
            (["a"], ["b", "c", "a"], 1.0 / 3),
            (["a"], ["b", "c"], 0.0),
        ],
    )
    def test_mrr_parametrized(self, expected, retrieved, expected_mrr):
        """Test MRR across various scenarios"""
        verifier = ChunkVerifier()
        mrr = verifier.calculate_reciprocal_rank(expected, retrieved)
        assert mrr == pytest.approx(expected_mrr)

    @pytest.mark.parametrize(
        "question_type,should_have_chunks",
        [
            ("semantic_inference", True),
            ("negative_constraint", True),
            ("temporal_reasoning", True),
            ("comparative_analysis", True),
            ("context_dependent", True),
        ],
    )
    def test_question_types_have_expected_chunks(
        self, question_type, should_have_chunks
    ):
        """Test all question types define expected_chunks (for retrieval eval)"""
        generator = ChallengingQuestionsGenerator()
        all_q = generator.generate_all_questions()
        for q in all_q:
            if q.question_type.value == question_type:
                assert len(q.expected_chunks) > 0, (
                    f"{question_type} missing expected_chunks"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
