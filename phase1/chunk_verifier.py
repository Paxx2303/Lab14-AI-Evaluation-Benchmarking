"""
Phase 1: Chunk Verification System
Tests whether retrieval systems can find exact document chunks needed to answer questions
"""

import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class VerificationResult:
    question_id: str
    question_type: str
    expected_chunks: List[str]
    retrieved_chunks: List[str]
    hit_rate: float
    reciprocal_rank: float
    precision_at_k: float
    verification_passed: bool

class ChunkVerifier:
    """Verify retrieval system can find correct chunks for challenging questions"""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "document_chunks"
    
    def load_challenging_questions(self, filepath: str) -> List[Dict]:
        """Load challenging questions from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve relevant chunks for a query"""
        # Embed the query
        query_embedding = self.embedding_model.encode(query)
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        # Extract chunk IDs and scores
        retrieved = []
        for result in search_result:
            chunk_id = result.payload.get("chunk_id", f"chunk_{result.id}")
            score = result.score
            retrieved.append((chunk_id, score))
        
        return retrieved
    
    def calculate_hit_rate(self, expected: List[str], retrieved: List[str]) -> float:
        """Calculate Hit Rate@k: percentage of queries with at least one relevant chunk"""
        return float(any(chunk in expected for chunk in retrieved))
    
    def calculate_reciprocal_rank(self, expected: List[str], retrieved: List[str]) -> float:
        """Calculate Mean Reciprocal Rank: average of 1/rank of first relevant chunk"""
        for i, chunk in enumerate(retrieved):
            if chunk in expected:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_precision_at_k(self, expected: List[str], retrieved: List[str]) -> float:
        """Calculate Precision@k: relevance of retrieved chunks"""
        if not retrieved:
            return 0.0
        
        relevant_count = sum(1 for chunk in retrieved if chunk in expected)
        return relevant_count / len(retrieved)
    
    def verify_question(self, question: Dict, top_k: int = 5) -> VerificationResult:
        """Verify a single challenging question"""
        query = question["question"]
        expected_chunks = question["expected_chunks"]
        question_type = question["question_type"]
        
        # Retrieve chunks
        retrieved_with_scores = self.retrieve_chunks(query, top_k)
        retrieved_chunks = [chunk_id for chunk_id, _ in retrieved_with_scores]
        
        # Calculate metrics
        hit_rate = self.calculate_hit_rate(expected_chunks, retrieved_chunks)
        reciprocal_rank = self.calculate_reciprocal_rank(expected_chunks, retrieved_chunks)
        precision_at_k = self.calculate_precision_at_k(expected_chunks, retrieved_chunks)
        
        # Determine if verification passed (hit_rate > 0)
        verification_passed = hit_rate > 0
        
        return VerificationResult(
            question_id=question["id"],
            question_type=question_type,
            expected_chunks=expected_chunks,
            retrieved_chunks=retrieved_chunks,
            hit_rate=hit_rate,
            reciprocal_rank=reciprocal_rank,
            precision_at_k=precision_at_k,
            verification_passed=verification_passed
        )
    
    def run_verification(self, questions_filepath: str, top_k: int = 5) -> Dict[str, Any]:
        """Run chunk verification on all challenging questions"""
        questions = self.load_challenging_questions(questions_filepath)
        results = []
        
        print(f"Running chunk verification on {len(questions)} questions...")
        
        for question in questions:
            result = self.verify_question(question, top_k)
            results.append(result)
            
            status = "PASS" if result.verification_passed else "FAIL"
            print(f"{result.question_id}: {status} - Hit Rate: {result.hit_rate:.2f}")
        
        # Calculate aggregate metrics
        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.verification_passed)
        overall_hit_rate = sum(r.hit_rate for r in results) / total_questions
        overall_mrr = sum(r.reciprocal_rank for r in results) / total_questions
        overall_precision = sum(r.precision_at_k for r in results) / total_questions
        
        # Group by question type
        type_metrics = {}
        for question_type in set(r.question_type for r in results):
            type_results = [r for r in results if r.question_type == question_type]
            type_metrics[question_type] = {
                "count": len(type_results),
                "hit_rate": sum(r.hit_rate for r in type_results) / len(type_results),
                "mrr": sum(r.reciprocal_rank for r in type_results) / len(type_results),
                "precision": sum(r.precision_at_k for r in type_results) / len(type_results)
            }
        
        return {
            "summary": {
                "total_questions": total_questions,
                "passed_questions": passed_questions,
                "pass_rate": passed_questions / total_questions,
                "overall_hit_rate": overall_hit_rate,
                "overall_mrr": overall_mrr,
                "overall_precision": overall_precision
            },
            "by_question_type": type_metrics,
            "detailed_results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "expected_chunks": r.expected_chunks,
                    "retrieved_chunks": r.retrieved_chunks,
                    "hit_rate": r.hit_rate,
                    "reciprocal_rank": r.reciprocal_rank,
                    "precision_at_k": r.precision_at_k,
                    "verification_passed": r.verification_passed
                }
                for r in results
            ]
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save verification results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved verification results to {filepath}")

if __name__ == "__main__":
    verifier = ChunkVerifier()
    
    # Generate challenging questions first
    from challenging_questions import ChallengingQuestionsGenerator
    generator = ChallengingQuestionsGenerator()
    generator.save_questions("data/challenging_questions.json")
    
    # Run verification (assuming Qdrant is running and has data)
    try:
        results = verifier.run_verification("data/challenging_questions.json")
        verifier.save_results(results, "phase1/verification_results.json")
        
        print("\n=== Verification Summary ===")
        summary = results["summary"]
        print(f"Pass Rate: {summary['pass_rate']:.2%}")
        print(f"Overall Hit Rate: {summary['overall_hit_rate']:.3f}")
        print(f"Overall MRR: {summary['overall_mrr']:.3f}")
        print(f"Overall Precision: {summary['overall_precision']:.3f}")
        
    except Exception as e:
        print(f"Error running verification: {e}")
        print("Make sure Qdrant is running and contains document chunks.")
