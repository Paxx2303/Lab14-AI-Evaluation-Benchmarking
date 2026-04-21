"""
Comprehensive Test Script for RAG V1 and V2 Systems
Tests both systems with sample data and provides comparison metrics
"""

import json
import time
import os
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent / "rag_v1"))
sys.path.append(str(Path(__file__).parent / "rag_v2"))
sys.path.append(str(Path(__file__).parent / "phase1"))

from rag_v1.simple_rag import SimpleRAG
from rag_v2.enhanced_rag import EnhancedRAG
from phase1.chunk_verifier import ChunkVerifier

class RAGSystemTester:
    """Test and compare RAG systems"""
    
    def __init__(self):
        self.v1_rag = SimpleRAG()
        self.v2_rag = EnhancedRAG()
        self.verifier = ChunkVerifier()
        
        # Test questions covering different complexity levels
        self.test_questions = [
            {
                "id": "test_001",
                "question": "What is Hit Rate in RAG evaluation?",
                "expected_concepts": ["hit rate", "retrieval", "evaluation"],
                "difficulty": "easy",
                "category": "evaluation"
            },
            {
                "id": "test_002", 
                "question": "How does Mean Reciprocal Rank (MRR) work?",
                "expected_concepts": ["MRR", "reciprocal rank", "retrieval quality"],
                "difficulty": "medium",
                "category": "evaluation"
            },
            {
                "id": "test_003",
                "question": "Compare semantic search versus keyword search approaches",
                "expected_concepts": ["semantic search", "keyword search", "BM25", "embeddings"],
                "difficulty": "medium",
                "category": "retrieval"
            },
            {
                "id": "test_004",
                "question": "What are the advantages of dynamic chunking over fixed-size methods?",
                "expected_concepts": ["dynamic chunking", "fixed-size", "semantic boundaries", "content density"],
                "difficulty": "hard",
                "category": "chunking"
            },
            {
                "id": "test_005",
                "question": "How does neural re-ranking improve RAG system performance?",
                "expected_concepts": ["neural re-ranking", "cross-encoder", "precision", "retrieval quality"],
                "difficulty": "hard",
                "category": "reranking"
            }
        ]
    
    def setup_test_data(self):
        """Set up sample data for testing"""
        print("Setting up test data...")
        
        # Create enhanced sample documents for V2
        from rag_v2.enhanced_rag import create_enhanced_sample_documents
        enhanced_docs = create_enhanced_sample_documents()
        self.v2_rag.process_documents(enhanced_docs)
        
        # Create simple sample documents for V1
        from rag_v1.simple_rag import create_sample_chunks
        simple_chunks = create_sample_chunks()
        self.v1_rag.add_document_chunks(simple_chunks)
        
        print("Test data setup complete")
    
    def test_v1_system(self) -> List[Dict[str, Any]]:
        """Test V1 RAG system"""
        print("\n=== Testing V1 Simple RAG System ===")
        results = []
        
        for test_case in self.test_questions:
            print(f"\nTesting: {test_case['question']}")
            start_time = time.time()
            
            try:
                result = self.v1_rag.query(test_case['question'])
                response_time = time.time() - start_time
                
                test_result = {
                    "system": "V1",
                    "test_id": test_case['id'],
                    "question": test_case['question'],
                    "answer": result.answer,
                    "response_time": response_time,
                    "retrieved_chunks_count": len(result.retrieved_chunks),
                    "retrieval_scores": result.retrieval_scores,
                    "expected_concepts": test_case['expected_concepts'],
                    "difficulty": test_case['difficulty'],
                    "category": test_case['category'],
                    "status": "success"
                }
                
                print(f"  Answer: {result.answer[:100]}...")
                print(f"  Response Time: {response_time:.2f}s")
                print(f"  Chunks Retrieved: {len(result.retrieved_chunks)}")
                
            except Exception as e:
                test_result = {
                    "system": "V1",
                    "test_id": test_case['id'],
                    "question": test_case['question'],
                    "error": str(e),
                    "status": "error"
                }
                print(f"  Error: {e}")
            
            results.append(test_result)
        
        return results
    
    def test_v2_system(self) -> List[Dict[str, Any]]:
        """Test V2 Enhanced RAG system"""
        print("\n=== Testing V2 Enhanced RAG System ===")
        results = []
        
        for test_case in self.test_questions:
            print(f"\nTesting: {test_case['question']}")
            start_time = time.time()
            
            try:
                result = self.v2_rag.query(test_case['question'])
                response_time = time.time() - start_time
                
                test_result = {
                    "system": "V2",
                    "test_id": test_case['id'],
                    "question": test_case['question'],
                    "answer": result.answer,
                    "response_time": response_time,
                    "retrieved_chunks_count": len(result.retrieved_chunks),
                    "retrieval_method": result.retrieval_method,
                    "query_expansion": result.query_expansion,
                    "pipeline_stages": result.pipeline_stages,
                    "expected_concepts": test_case['expected_concepts'],
                    "difficulty": test_case['difficulty'],
                    "category": test_case['category'],
                    "status": "success"
                }
                
                print(f"  Answer: {result.answer[:100]}...")
                print(f"  Response Time: {response_time:.2f}s")
                print(f"  Retrieval Method: {result.retrieval_method}")
                print(f"  Query Expansion: {len(result.query_expansion)} alternatives")
                print(f"  Pipeline Stages: {list(result.pipeline_stages.keys())}")
                
            except Exception as e:
                test_result = {
                    "system": "V2",
                    "test_id": test_case['id'],
                    "question": test_case['question'],
                    "error": str(e),
                    "status": "error"
                }
                print(f"  Error: {e}")
            
            results.append(test_result)
        
        return results
    
    def run_chunk_verification(self) -> Dict[str, Any]:
        """Run Phase 1 chunk verification"""
        print("\n=== Running Phase 1 Chunk Verification ===")
        
        try:
            # Generate challenging questions
            from phase1.challenging_questions import ChallengingQuestionsGenerator
            generator = ChallengingQuestionsGenerator()
            generator.save_questions("data/challenging_questions.json")
            
            # Run verification
            results = self.verifier.run_verification("data/challenging_questions.json")
            
            print(f"Verification completed:")
            print(f"  Pass Rate: {results['summary']['pass_rate']:.2%}")
            print(f"  Overall Hit Rate: {results['summary']['overall_hit_rate']:.3f}")
            print(f"  Overall MRR: {results['summary']['overall_mrr']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"Chunk verification error: {e}")
            return {"error": str(e)}
    
    def compare_systems(self, v1_results: List[Dict], v2_results: List[Dict]) -> Dict[str, Any]:
        """Compare V1 and V2 system performance"""
        print("\n=== System Comparison ===")
        
        # Calculate metrics for each system
        def calculate_metrics(results):
            successful_results = [r for r in results if r.get('status') == 'success']
            
            if not successful_results:
                return {"error": "No successful results"}
            
            avg_response_time = sum(r['response_time'] for r in successful_results) / len(successful_results)
            avg_chunks_retrieved = sum(r.get('retrieved_chunks_count', 0) for r in successful_results) / len(successful_results)
            success_rate = len(successful_results) / len(results)
            
            return {
                "avg_response_time": avg_response_time,
                "avg_chunks_retrieved": avg_chunks_retrieved,
                "success_rate": success_rate,
                "total_tests": len(results),
                "successful_tests": len(successful_results)
            }
        
        v1_metrics = calculate_metrics(v1_results)
        v2_metrics = calculate_metrics(v2_results)
        
        comparison = {
            "v1_metrics": v1_metrics,
            "v2_metrics": v2_metrics,
            "improvements": {}
        }
        
        # Calculate improvements
        if "error" not in v1_metrics and "error" not in v2_metrics:
            comparison["improvements"] = {
                "response_time_improvement": ((v1_metrics["avg_response_time"] - v2_metrics["avg_response_time"]) / v1_metrics["avg_response_time"]) * 100,
                "success_rate_improvement": v2_metrics["success_rate"] - v1_metrics["success_rate"],
                "chunks_retrieved_difference": v2_metrics["avg_chunks_retrieved"] - v1_metrics["avg_chunks_retrieved"]
            }
            
            print(f"V1 Average Response Time: {v1_metrics['avg_response_time']:.2f}s")
            print(f"V2 Average Response Time: {v2_metrics['avg_response_time']:.2f}s")
            print(f"Response Time Improvement: {comparison['improvements']['response_time_improvement']:.1f}%")
            print(f"V1 Success Rate: {v1_metrics['success_rate']:.2%}")
            print(f"V2 Success Rate: {v2_metrics['success_rate']:.2%}")
        
        return comparison
    
    def save_comprehensive_report(self, v1_results: List[Dict], v2_results: List[Dict], 
                                 comparison: Dict, verification_results: Dict):
        """Save comprehensive test report"""
        report = {
            "test_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_questions),
                "systems_tested": ["V1_Simple_RAG", "V2_Enhanced_RAG"]
            },
            "v1_results": v1_results,
            "v2_results": v2_results,
            "system_comparison": comparison,
            "phase1_verification": verification_results,
            "test_questions": self.test_questions
        }
        
        # Save detailed report
        with open("test_results/comprehensive_test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary = {
            "timestamp": report["test_summary"]["timestamp"],
            "v1_performance": comparison.get("v1_metrics", {}),
            "v2_performance": comparison.get("v2_metrics", {}),
            "improvements": comparison.get("improvements", {}),
            "verification_summary": verification_results.get("summary", {})
        }
        
        with open("test_results/test_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nReports saved:")
        print(f"  Detailed: test_results/comprehensive_test_report.json")
        print(f"  Summary: test_results/test_summary.json")
    
    def run_complete_test_suite(self):
        """Run complete test suite"""
        print("=== Complete RAG Systems Test Suite ===")
        
        # Create test results directory
        os.makedirs("test_results", exist_ok=True)
        
        # Setup test data
        self.setup_test_data()
        
        # Test both systems
        v1_results = self.test_v1_system()
        v2_results = self.test_v2_system()
        
        # Run chunk verification
        verification_results = self.run_chunk_verification()
        
        # Compare systems
        comparison = self.compare_systems(v1_results, v2_results)
        
        # Save comprehensive report
        self.save_comprehensive_report(v1_results, v2_results, comparison, verification_results)
        
        print("\n=== Test Suite Complete ===")
        return {
            "v1_results": v1_results,
            "v2_results": v2_results,
            "comparison": comparison,
            "verification": verification_results
        }

def main():
    """Run the complete test suite"""
    tester = RAGSystemTester()
    results = tester.run_complete_test_suite()
    
    print("\n=== Final Summary ===")
    if "comparison" in results and "improvements" in results["comparison"]:
        improvements = results["comparison"]["improvements"]
        print(f"V2 vs V1 Improvements:")
        print(f"  Response Time: {improvements.get('response_time_improvement', 0):.1f}%")
        print(f"  Success Rate: {improvements.get('success_rate_improvement', 0):.1%}")
    
    if "verification" in results and "summary" in results["verification"]:
        verification = results["verification"]["summary"]
        print(f"\nPhase 1 Verification:")
        print(f"  Pass Rate: {verification.get('pass_rate', 0):.2%}")
        print(f"  Hit Rate: {verification.get('overall_hit_rate', 0):.3f}")

if __name__ == "__main__":
    main()
