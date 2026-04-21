"""
Phase 1: Challenging Questions for Chunk Verification
Based on Planning.md requirements for testing retrieval precision
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class QuestionType(Enum):
    SEMANTIC_INFERENCE = "semantic_inference"
    NEGATIVE_CONSTRAINT = "negative_constraint"
    TEMPORAL_REASONING = "temporal_reasoning"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    CONTEXT_DEPENDENT = "context_dependent"

@dataclass
class ChallengingQuestion:
    id: str
    question: str
    question_type: QuestionType
    expected_chunks: List[str]
    difficulty: str
    explanation: str
    ground_truth_answer: str
    distractor_chunks: List[str] = None

class ChallengingQuestionsGenerator:
    """Generate challenging questions for chunk verification as per Planning.md"""
    
    def __init__(self):
        self.questions = []
    
    def generate_semantic_inference_questions(self) -> List[ChallengingQuestion]:
        """Questions requiring understanding implicit relationships"""
        return [
            ChallengingQuestion(
                id="sem_001",
                question="What are the common failure patterns in systems that share similar architectural principles to microservices?",
                question_type=QuestionType.SEMANTIC_INFERENCE,
                expected_chunks=["chunk_architecture_microservices", "chunk_system_failures"],
                difficulty="hard",
                explanation="Requires connecting microservices architecture concepts with failure patterns",
                ground_truth_answer="Common failure patterns include network latency issues, service discovery failures, distributed transaction problems, and cascading failures due to tight coupling between services."
            ),
            ChallengingQuestion(
                id="sem_002", 
                question="Which evaluation methodologies implicitly assume that retrieved chunks are independent of each other?",
                question_type=QuestionType.SEMANTIC_INFERENCE,
                expected_chunks=["chunk_evaluation_metrics", "chunk_retrieval_assumptions"],
                difficulty="medium",
                explanation="Tests understanding of underlying assumptions in evaluation methods",
                ground_truth_answer="Traditional precision/recall metrics and simple relevance scoring assume chunk independence, while methods like RAGAS and faithfulness metrics account for inter-chunk dependencies."
            )
        ]
    
    def generate_negative_constraint_questions(self) -> List[ChallengingQuestion]:
        """Questions that must exclude specific information"""
        return [
            ChallengingQuestion(
                id="neg_001",
                question="Which evaluation metrics are NOT suitable for measuring retrieval quality in RAG systems, excluding precision and recall?",
                question_type=QuestionType.NEGATIVE_CONSTRAINT,
                expected_chunks=["chunk_evaluation_metrics", "chunk_retrieval_quality"],
                difficulty="medium",
                explanation="Must filter out precision/recall and identify unsuitable metrics",
                ground_truth_answer="Metrics like BLEU, ROUGE, and F1-score are not suitable for retrieval quality as they were designed for generation tasks, not retrieval assessment.",
                distractor_chunks=["chunk_precision_recall"]
            ),
            ChallengingQuestion(
                id="neg_002",
                question="What approaches should be avoided when implementing judge consensus for RAG evaluation, other than using a single judge?",
                question_type=QuestionType.NEGATIVE_CONSTRAINT, 
                expected_chunks=["chunk_judge_consensus", "chunk_evaluation_best_practices"],
                difficulty="hard",
                explanation="Tests knowledge of what NOT to do in judge consensus",
                ground_truth_answer="Avoid simple majority voting without confidence weighting, ignore inter-rater reliability metrics, and don't use judges with identical training data or architectures.",
                distractor_chunks=["chunk_single_judge_approach"]
            )
        ]
    
    def generate_temporal_reasoning_questions(self) -> List[ChallengingQuestion]:
        """Questions requiring understanding time-based relationships"""
        return [
            ChallengingQuestion(
                id="temp_001",
                question="What evaluation approaches were developed after 2020 that address limitations of earlier RAG assessment methods?",
                question_type=QuestionType.TEMPORAL_REASONING,
                expected_chunks=["chunk_modern_evaluation", "chunk_historical_methods"],
                difficulty="medium",
                explanation="Tests chronological understanding across evaluation evolution",
                ground_truth_answer="Post-2020 approaches include RAGAS (2022), ARES (2023), and Faithfulness metrics that specifically address hallucination and grounding issues missing from earlier methods."
            ),
            ChallengingQuestion(
                id="temp_002",
                question="Which retrieval system limitations were identified first: semantic drift or computational efficiency issues?",
                question_type=QuestionType.TEMPORAL_REASONING,
                expected_chunks=["chunk_retrieval_evolution", "chunk_historical_limitations"],
                difficulty="hard",
                explanation="Requires understanding historical development of retrieval systems",
                ground_truth_answer="Computational efficiency issues were identified first (2018-2019), while semantic drift was recognized as a critical problem later (2020-2021) as systems scaled."
            )
        ]
    
    def generate_comparative_analysis_questions(self) -> List[ChallengingQuestion]:
        """Questions requiring comparison of multiple concepts"""
        return [
            ChallengingQuestion(
                id="comp_001",
                question="Compare the trade-offs between using GPT-4 versus Claude-3 as judge models in terms of cost, accuracy, and bias.",
                question_type=QuestionType.COMPARATIVE_ANALYSIS,
                expected_chunks=["chunk_gpt4_judge", "chunk_claude3_judge", "chunk_judge_comparison"],
                difficulty="medium",
                explanation="Must retrieve and synthesize comparison information from multiple sources",
                ground_truth_answer="GPT-4 offers higher accuracy (~85%) but costs 3x more and shows tech industry bias. Claude-3 provides slightly lower accuracy (~80%) at 2x cost with less domain bias but more conservative scoring."
            ),
            ChallengingQuestion(
                id="comp_002",
                question="How do BM25 and semantic search complement each other in hybrid retrieval systems?",
                question_type=QuestionType.COMPARATIVE_ANALYSIS,
                expected_chunks=["chunk_bm25_search", "chunk_semantic_search", "chunk_hybrid_approaches"],
                difficulty="medium",
                explanation="Requires understanding complementary strengths of different search methods",
                ground_truth_answer="BM25 excels at exact keyword matching and is computationally efficient, while semantic search captures conceptual similarity and handles synonyms. Hybrid systems combine BM25's precision with semantic search's recall."
            )
        ]
    
    def generate_context_dependent_questions(self) -> List[ChallengingQuestion]:
        """Questions where meaning changes based on context"""
        return [
            ChallengingQuestion(
                id="ctx_001",
                question="In the context of enterprise deployment, what are the key considerations for scaling evaluation systems that differ from research settings?",
                question_type=QuestionType.CONTEXT_DEPENDENT,
                expected_chunks=["chunk_enterprise_deployment", "chunk_research_settings", "chunk_scaling_considerations"],
                difficulty="hard",
                explanation="Tests contextual understanding vs literal matching",
                ground_truth_answer="Enterprise settings require SLA compliance, cost optimization, data privacy, and integration with existing systems, while research settings prioritize experimental flexibility and comprehensive metrics regardless of cost."
            ),
            ChallengingQuestion(
                id="ctx_002",
                question="When evaluating RAG systems for customer support versus academic research, how should the weighting of faithfulness versus completeness change?",
                question_type=QuestionType.CONTEXT_DEPENDENT,
                expected_chunks=["chunk_customer_support_eval", "chunk_academic_eval", "chunk_metric_weighting"],
                difficulty="hard",
                explanation="Context determines metric priorities",
                ground_truth_answer="Customer support should weight faithfulness higher (70%) to prevent misinformation, while academic research can prioritize completeness (60%) to ensure comprehensive coverage of topics."
            )
        ]
    
    def generate_all_questions(self) -> List[ChallengingQuestion]:
        """Generate all types of challenging questions"""
        all_questions = []
        all_questions.extend(self.generate_semantic_inference_questions())
        all_questions.extend(self.generate_negative_constraint_questions())
        all_questions.extend(self.generate_temporal_reasoning_questions())
        all_questions.extend(self.generate_comparative_analysis_questions())
        all_questions.extend(self.generate_context_dependent_questions())
        return all_questions
    
    def save_questions(self, filepath: str):
        """Save questions to JSON file"""
        questions_data = []
        for q in self.generate_all_questions():
            questions_data.append({
                "id": q.id,
                "question": q.question,
                "question_type": q.question_type.value,
                "expected_chunks": q.expected_chunks,
                "difficulty": q.difficulty,
                "explanation": q.explanation,
                "ground_truth_answer": q.ground_truth_answer,
                "distractor_chunks": q.distractor_chunks or []
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(questions_data)} challenging questions to {filepath}")

if __name__ == "__main__":
    generator = ChallengingQuestionsGenerator()
    generator.save_questions("data/challenging_questions.json")
