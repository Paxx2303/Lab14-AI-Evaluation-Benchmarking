"""
RAG V1: Simple RAG System with Qdrant
Baseline implementation as described in Planning.md
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    response_time: float

class SimpleRAG:
    """
    V1 RAG System - Simple semantic search with embeddings
    Baseline implementation per Planning.md
    """
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "document_chunks",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 openai_model: str = "gpt-3.5-turbo"):
        
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name
        self.openai_model = openai_model
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Simple prompt template
        self.prompt_template = """Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    def ensure_collection_exists(self, vector_size: int = 384):
        """Create collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text using sentence transformer"""
        return self.embedding_model.encode(text).tolist()
    
    def add_document_chunks(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to Qdrant collection"""
        self.ensure_collection_exists()
        
        points = []
        for i, chunk in enumerate(chunks):
            # Embed chunk content
            embedding = self.embed_text(chunk["content"])
            
            # Create point
            point = PointStruct(
                id=chunk.get("id", i),
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "source": chunk.get("source", "unknown"),
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "metadata": chunk.get("metadata", {})
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Added {len(points)} chunks to {self.collection_name}")
    
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple semantic search - V1 baseline approach
        """
        # Embed query
        query_embedding = self.embed_text(query)
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        chunks = []
        scores = []
        for result in search_results:
            chunk = {
                "content": result.payload["content"],
                "source": result.payload["source"],
                "chunk_id": result.payload["chunk_id"],
                "metadata": result.payload["metadata"]
            }
            chunks.append(chunk)
            scores.append(result.score)
        
        return chunks, scores
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using OpenAI"""
        # Combine retrieved chunks as context
        context = "\n\n".join([chunk["content"] for chunk in retrieved_chunks])
        
        # Create prompt
        prompt = self.prompt_template.format(context=context, question=query)
        
        # Generate response
        response = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        End-to-end RAG query - V1 simple approach
        """
        import time
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks, retrieval_scores = self.retrieve_chunks(question, top_k)
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, retrieved_chunks)
        
        response_time = time.time() - start_time
        
        return RAGResult(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            retrieval_scores=retrieval_scores,
            response_time=response_time
        )
    
    def batch_query(self, questions: List[str], top_k: int = 5) -> List[RAGResult]:
        """Process multiple queries"""
        results = []
        for question in questions:
            result = self.query(question, top_k)
            results.append(result)
            print(f"Processed: {question[:50]}...")
        return results
    
    def save_results(self, results: List[RAGResult], filepath: str):
        """Save results to JSON file"""
        results_data = []
        for result in results:
            results_data.append({
                "query": result.query,
                "answer": result.answer,
                "retrieved_chunks": result.retrieved_chunks,
                "retrieval_scores": result.retrieval_scores,
                "response_time": result.response_time
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results_data)} results to {filepath}")

def create_sample_chunks():
    """Create sample document chunks for testing"""
    chunks = [
        {
            "id": "chunk_001",
            "content": "RAG evaluation requires multiple metrics to assess system performance. Key metrics include Hit Rate, MRR (Mean Reciprocal Rank), and faithfulness scores. These metrics help measure both retrieval quality and generation quality.",
            "source": "evaluation_guide.pdf",
            "chunk_id": "rag_metrics_intro",
            "metadata": {"category": "evaluation", "difficulty": "medium"}
        },
        {
            "id": "chunk_002", 
            "content": "Hit Rate@k measures the percentage of queries that have at least one relevant document in the top-k retrieved results. This is important for understanding whether the system can find any relevant information.",
            "source": "evaluation_guide.pdf",
            "chunk_id": "hit_rate_definition",
            "metadata": {"category": "evaluation", "difficulty": "easy"}
        },
        {
            "id": "chunk_003",
            "content": "Mean Reciprocal Rank (MRR) calculates the average of the reciprocal ranks of the first relevant document. For example, if the first relevant document appears at position 3, the reciprocal rank is 1/3.",
            "source": "evaluation_guide.pdf", 
            "chunk_id": "mrr_definition",
            "metadata": {"category": "evaluation", "difficulty": "medium"}
        },
        {
            "id": "chunk_004",
            "content": "Faithfulness measures whether the generated answer is factually consistent with the retrieved context. High faithfulness means the answer doesn't hallucinate or contradict the source material.",
            "source": "generation_quality.pdf",
            "chunk_id": "faithfulness_metric",
            "metadata": {"category": "generation", "difficulty": "medium"}
        },
        {
            "id": "chunk_005",
            "content": "Judge consensus in RAG evaluation involves using multiple LLMs to score answers and then aggregating their opinions. This helps reduce bias and improve reliability of evaluations.",
            "source": "judge_system.pdf",
            "chunk_id": "judge_consensus",
            "metadata": {"category": "judging", "difficulty": "hard"}
        }
    ]
    return chunks

if __name__ == "__main__":
    # Initialize simple RAG system
    rag = SimpleRAG()
    
    # Create and add sample chunks
    sample_chunks = create_sample_chunks()
    rag.add_document_chunks(sample_chunks)
    
    # Test queries
    test_questions = [
        "What is Hit Rate in RAG evaluation?",
        "How is MRR calculated?",
        "What does faithfulness measure?",
        "Why use multiple judges in evaluation?"
    ]
    
    print("=== V1 Simple RAG System Test ===")
    results = rag.batch_query(test_questions)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n--- Question {i+1} ---")
        print(f"Query: {result.query}")
        print(f"Answer: {result.answer}")
        print(f"Response Time: {result.response_time:.2f}s")
        print(f"Retrieved {len(result.retrieved_chunks)} chunks")
    
    # Save results
    rag.save_results(results, "rag_v1/test_results.json")
