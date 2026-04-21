"""
RAG V2: Enhanced RAG System with Hybrid Search and Advanced Features
Implements the V2 Retrieval System design from Planning.md
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

@dataclass
class EnhancedRAGResult:
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    retrieval_method: str
    query_expansion: List[str]
    reranked_chunks: List[Dict[str, Any]]
    response_time: float
    pipeline_stages: Dict[str, float]

class QueryEnhancer:
    """Enhance queries with expansion and intent classification"""
    
    def __init__(self, openai_model: str = "gpt-3.5-turbo"):
        self.openai_model = openai_model
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def expand_query(self, query: str) -> List[str]:
        """Generate alternative queries using LLM"""
        prompt = f"""Generate 3 alternative ways to ask this question that might help find better information:
Original question: {query}

Alternative questions (one per line):"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            alternatives = response.choices[0].message.content.strip().split('\n')
            return [alt.strip() for alt in alternatives if alt.strip()]
        except:
            return [query]  # Fallback to original query
    
    def classify_intent(self, query: str) -> str:
        """Classify query intent for optimal retrieval strategy"""
        prompt = f"""Classify this query into one of these categories:
- factual: Looking for specific facts or definitions
- comparison: Comparing multiple concepts
- procedural: How to do something
- analytical: Analysis or explanation needed

Query: {query}

Category:"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip().lower()
        except:
            return "factual"  # Default fallback

class DynamicChunker:
    """Dynamic chunking with adaptive sizing and semantic boundaries"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.min_chunk_size = 100
        self.max_chunk_size = 500
        self.target_overlap = 50
    
    def calculate_content_density(self, text: str) -> float:
        """Calculate content density for adaptive chunk sizing"""
        # Simple heuristic: more unique words = higher density
        words = text.split()
        unique_words = set(words)
        density = len(unique_words) / len(words) if words else 0
        return density
    
    def find_semantic_boundaries(self, text: str) -> List[int]:
        """Find natural topic transition points"""
        sentences = re.split(r'[.!?]+', text)
        boundaries = [0]
        
        # Simple boundary detection based on sentence similarity
        if len(sentences) > 1:
            embeddings = self.embedding_model.encode(sentences)
            
            for i in range(1, len(sentences)):
                # Calculate similarity with previous sentence
                similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
                
                # Low similarity indicates topic change
                if similarity < 0.3:
                    boundaries.append(len('. '.join(sentences[:i])) + 2)
        
        return boundaries
    
    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Dynamic chunking with adaptive sizing"""
        density = self.calculate_content_density(text)
        
        # Adaptive chunk size based on density
        if density > 0.8:  # High density - smaller chunks
            chunk_size = self.min_chunk_size
        elif density < 0.4:  # Low density - larger chunks
            chunk_size = self.max_chunk_size
        else:
            chunk_size = 300  # Medium size
        
        # Find semantic boundaries
        boundaries = self.find_semantic_boundaries(text)
        
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            if end > start:
                chunk_text = text[start:end].strip()
                if len(chunk_text) > 50:  # Minimum chunk length
                    chunks.append({
                        "content": chunk_text,
                        "chunk_id": f"chunk_{i}",
                        "source": metadata.get("source", "unknown"),
                        "metadata": {
                            **(metadata or {}),
                            "chunk_size": len(chunk_text),
                            "density": density,
                            "position": i
                        }
                    })
        
        return chunks

class HybridRetriever:
    """Hybrid search combining semantic, keyword, and graph-based retrieval"""
    
    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.bm25_index = None
        self.documents = []
    
    def build_bm25_index(self, documents: List[str]):
        """Build BM25 index for keyword search"""
        self.documents = documents
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Semantic search using embeddings"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        search_results = self.qdrant_client.search(
            collection_name="document_chunks",
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        results = []
        for result in search_results:
            doc_id = result.payload.get("doc_index", result.id)
            results.append((doc_id, result.score))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Keyword search using BM25"""
        if not self.bm25_index:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Combine semantic and keyword search results"""
        semantic_results = self.semantic_search(query, top_k)
        keyword_results = self.keyword_search(query, top_k)
        
        # Combine and re-score
        combined_scores = {}
        
        # Add semantic scores (weighted 0.6)
        for doc_id, score in semantic_results:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score * 0.6
        
        # Add keyword scores (weighted 0.4)
        for doc_id, score in keyword_results:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score * 0.4
        
        # Sort by combined score
        ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_results[:top_k]

class NeuralReranker:
    """Neural re-ranking using cross-encoder"""
    
    def __init__(self, openai_model: str = "gpt-3.5-turbo"):
        self.openai_model = openai_model
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Re-rank documents using neural scoring"""
        if len(documents) <= top_k:
            return list(enumerate([1.0] * len(documents)))
        
        scores = []
        
        for i, doc in enumerate(documents):
            prompt = f"""Rate the relevance of this document to the query on a scale of 0-1:
            
Query: {query}

Document: {doc[:500]}...

Relevance score (0-1):"""
            
            try:
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                
                score = float(response.choices[0].message.content.strip())
                scores.append((i, score))
            except:
                scores.append((i, 0.5))  # Default score
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class EnhancedRAG:
    """
    V2 RAG System - Enhanced with hybrid search, dynamic chunking, and query enhancement
    Implements the multi-stage pipeline from Planning.md
    """
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "document_chunks_v2",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 openai_model: str = "gpt-3.5-turbo"):
        
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name
        self.openai_model = openai_model
        
        # Initialize components
        self.query_enhancer = QueryEnhancer(openai_model)
        self.chunker = DynamicChunker(self.embedding_model)
        self.hybrid_retriever = HybridRetriever(self.qdrant_client, self.embedding_model)
        self.reranker = NeuralReranker(openai_model)
        
        # Enhanced prompt template
        self.prompt_template = """You are a knowledgeable assistant. Based on the following context, provide a comprehensive and accurate answer to the question.

Context Information:
{context}

Question: {question}

Instructions:
- Use only the information provided in the context
- If the context doesn't contain enough information, say so clearly
- Provide a detailed and well-structured answer
- Include specific details and examples from the context when relevant

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
    
    def process_documents(self, documents: List[Dict[str, Any]]):
        """Process documents with dynamic chunking"""
        self.ensure_collection_exists()
        
        all_chunks = []
        doc_texts = []
        
        for doc_idx, doc in enumerate(documents):
            # Dynamic chunking
            chunks = self.chunker.chunk_document(doc["content"], doc.get("metadata", {}))
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk["doc_index"] = len(all_chunks)
                chunk["global_doc_id"] = doc_idx
                all_chunks.append(chunk)
                doc_texts.append(chunk["content"])
        
        # Build BM25 index
        self.hybrid_retriever.build_bm25_index(doc_texts)
        
        # Add chunks to Qdrant
        points = []
        for chunk in all_chunks:
            embedding = self.embedding_model.encode(chunk["content"]).tolist()
            
            point = PointStruct(
                id=chunk["doc_index"],
                vector=embedding,
                payload=chunk
            )
            points.append(point)
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def multi_stage_retrieval(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Multi-stage retrieval pipeline:
        Stage 1: Broad semantic search (top-k=50)
        Stage 2: Keyword refinement (filter to top-k=20)
        Stage 3: Neural re-ranking (final top-k=5)
        """
        pipeline_times = {}
        
        # Stage 1: Query enhancement
        start_time = time.time()
        expanded_queries = self.query_enhancer.expand_query(query)
        intent = self.query_enhancer.classify_intent(query)
        pipeline_times["query_enhancement"] = time.time() - start_time
        
        # Stage 2: Broad hybrid search
        start_time = time.time()
        all_results = []
        for expanded_query in [query] + expanded_queries:
            results = self.hybrid_retriever.hybrid_search(expanded_query, top_k=50)
            all_results.extend(results)
        
        # Remove duplicates and sort
        unique_results = {}
        for doc_id, score in all_results:
            if doc_id not in unique_results:
                unique_results[doc_id] = score
            else:
                unique_results[doc_id] = max(unique_results[doc_id], score)
        
        broad_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)[:20]
        pipeline_times["broad_search"] = time.time() - start_time
        
        # Stage 3: Get documents for re-ranking
        start_time = time.time()
        doc_ids = [doc_id for doc_id, _ in broad_results]
        
        # Retrieve full documents
        retrieved_docs = []
        for doc_id in doc_ids:
            search_result = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True
            )
            if search_result:
                retrieved_docs.append(search_result[0].payload["content"])
        
        # Stage 4: Neural re-ranking
        reranked_indices = self.reranker.rerank(query, retrieved_docs, top_k)
        
        final_chunks = []
        final_scores = []
        
        for rank_idx, score in reranked_indices:
            original_doc_id = doc_ids[rank_idx]
            search_result = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[original_doc_id],
                with_payload=True
            )
            if search_result:
                chunk = search_result[0].payload
                final_chunks.append(chunk)
                final_scores.append(score)
        
        pipeline_times["reranking"] = time.time() - start_time
        
        return final_chunks, {
            "query_enhancement": pipeline_times["query_enhancement"],
            "broad_search": pipeline_times["broad_search"],
            "reranking": pipeline_times["reranking"],
            "intent": intent,
            "expanded_queries": expanded_queries
        }
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate enhanced answer using retrieved context"""
        # Create rich context with source information
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Unknown")
            content = chunk["content"]
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Create enhanced prompt
        prompt = self.prompt_template.format(context=context, question=query)
        
        # Generate response
        response = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant that provides accurate, well-structured answers based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, question: str, top_k: int = 5) -> EnhancedRAGResult:
        """End-to-end enhanced RAG query"""
        start_time = time.time()
        
        # Multi-stage retrieval
        retrieved_chunks, pipeline_info = self.multi_stage_retrieval(question, top_k)
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_chunks)
        
        response_time = time.time() - start_time
        
        return EnhancedRAGResult(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            retrieval_scores=[1.0] * len(retrieved_chunks),  # Will be populated by reranker
            retrieval_method="hybrid_multistage",
            query_expansion=pipeline_info.get("expanded_queries", []),
            reranked_chunks=retrieved_chunks,
            response_time=response_time,
            pipeline_stages=pipeline_info
        )

def create_enhanced_sample_documents():
    """Create sample documents with varied content for testing V2"""
    documents = [
        {
            "content": """RAG evaluation methodologies have evolved significantly since 2020. Early approaches focused primarily on simple relevance metrics like precision and recall. However, these methods failed to capture important aspects such as factual consistency and hallucination. The introduction of RAGAS in 2022 marked a turning point, providing comprehensive metrics including faithfulness, answer relevance, and context recall. More recently, ARES (2023) has emerged as a sophisticated framework that combines multiple evaluation dimensions with automated judge systems.""",
            "source": "evaluation_evolution.pdf",
            "metadata": {"category": "evaluation", "year": "2023", "difficulty": "medium"}
        },
        {
            "content": """Hybrid retrieval systems combine the strengths of multiple search approaches. Semantic search using embeddings excels at understanding conceptual similarity and handling synonyms, making it ideal for queries that require understanding meaning rather than exact keywords. However, semantic search can miss exact matches and specific terminology. BM25 keyword search provides precise matching for technical terms and exact phrases but fails to understand semantic relationships. The optimal approach combines both methods, using semantic search for broad retrieval and keyword search for precision refinement.""",
            "source": "hybrid_search_guide.pdf",
            "metadata": {"category": "retrieval", "year": "2023", "difficulty": "medium"}
        },
        {
            "content": """Dynamic chunking represents a significant improvement over fixed-size chunking approaches. Instead of using uniform chunk sizes, dynamic chunking adapts to content density and semantic boundaries. High-density content with many unique terms benefits from smaller chunks to maintain focus, while low-density content can use larger chunks to preserve context. Semantic boundary detection ensures that chunks rarely cut across topic transitions, maintaining coherence and improving retrieval relevance. This adaptive approach typically improves retrieval performance by 15-25% compared to fixed-size methods.""",
            "source": "advanced_chunking.pdf",
            "metadata": {"category": "chunking", "year": "2023", "difficulty": "hard"}
        },
        {
            "content": """Neural re-ranking has become essential for modern RAG systems. After initial retrieval using broad search strategies, neural re-ranking applies cross-encoder models to re-evaluate the relevance of retrieved documents. This second-stage filtering significantly improves precision by considering the query-document relationship more deeply. Cross-encoders can capture complex interactions between query terms and document content that simple vector similarity misses. The computational cost is higher but justified by the substantial improvement in retrieval quality, especially for complex queries requiring understanding of nuanced relationships.""",
            "source": "neural_reranking.pdf",
            "metadata": {"category": "reranking", "year": "2023", "difficulty": "hard"}
        },
        {
            "content": """Query enhancement techniques significantly improve RAG system performance. Query expansion generates alternative formulations of the original query, increasing the likelihood of finding relevant documents. Intent classification categorizes queries into types like factual, comparative, or procedural, allowing the system to apply optimal retrieval strategies. Context injection maintains conversation history and user preferences, enabling more personalized and contextually appropriate responses. These enhancements are particularly valuable for complex queries where the initial formulation may not optimally match the relevant information in the knowledge base.""",
            "source": "query_enhancement.pdf",
            "metadata": {"category": "query_processing", "year": "2023", "difficulty": "medium"}
        }
    ]
    return documents

if __name__ == "__main__":
    # Initialize enhanced RAG system
    enhanced_rag = EnhancedRAG()
    
    # Process enhanced sample documents
    sample_docs = create_enhanced_sample_documents()
    enhanced_rag.process_documents(sample_docs)
    
    # Test queries that showcase V2 capabilities
    test_questions = [
        "How have RAG evaluation methods evolved since 2020?",
        "Compare the advantages and disadvantages of semantic search versus keyword search",
        "What makes dynamic chunking better than fixed-size approaches?",
        "Why is neural re-ranking important for RAG systems?",
        "How does query enhancement improve retrieval performance?"
    ]
    
    print("=== V2 Enhanced RAG System Test ===")
    results = []
    
    for question in test_questions:
        print(f"\n--- Processing: {question} ---")
        result = enhanced_rag.query(question)
        results.append(result)
        
        print(f"Answer: {result.answer[:200]}...")
        print(f"Response Time: {result.response_time:.2f}s")
        print(f"Pipeline Stages: {result.pipeline_stages}")
        print(f"Query Expansion: {result.query_expansion}")
    
    # Save results
    results_data = []
    for result in results:
        results_data.append({
            "query": result.query,
            "answer": result.answer,
            "retrieved_chunks": result.retrieved_chunks,
            "retrieval_method": result.retrieval_method,
            "query_expansion": result.query_expansion,
            "response_time": result.response_time,
            "pipeline_stages": result.pipeline_stages
        })
    
    with open("rag_v2/test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(results_data)} enhanced results to rag_v2/test_results.json")
