"""
Data Chunking and Storage Pipeline for Qdrant
Processes documents from the data/docs folder and stores them in Qdrant
"""

import os
import json
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import re


@dataclass
class DocumentChunk:
    content: str
    source: str
    chunk_id: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    """Process different document types (PDF, TXT, MD)"""

    @staticmethod
    def read_pdf(filepath: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(filepath, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    @staticmethod
    def read_text(filepath: str) -> str:
        """Read text from TXT or MD file"""
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def process_document(filepath: str) -> Dict[str, Any]:
        """Process document based on file type"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")

        # Determine file type and extract content
        if filepath.suffix.lower() == ".pdf":
            content = DocumentProcessor.read_pdf(str(filepath))
        elif filepath.suffix.lower() in [".txt", ".md"]:
            content = DocumentProcessor.read_text(str(filepath))
        else:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")

        return {
            "content": content,
            "source": filepath.name,
            "filepath": str(filepath),
            "file_type": filepath.suffix.lower(),
            "size": len(content),
            "line_count": len(content.split("\n")),
        }


class AdvancedChunker:
    """Advanced chunking with multiple strategies"""

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model

    def semantic_chunking(
        self, text: str, chunk_size: int = 300, overlap: int = 50
    ) -> List[DocumentChunk]:
        """Chunk text based on semantic boundaries"""
        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        chunk_start = 0

        for i, sentence in enumerate(sentences):
            # If adding this sentence would exceed chunk size, create a new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Create chunk
                chunk_id = f"semantic_chunk_{len(chunks)}"
                chunk_end = chunk_start + len(current_chunk)

                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    source="semantic_chunking",
                    chunk_id=chunk_id,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata={
                        "strategy": "semantic",
                        "sentence_count": len(current_chunk.split(".")),
                        "char_count": len(current_chunk),
                    },
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                )
                current_chunk = overlap_text + " " + sentence
                chunk_start = chunk_end - len(overlap_text)
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk if there's content left
        if current_chunk.strip():
            chunk_id = f"semantic_chunk_{len(chunks)}"
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                source="semantic_chunking",
                chunk_id=chunk_id,
                start_char=chunk_start,
                end_char=len(text),
                metadata={
                    "strategy": "semantic",
                    "sentence_count": len(current_chunk.split(".")),
                    "char_count": len(current_chunk),
                },
            )
            chunks.append(chunk)

        return chunks

    def fixed_size_chunking(
        self, text: str, chunk_size: int = 500, overlap: int = 100
    ) -> List[DocumentChunk]:
        """Fixed-size chunking with overlap"""
        chunks = []

        for i in range(0, len(text), chunk_size - overlap):
            end = min(i + chunk_size, len(text))
            chunk_text = text[i:end]

            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                chunk_id = f"fixed_chunk_{len(chunks)}"
                chunk = DocumentChunk(
                    content=chunk_text.strip(),
                    source="fixed_size_chunking",
                    chunk_id=chunk_id,
                    start_char=i,
                    end_char=end,
                    metadata={
                        "strategy": "fixed_size",
                        "char_count": len(chunk_text),
                        "overlap": overlap,
                    },
                )
                chunks.append(chunk)

        return chunks

    def paragraph_chunking(self, text: str) -> List[DocumentChunk]:
        """Chunk by paragraphs"""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_pos = 0

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) > 20:  # Skip very short paragraphs
                chunk_id = f"paragraph_chunk_{i}"
                # Find the actual position in the original text
                start_pos = text.find(paragraph, current_pos)
                end_pos = start_pos + len(paragraph)

                chunk = DocumentChunk(
                    content=paragraph,
                    source="paragraph_chunking",
                    chunk_id=chunk_id,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata={
                        "strategy": "paragraph",
                        "paragraph_index": i,
                        "char_count": len(paragraph),
                    },
                )
                chunks.append(chunk)
                current_pos = end_pos

        return chunks

    def hybrid_chunking(self, text: str, doc_source: str) -> List[DocumentChunk]:
        """Hybrid chunking that combines multiple strategies"""
        # Clean and preprocess text
        text = self._clean_text(text)

        # Try paragraph chunking first
        para_chunks = self.paragraph_chunking(text)

        # If paragraphs are too long, apply semantic chunking to them
        final_chunks = []
        for para_chunk in para_chunks:
            if len(para_chunk.content) > 600:
                # Apply semantic chunking to long paragraphs
                semantic_chunks = self.semantic_chunking(para_chunk.content, 350, 30)
                for sem_chunk in semantic_chunks:
                    sem_chunk.source = doc_source
                    sem_chunk.metadata["parent_strategy"] = "paragraph"
                    sem_chunk.metadata["parent_chunk_id"] = para_chunk.chunk_id
                    final_chunks.append(sem_chunk)
            else:
                para_chunk.source = doc_source
                final_chunks.append(para_chunk)

        # Filter very short chunks and merge if needed
        final_chunks = self._filter_and_merge_chunks(final_chunks, doc_source)

        return final_chunks

    def _clean_text(self, text: str) -> str:
        """Clean text for better chunking"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters that might interfere
        text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\n]", "", text)
        # Ensure proper spacing after punctuation
        text = re.sub(r"([.!?])\s*", r"\1 ", text)
        return text.strip()

    def _filter_and_merge_chunks(
        self, chunks: List[DocumentChunk], doc_source: str
    ) -> List[DocumentChunk]:
        """Filter very short chunks and merge adjacent ones"""
        filtered_chunks = []

        for chunk in chunks:
            # Only keep chunks with meaningful content
            if len(chunk.content.strip()) >= 50:
                filtered_chunks.append(chunk)

        # If we have very short chunks, try to merge with neighbors
        if len(filtered_chunks) > 1:
            merged_chunks = []
            i = 0
            while i < len(filtered_chunks):
                current_chunk = filtered_chunks[i]

                # If current chunk is too short, try to merge with next
                if len(current_chunk.content) < 100 and i + 1 < len(filtered_chunks):
                    next_chunk = filtered_chunks[i + 1]

                    # Merge chunks
                    merged_content = current_chunk.content + "\n\n" + next_chunk.content
                    merged_chunk = DocumentChunk(
                        content=merged_content,
                        source=doc_source,
                        chunk_id=f"merged_{len(merged_chunks)}",
                        start_char=current_chunk.start_char,
                        end_char=next_chunk.end_char,
                        metadata={
                            "strategy": "merged",
                            "original_chunks": [
                                current_chunk.chunk_id,
                                next_chunk.chunk_id,
                            ],
                            "char_count": len(merged_content),
                        },
                    )
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip the next chunk as it's merged
                else:
                    merged_chunks.append(current_chunk)
                    i += 1

            filtered_chunks = merged_chunks

        return filtered_chunks


class QdrantStorage:
    """Store and manage document chunks in Qdrant"""

    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = "document_chunks"

    def ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Created collection: {self.collection_name}")

    def generate_chunk_id(self, content: str, source: str) -> str:
        """Generate unique chunk ID as UUID"""
        unique_string = f"{source}_{len(content)}_{content[:100]}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

    def store_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """Store chunks in Qdrant"""
        self.ensure_collection_exists()

        points = []
        for chunk in chunks:
            # Generate unique ID
            chunk_id = self.generate_chunk_id(chunk.content, chunk.source)

            # Embed content
            embedding = self.embedding_model.encode(chunk.content).tolist()

            # Create point
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "content": chunk.content,
                    "source": chunk.source,
                    "chunk_id": chunk_id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                },
            )
            points.append(point)

        # Upload in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            print(
                f"Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}"
            )

        print(f"Successfully stored {len(points)} chunks in Qdrant")
        return len(points)


class DataPipeline:
    """Complete data processing pipeline"""

    def __init__(
        self,
        docs_dir: str = "data/docs",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.docs_dir = Path(docs_dir)
        self.processor = DocumentProcessor()
        self.chunker = AdvancedChunker(SentenceTransformer("all-MiniLM-L6-v2"))
        self.storage = QdrantStorage(qdrant_host, qdrant_port)

    def discover_documents(self) -> List[str]:
        """Discover all documents in the docs directory"""
        supported_extensions = [".pdf", ".txt", ".md"]
        documents = []

        for ext in supported_extensions:
            documents.extend(self.docs_dir.glob(f"*{ext}"))

        return [str(doc) for doc in documents]

    def preview_document_chunks(
        self, doc_file: str, chunking_strategy: str = "hybrid"
    ) -> Dict[str, Any]:
        """Preview chunking results for a single document without storing"""
        print(f"Previewing chunks for: {doc_file}")

        try:
            # Process document
            doc_data = self.processor.process_document(doc_file)

            # Chunk document
            if chunking_strategy == "semantic":
                chunks = self.chunker.semantic_chunking(doc_data["content"])
            elif chunking_strategy == "fixed":
                chunks = self.chunker.fixed_size_chunking(doc_data["content"])
            elif chunking_strategy == "paragraph":
                chunks = self.chunker.paragraph_chunking(doc_data["content"])
            else:  # hybrid (default)
                chunks = self.chunker.hybrid_chunking(
                    doc_data["content"], doc_data["source"]
                )

            # Update chunk source
            for chunk in chunks:
                chunk.source = doc_data["source"]
                chunk.metadata["filepath"] = doc_data["filepath"]
                chunk.metadata["file_type"] = doc_data["file_type"]

            # Create preview
            preview = {
                "document": doc_data,
                "chunking_strategy": chunking_strategy,
                "total_chunks": len(chunks),
                "chunks_preview": [],
            }

            # Show first few chunks as preview
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                preview["chunks_preview"].append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "content_preview": chunk.content[:200] + "..."
                        if len(chunk.content) > 200
                        else chunk.content,
                        "char_count": len(chunk.content),
                        "strategy": chunk.metadata.get("strategy", "unknown"),
                    }
                )

            if len(chunks) > 3:
                preview["remaining_chunks"] = len(chunks) - 3

            return preview

        except Exception as e:
            print(f"Error previewing {doc_file}: {e}")
            return {"error": str(e)}

    def process_all_documents(
        self, chunking_strategy: str = "hybrid", preview_only: bool = False
    ) -> Dict[str, Any]:
        """Process all documents and store chunks"""
        # Discover documents
        doc_files = self.discover_documents()
        print(f"Found {len(doc_files)} documents to process")

        if not doc_files:
            print("No documents found in data/docs directory")
            return {"processed": 0, "chunks_created": 0, "documents": []}

        all_chunks = []
        processed_docs = []
        chunk_statistics = {
            "total_chars": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": float("inf"),
            "max_chunk_size": 0,
            "strategies_used": set(),
        }

        for doc_file in doc_files:
            try:
                print(f"Processing: {doc_file}")

                # Process document
                doc_data = self.processor.process_document(doc_file)

                # Chunk document
                if chunking_strategy == "semantic":
                    chunks = self.chunker.semantic_chunking(doc_data["content"])
                elif chunking_strategy == "fixed":
                    chunks = self.chunker.fixed_size_chunking(doc_data["content"])
                elif chunking_strategy == "paragraph":
                    chunks = self.chunker.paragraph_chunking(doc_data["content"])
                else:  # hybrid (default)
                    chunks = self.chunker.hybrid_chunking(
                        doc_data["content"], doc_data["source"]
                    )

                # Update chunk source and collect statistics
                for chunk in chunks:
                    chunk.source = doc_data["source"]
                    chunk.metadata["filepath"] = doc_data["filepath"]
                    chunk.metadata["file_type"] = doc_data["file_type"]
                    chunk.metadata["doc_category"] = self._categorize_document(
                        doc_data["source"]
                    )

                    # Update statistics
                    chunk_size = len(chunk.content)
                    chunk_statistics["total_chars"] += chunk_size
                    chunk_statistics["min_chunk_size"] = min(
                        chunk_statistics["min_chunk_size"], chunk_size
                    )
                    chunk_statistics["max_chunk_size"] = max(
                        chunk_statistics["max_chunk_size"], chunk_size
                    )
                    chunk_statistics["strategies_used"].add(
                        chunk.metadata.get("strategy", "unknown")
                    )

                all_chunks.extend(chunks)
                processed_docs.append(
                    {
                        "filepath": doc_file,
                        "source": doc_data["source"],
                        "chunks_created": len(chunks),
                        "char_count": doc_data["size"],
                        "category": self._categorize_document(doc_data["source"]),
                    }
                )

                print(
                    f"  Created {len(chunks)} chunks from {doc_data['size']} characters"
                )

            except Exception as e:
                print(f"Error processing {doc_file}: {e}")
                continue

        # Calculate final statistics
        if all_chunks:
            chunk_statistics["avg_chunk_size"] = chunk_statistics["total_chars"] / len(
                all_chunks
            )
            chunk_statistics["strategies_used"] = list(
                chunk_statistics["strategies_used"]
            )
        else:
            chunk_statistics["min_chunk_size"] = 0

        # Store chunks in Qdrant (unless preview only)
        stored_count = 0
        if not preview_only and all_chunks:
            stored_count = self.storage.store_chunks(all_chunks)

        # Save processing summary
        summary = {
            "processed_documents": len(processed_docs),
            "total_chunks_created": len(all_chunks),
            "chunks_stored": stored_count,
            "chunking_strategy": chunking_strategy,
            "preview_only": preview_only,
            "documents": processed_docs,
            "statistics": chunk_statistics,
        }

        # Save summary to file
        with open("data/processing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Processing Summary ===")
        print(f"Documents processed: {summary['processed_documents']}")
        print(f"Chunks created: {summary['total_chunks_created']}")
        if not preview_only:
            print(f"Chunks stored: {summary['chunks_stored']}")
        print(f"Strategy used: {chunking_strategy}")
        print(f"Average chunk size: {chunk_statistics['avg_chunk_size']:.1f} chars")
        print(
            f"Chunk size range: {chunk_statistics['min_chunk_size']} - {chunk_statistics['max_chunk_size']} chars"
        )

        return summary

    def _categorize_document(self, filename: str) -> str:
        """Categorize document based on filename"""
        filename_lower = filename.lower()

        if "access" in filename_lower or "sop" in filename_lower:
            return "access_control"
        elif (
            "hr" in filename_lower
            or "leave" in filename_lower
            or "policy" in filename_lower
        ):
            return "hr_policy"
        elif (
            "helpdesk" in filename_lower
            or "faq" in filename_lower
            or "it" in filename_lower
        ):
            return "it_support"
        elif "refund" in filename_lower:
            return "refund_policy"
        elif "sla" in filename_lower:
            return "service_level"
        else:
            return "general"


def main():
    """Run the complete data pipeline"""
    print("=== Data Chunking and Storage Pipeline ===")

    # Initialize pipeline
    pipeline = DataPipeline()

    # First, preview chunking for one document to see the results
    print("\n--- Preview Mode ---")
    doc_files = pipeline.discover_documents()
    if doc_files:
        preview_file = doc_files[0]  # Preview first document
        preview = pipeline.preview_document_chunks(preview_file, "hybrid")

        print(f"Preview for {preview_file}:")
        print(f"  Total chunks: {preview.get('total_chunks', 0)}")
        print(f"  Document size: {preview['document']['size']} chars")

        if "chunks_preview" in preview:
            for i, chunk_preview in enumerate(preview["chunks_preview"]):
                print(
                    f"  Chunk {i + 1} ({chunk_preview['strategy']}): {chunk_preview['char_count']} chars"
                )
                print(f"    Preview: {chunk_preview['content_preview'][:100]}...")

    # Process all documents with hybrid chunking
    print("\n--- Processing All Documents ---")
    summary = pipeline.process_all_documents(chunking_strategy="hybrid")

    print(f"\nPipeline completed successfully!")
    print(f"Summary saved to: data/processing_summary.json")

    # Show some statistics
    if "statistics" in summary:
        stats = summary["statistics"]
        print(f"\n=== Chunking Statistics ===")
        print(f"Documents processed: {summary['processed_documents']}")
        print(f"Total chunks created: {summary['total_chunks_created']}")
        print(f"Average chunk size: {stats['avg_chunk_size']:.1f} characters")
        print(
            f"Chunk size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} characters"
        )
        print(f"Strategies used: {', '.join(stats['strategies_used'])}")


if __name__ == "__main__":
    main()
