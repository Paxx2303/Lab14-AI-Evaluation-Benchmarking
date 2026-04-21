# RAG Evaluation Benchmarking System

Complete implementation of Phase 1 and Phase 2 from Planning.md with Qdrant database, featuring two RAG system versions with comprehensive evaluation capabilities.

## System Overview

This project implements a sophisticated RAG (Retrieval-Augmented Generation) evaluation benchmarking system with:

- **Phase 1**: Challenging questions for chunk verification
- **Phase 2**: V1 (Simple) and V2 (Enhanced) RAG systems
- **Qdrant Database**: Vector storage with Docker containerization
- **Comprehensive Testing**: Automated comparison and evaluation

## Architecture

```
RAG Evaluation System
    |
    |-- Phase 1: Chunk Verification
    |   |-- Challenging Questions Generator
    |   |-- Chunk Verifier
    |   `-- Verification Metrics
    |
    |-- Phase 2: RAG Systems
    |   |-- V1: Simple RAG
    |   |   |-- Basic semantic search
    |   |   |-- Simple embeddings
    |   |   `-- Direct generation
    |   |
    |   `-- V2: Enhanced RAG
    |       |-- Hybrid search (semantic + keyword)
    |       |-- Dynamic chunking
    |       |-- Query enhancement
    |       |-- Neural re-ranking
    |       `-- Multi-stage pipeline
    |
    |-- Data Pipeline
    |   |-- Document processing
    |   |-- Advanced chunking
    |   `-- Qdrant storage
    |
    `-- Testing & Evaluation
        |-- System comparison
        |-- Performance metrics
        `-- Comprehensive reporting
```

## Quick Start

### Prerequisites

1. Docker and Docker Compose installed
2. Python 3.8+
3. OpenAI API key
4. Sufficient disk space for Qdrant data

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd Lab14-AI-Evaluation-Benchmarking
pip install -r requirements.txt
```

2. **Set up environment:**
```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

3. **Start Qdrant:**
```bash
docker-compose up -d
```

4. **Run complete setup and testing:**
```bash
python setup_and_run.py
```

## Manual Setup Steps

### Step 1: Start Qdrant Container
```bash
docker-compose up -d
```

This starts Qdrant with:
- HTTP API on port 6333
- gRPC API on port 6334
- Persistent data storage
- Health checks

### Step 2: Process Documents
```bash
python data/chunking_pipeline.py
```

This processes documents from `data/docs/` using hybrid chunking and stores them in Qdrant.

### Step 3: Run Phase 1 Verification
```bash
python phase1/challenging_questions.py
python phase1/chunk_verifier.py
```

Generates challenging questions and tests chunk retrieval accuracy.

### Step 4: Test RAG Systems
```bash
python rag_v1/simple_rag.py
python rag_v2/enhanced_rag.py
```

Tests both RAG versions with sample data.

### Step 5: Comprehensive Testing
```bash
python test_rag_systems.py
```

Runs complete comparison between V1 and V2 systems.

## System Components

### Phase 1: Chunk Verification

**Location**: `phase1/`

**Features**:
- **Challenging Question Types**:
  - Semantic Inference Questions
  - Negative Constraint Questions
  - Temporal Reasoning Questions
  - Comparative Analysis Questions
  - Context-Dependent Questions

- **Verification Metrics**:
  - Hit Rate@k
  - Mean Reciprocal Rank (MRR)
  - Precision@k
  - Pass Rate

**Usage**:
```python
from phase1.challenging_questions import ChallengingQuestionsGenerator
from phase1.chunk_verifier import ChunkVerifier

# Generate challenging questions
generator = ChallengingQuestionsGenerator()
generator.save_questions("data/challenging_questions.json")

# Verify chunk retrieval
verifier = ChunkVerifier()
results = verifier.run_verification("data/challenging_questions.json")
```

### V1: Simple RAG System

**Location**: `rag_v1/simple_rag.py`

**Features**:
- Simple semantic search with embeddings
- Basic prompt template
- Direct OpenAI generation
- Minimal configuration

**Architecture**:
```
Query -> Embed -> Search -> Generate -> Answer
```

**Usage**:
```python
from rag_v1.simple_rag import SimpleRAG

rag = SimpleRAG()
result = rag.query("What is RAG evaluation?")
print(result.answer)
```

### V2: Enhanced RAG System

**Location**: `rag_v2/enhanced_rag.py`

**Features**:
- **Hybrid Search**: Semantic + BM25 keyword search
- **Dynamic Chunking**: Adaptive sizing with semantic boundaries
- **Query Enhancement**: Expansion and intent classification
- **Neural Re-ranking**: Cross-encoder re-scoring
- **Multi-stage Pipeline**: 4-stage retrieval process

**Architecture**:
```
Query -> Query Enhancement -> Broad Search -> Keyword Refinement -> Neural Re-ranking -> Generate -> Answer
```

**Pipeline Stages**:
1. **Query Enhancement**: Generate alternative queries and classify intent
2. **Broad Search**: Semantic search (top-k=50)
3. **Keyword Refinement**: BM25 filtering (top-k=20)
4. **Neural Re-ranking**: Cross-encoder scoring (final top-k=5)

**Usage**:
```python
from rag_v2.enhanced_rag import EnhancedRAG

rag = EnhancedRAG()
result = rag.query("Compare semantic and keyword search approaches")
print(f"Method: {result.retrieval_method}")
print(f"Query Expansion: {result.query_expansion}")
print(f"Pipeline Times: {result.pipeline_stages}")
print(result.answer)
```

### Data Pipeline

**Location**: `data/chunking_pipeline.py`

**Features**:
- **Multi-format Support**: PDF, TXT, MD files
- **Advanced Chunking Strategies**:
  - Semantic chunking
  - Fixed-size chunking
  - Paragraph chunking
  - Hybrid chunking (recommended)
- **Qdrant Integration**: Automatic storage with metadata

**Chunking Strategies**:

1. **Semantic Chunking**: Breaks at semantic boundaries
2. **Fixed-size Chunking**: Uniform size with overlap
3. **Paragraph Chunking**: Natural paragraph breaks
4. **Hybrid Chunking**: Combines multiple strategies adaptively

**Usage**:
```python
from data.chunking_pipeline import DataPipeline

pipeline = DataPipeline()
summary = pipeline.process_all_documents(chunking_strategy="hybrid")
print(f"Processed {summary['chunks_stored']} chunks")
```

## Performance Comparison

### V1 vs V2 Expected Improvements

Based on Planning.md targets, V2 should outperform V1 by:

- **Hit Rate**: +15% improvement
- **Answer Quality**: Enhanced context and re-ranking
- **Query Understanding**: Better handling of complex queries
- **Retrieval Precision**: Hybrid search vs semantic-only

### Metrics Tracked

**Retrieval Metrics**:
- Hit Rate@k
- Mean Reciprocal Rank (MRR)
- Precision@k
- Retrieval latency

**Generation Metrics**:
- Answer relevance
- Faithfulness to context
- Response completeness

**System Metrics**:
- End-to-end latency
- Token usage
- Cost efficiency

## File Structure

```
Lab14-AI-Evaluation-Benchmarking/
    |
    |-- docker-compose.yml          # Qdrant container setup
    |-- qdrant/
    |   `-- config.yaml            # Qdrant configuration
    |
    |-- phase1/                    # Phase 1: Chunk Verification
    |   |-- challenging_questions.py
    |   `-- chunk_verifier.py
    |
    |-- rag_v1/                    # V1: Simple RAG
    |   `-- simple_rag.py
    |
    |-- rag_v2/                    # V2: Enhanced RAG
    |   `-- enhanced_rag.py
    |
    |-- data/                      # Data processing
    |   |-- chunking_pipeline.py
    |   |-- docs/                  # Document storage
    |   `-- synthetic_gen.py       # Synthetic data generation
    |
    |-- test_results/              # Test outputs
    |-- setup_and_run.py          # Complete setup script
    |-- test_rag_systems.py       # Comprehensive testing
    `-- requirements.txt          # Dependencies
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_MODEL=gpt-3.5-turbo
```

### Qdrant Configuration

The `qdrant/config.yaml` file contains:
- Performance settings
- Memory limits
- API configurations
- Logging preferences

## Testing and Evaluation

### Running Tests

**Complete Test Suite**:
```bash
python test_rag_systems.py
```

**Individual Components**:
```bash
# Phase 1 verification
python phase1/chunk_verifier.py

# V1 testing
python rag_v1/simple_rag.py

# V2 testing
python rag_v2/enhanced_rag.py
```

### Test Reports

After running tests, check:
- `test_results/comprehensive_test_report.json` - Detailed results
- `test_results/test_summary.json` - Executive summary
- `data/processing_summary.json` - Data processing stats
- `phase1/verification_results.json` - Chunk verification metrics

### Expected Results

Based on Planning.md requirements:

**Phase 1 Success Criteria**:
- 50+ challenging test cases created
- Hit Rate > 70% on chunk verification
- MRR > 0.5 for semantic questions

**Phase 2 Success Criteria**:
- V2 outperforms V1 by >15% on Hit Rate
- Enhanced answer quality through re-ranking
- Successful multi-stage pipeline execution

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**:
   - Ensure Docker is running
   - Check if ports 6333/6334 are available
   - Verify container is healthy: `docker-compose ps`

2. **OpenAI API Errors**:
   - Check API key is set correctly
   - Verify sufficient API credits
   - Check rate limits

3. **Document Processing Errors**:
   - Ensure documents exist in `data/docs/`
   - Check file formats are supported
   - Verify sufficient memory for large documents

4. **Import Errors**:
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version compatibility
   - Verify virtual environment activation

### Debug Mode

Enable debug logging by setting:
```bash
export LOG_LEVEL=DEBUG
```

## Contributing

### Adding New Document Types

Extend `DocumentProcessor` in `data/chunking_pipeline.py`:

```python
@staticmethod
def process_new_format(filepath: str) -> Dict[str, Any]:
    # Implementation for new format
    pass
```

### Adding New Chunking Strategies

Extend `AdvancedChunker` class:

```python
def custom_chunking(self, text: str) -> List[DocumentChunk]:
    # Custom chunking implementation
    pass
```

### Adding New Question Types

Extend `ChallengingQuestionsGenerator`:

```python
def generate_custom_questions(self) -> List[ChallengingQuestion]:
    # Custom question generation
    pass
```

## License

This project implements the Planning.md specifications for AI Evaluation Benchmarking. All components are designed for research and evaluation purposes.

## Support

For issues and questions:
1. Check this README and Planning.md
2. Review test reports and logs
3. Verify environment setup
4. Check Qdrant container status
