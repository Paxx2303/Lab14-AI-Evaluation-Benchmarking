"""
Setup and Run Script for RAG Evaluation Benchmarking
Sets up Qdrant, processes documents, and runs the complete test suite
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    print("Checking requirements...")
    
    required_packages = [
        "qdrant-client", "sentence-transformers", "openai", 
        "python-dotenv", "pandas", "tqdm", "jinja2", "numpy",
        "langchain", "rank-bm25", "scikit-learn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("All required packages are installed")
    return True

def check_qdrant_connection():
    """Check if Qdrant is running"""
    print("Checking Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print("Qdrant connection successful")
        return True
    except Exception as e:
        print(f"Qdrant connection failed: {e}")
        print("Please start Qdrant using: docker-compose up -d")
        return False

def setup_environment():
    """Setup environment variables and directories"""
    print("Setting up environment...")
    
    # Create necessary directories
    directories = ["test_results", "qdrant", "phase1", "rag_v1", "rag_v2", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in .env file or environment")
        return False
    
    print("Environment setup complete")
    return True

def start_qdrant():
    """Start Qdrant container"""
    print("Starting Qdrant container...")
    
    try:
        # Check if docker-compose file exists
        if not os.path.exists("docker-compose.yml"):
            print("docker-compose.yml not found")
            return False
        
        # Start container
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("Qdrant container started successfully")
            
            # Wait for Qdrant to be ready
            print("Waiting for Qdrant to be ready...")
            time.sleep(10)
            
            return True
        else:
            print(f"Failed to start Qdrant: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error starting Qdrant: {e}")
        return False

def process_documents():
    """Process documents and store in Qdrant"""
    print("Processing documents...")
    
    try:
        # Import and run the data pipeline
        sys.path.append(str(Path(__file__).parent / "data"))
        from chunking_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        summary = pipeline.process_all_documents(chunking_strategy="hybrid")
        
        print(f"Document processing complete: {summary['chunks_stored']} chunks stored")
        return True
        
    except Exception as e:
        print(f"Error processing documents: {e}")
        return False

def run_tests():
    """Run the complete test suite"""
    print("Running test suite...")
    
    try:
        from test_rag_systems import RAGSystemTester
        
        tester = RAGSystemTester()
        results = tester.run_complete_test_suite()
        
        print("Test suite completed successfully")
        return True
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main setup and run process"""
    print("=== RAG Evaluation Benchmarking Setup ===")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("Setup failed at requirements check")
        return False
    
    # Step 2: Setup environment
    if not setup_environment():
        print("Setup failed at environment setup")
        return False
    
    # Step 3: Start Qdrant
    if not start_qdrant():
        print("Setup failed at Qdrant startup")
        return False
    
    # Step 4: Check Qdrant connection
    if not check_qdrant_connection():
        print("Setup failed at Qdrant connection check")
        return False
    
    # Step 5: Process documents
    if not process_documents():
        print("Setup failed at document processing")
        return False
    
    # Step 6: Run tests
    if not run_tests():
        print("Setup failed at test execution")
        return False
    
    print("\n=== Setup Complete ===")
    print("All systems are ready! Check the following files:")
    print("  - test_results/comprehensive_test_report.json")
    print("  - test_results/test_summary.json")
    print("  - data/processing_summary.json")
    print("  - phase1/verification_results.json")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)
