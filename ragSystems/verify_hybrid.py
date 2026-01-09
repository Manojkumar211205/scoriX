
import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Go up two levels from d:\scoriX_agent\ragSystems\verify_hybrid.py? No, wait. 
# File is in d:\scoriX_agent\ragSystems, so project root d:\scoriX_agent is one level up.
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def test_hybrid_search():
    print("Initializing Hybrid Module Mocks...")
    
    # Mocking QdrantClient to avoid real connection issues during quick test
    sys.modules["qdrant_client"] = MagicMock()
    sys.modules["qdrant_client.models"] = MagicMock()
    sys.modules["sentence_transformers"] = MagicMock()
    sys.modules["transformers"] = MagicMock()
    sys.modules["torch"] = MagicMock()
    
    try:
        from ragSystems.ragProcessor import HybridRagProcessor
    except ImportError as e:
        print(f"Failed to import HybridRagProcessor: {e}")
        return

    print("Instantiation HybridRagProcessor...")
    # Mocking internals or assuming they work if imports succeeded
    # For a real test, we would need the actual Qdrant instance or a robust mock
    # Here we just verify the class structure and method signatures existence
    
    try:
        processor = HybridRagProcessor(collection_name="test_collection")
        print("HybridRagProcessor instantiated successfully.")
        
        # Test method availability
        if hasattr(processor, "process_and_store") and hasattr(processor, "search"):
            print("SUCCESS: HybridRagProcessor has expected methods.")
        else:
            print("FAILURE: Missing methods in HybridRagProcessor.")
            
    except Exception as e:
        print(f"Error during instantiation: {e}")

if __name__ == "__main__":
    test_hybrid_search()
