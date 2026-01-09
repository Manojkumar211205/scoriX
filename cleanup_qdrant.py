"""
Cleanup script to delete all Qdrant collections
Run this once to clear old collections with wrong vector sizes
"""
from qdrant_client import QdrantClient

try:
    print("Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333, timeout=10)
    collections = client.get_collections().collections
    
    print(f"\nFound {len(collections)} collections:")
    for col in collections:
        print(f"\n  📦 {col.name}")
        try:
            # Get collection info
            info = client.get_collection(col.name)
            print(f"     Vectors: {info.config.params.vectors}")
            
            # Delete
            print(f"     Deleting...")
            client.delete_collection(collection_name=col.name)
            print(f"     ✅ Deleted successfully")
        except Exception as e:
            print(f"     ❌ Error deleting: {e}")
    
    print("\n✅ Cleanup complete!")
    print("Now you can run your app and fresh collections will be created.")
    
except Exception as e:
    print(f"\n❌ Error connecting to Qdrant: {e}")
    print("\nMake sure Qdrant server is running on localhost:6333")
    print("Or the app will use in-memory Qdrant instead.")
