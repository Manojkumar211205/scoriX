from ragSystems.embedder import TextEmbedder, HybridEmbedder
from ragSystems.qdrantManager import QdrantManager, HybridQdrantManager



class ragProcessor:
    def __init__(self,collection_name):
        self.textEmbedder = TextEmbedder()
        self.qdrantManager = QdrantManager(collection_name=collection_name)

    def docStoring(self,chunks):
        embeddings = self.textEmbedder.embedChunksInMultiVector(chunks)
        spares = self.textEmbedder.getSparsEmbeddings(chunks)
        self.qdrantManager.upsertPoints(embeddings, spares, chunks,)

    def search(self,query):
        embeddings = self.textEmbedder.multiVectorEmbedder(query)
        spares = self.textEmbedder.getSparsEmbeddings(query)
        result = self.qdrantManager.hybrid_search(embeddings, spares)
        output = ""
        for r in result.points:
            output += r.payload['text']
            output += "\n"

        return output


class HybridRagProcessor:
    def __init__(self, collection_name="rag_hybrid_final"):
        self.textEmbedder = HybridEmbedder()
        self.qdrantManager = HybridQdrantManager(collection_name=collection_name)

    def process_and_store(self, chunks: list[str], metadata: list[dict]):
        """
        Process text chunks and store them with metadata.
        chunks: List of text strings.
        metadata: List of dicts corresponding to chunks.
        """
        print("Embeddings generation started...")
        dense_vecs = self.textEmbedder.embed_text(chunks)
        sparse_vecs = self.textEmbedder.get_sparse_embeddings(chunks)
        print("Embeddings generated.")
        
        self.qdrantManager.upsert_integrated_hybrid(dense_vecs, sparse_vecs, metadata)

    def search(self, query, metadata_filter: dict = None, top_k: int = 3):
        """
        Search for query with optional metadata filtering.
        """
        dense_vec = self.textEmbedder.embed_text(query)[0] # Take the first (and only) vector
        sparse_vec = self.textEmbedder.get_sparse_embeddings(query)[0]
        
        results = self.qdrantManager.search_integrated_hybrid(
            query_dense=dense_vec,
            query_sparse=sparse_vec,
            metadata_filter=metadata_filter,
            top_k=top_k
        )
        
        output = []
        for r in results.points:
            output.append(r.payload)
            
        return output

