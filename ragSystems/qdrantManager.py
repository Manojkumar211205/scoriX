from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct, SparseVector,
    SearchParams, Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue, MatchAny
)

class QdrantManager:
    def __init__(self, collection_name="rag_m3", vector_size=1024, host="localhost", port=6333):
        """Initialize local Qdrant connection"""
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

        # Create or reset collection with NAMED dense + named sparse vectors
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
            },  # Named dense
            sparse_vectors_config={
                "sparse": SparseVectorParams()  # Named sparse (fixed to DOT)
            }
        )
        print(f"✅ Collection '{collection_name}' ready for hybrid embeddings")

    def upsertPoints(self, dense_vectors, sparse_vectors, payloads, batch_size=50):
        """Store both dense & sparse embeddings"""
        # Flatten batches: list of (seq_len, dim) tensors/arrays, one per chunk
        all_dense_chunks = [chunk for batch in dense_vectors for chunk in batch]

        points = []
        for i, (dense_chunk, sparse_dict, data) in enumerate(zip(all_dense_chunks, sparse_vectors, payloads)):
            # Pool dense: mean over sequence length
            if hasattr(dense_chunk, "mean"):  # Torch tensor
                pooled_dense = dense_chunk.mean(dim=0).cpu().numpy()
            else:  # NumPy array
                pooled_dense = np.mean(dense_chunk, axis=0)

            # Convert sparse {int_token_id: float_count} to SparseVector
            sparse_sv = self.convert_sparse_dict_to_qdrant_format(sparse_dict)

            points.append(PointStruct(
                id=i + 1,
                vector={  # Singular 'vector' for named multi-vectors
                    "dense": pooled_dense.tolist(),
                    "sparse": sparse_sv  # SparseVector object
                },
                payload={"text": data}
            ))
        
        # Batch upsert
        total_points = len(points)
        print(f"Total points to upsert: {total_points}")
        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                print(f"✅ Upserted batch {i // batch_size + 1}/{(total_points + batch_size - 1) // batch_size} ({len(batch)} points)")
            except Exception as e:
                print(f"❌ Error upserting batch {i}: {e}")
        print(f"✅ Upserted {len(points)} points successfully.")

    def convert_sparse_dict_to_qdrant_format(self, sparse_dict):
        """
        sparse_dict: {token_id_int: float_count, ...}
        Returns: SparseVector with sorted indices/values
        """
        if not sparse_dict:
            return SparseVector(indices=[], values=[])
        indices = sorted(sparse_dict.keys())
        values = [float(sparse_dict[idx]) for idx in indices]
        return SparseVector(indices=indices, values=values)

    def hybrid_search(self, dense_emb, sparse_vec, top_k=5, fusion="RRF"):
        """Perform hybrid search combining dense + sparse via prefetch fusion"""
        # Handle list input (e.g., [tensor] from single-batch encode)
        if isinstance(dense_emb, list):
            if len(dense_emb) == 1:
                dense_emb = dense_emb[0]
            else:
                raise ValueError("hybrid_search expects single query; got multi-batch dense_emb list")

        # Handle if sparse_vec is list (from batch=1)
        if isinstance(sparse_vec, list):
            sparse_vec = sparse_vec[0]

        # Pool dense: mean over sequence length to (dim,)
        if hasattr(dense_emb, 'shape') and len(dense_emb.shape) == 3:  # (1, seq, dim)
            pooled_dense = dense_emb.mean(dim=1)[0].cpu().numpy()
        elif hasattr(dense_emb, 'shape') and len(dense_emb.shape) == 2:  # (seq, dim)
            pooled_dense = dense_emb.mean(dim=0).cpu().numpy()
        else:  # Already pooled or fallback
            if hasattr(dense_emb, 'cpu'):
                pooled_dense = dense_emb.cpu().numpy()
            else:
                pooled_dense = np.array(dense_emb)

        # Convert to SparseVector
        sparse_sv = self.convert_sparse_dict_to_qdrant_format(sparse_vec)

        # Hybrid via prefetch: separate searches + fusion (RRF or DBSF)
        prefetch_limit = top_k * 3  # Oversample for better fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=sparse_sv,
                    using="sparse",
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=pooled_dense.tolist(),
                    using="dense",
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=getattr(Fusion, fusion)),  # e.g., Fusion.RRF
            search_params=SearchParams(hnsw_ef=128),  # Tune dense HNSW (optional)
            limit=top_k
        )

        print("\n🔍 Top Results (Hybrid Fusion: {}):".format(fusion))
        for r in results.points:
            print(f"→ Score: {r.score:.4f} | Text: {r.payload['text']}")
        return results


class HybridQdrantManager:
    def __init__(self, collection_name="rag_hybrid", vector_size=384, host="localhost", port=6333):
        """Initialize local Qdrant connection for Hybrid Search with fallback to in-memory"""
        self.collection_name = collection_name
        
        # Try to connect to Qdrant server, fallback to in-memory if it fails
        try:
            print(f"🔄 Attempting to connect to Qdrant at {host}:{port}...")
            self.client = QdrantClient(host=host, port=port, timeout=5)
            # Test connection
            self.client.get_collections()
            print(f"✅ Connected to Qdrant server at {host}:{port}")
        except Exception as e:
            print(f"⚠️  Could not connect to Qdrant server: {e}")
            print(f"🔄 Falling back to in-memory Qdrant...")
            self.client = QdrantClient(":memory:")
            print(f"✅ Using in-memory Qdrant (data will not persist)")

        # Create collection if it doesn't exist
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                print(f"📦 Collection '{self.collection_name}' already exists, using existing collection")
            else:
                print(f"📦 Creating collection '{self.collection_name}' with vector_size={vector_size}...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    }
                )
                print(f"✅ Collection '{self.collection_name}' created successfully (vector_size={vector_size})")
        except Exception as e:
            print(f"❌ Error managing collection: {e}")
            raise

    def convert_sparse_dict_to_qdrant_format(self, sparse_dict):
        """
        sparse_dict: {token_id_int: float_count, ...}
        Returns: SparseVector with sorted indices/values
        """
        if not sparse_dict:
            return SparseVector(indices=[], values=[])
        indices = sorted(sparse_dict.keys())
        values = [float(sparse_dict[idx]) for idx in indices]
        return SparseVector(indices=indices, values=values)

    def upsert_integrated_hybrid(self, dense_vectors, sparse_vectors, metadata, batch_size=50):
        """
        Store single-vector sentence embeddings + sparse + metadata.
        dense_vectors: List of 1D arrays/lists (one per sentence).
        sparse_vectors: List of sparse dicts {token_id: frequency}.
        metadata: List of dicts containing payload data.
        """
        points = []
        for i, (dense_vec, sparse_dict, meta) in enumerate(zip(dense_vectors, sparse_vectors, metadata)):
            
            # Ensure dense_vec is a list (if numpy or tensor)
            if hasattr(dense_vec, "tolist"):
                dense_vec = dense_vec.tolist()
            
            # Convert sparse dict to SparseVector
            sparse_sv = self.convert_sparse_dict_to_qdrant_format(sparse_dict)

            points.append(PointStruct(
                id=i + 1, # Simple incremental ID (consider UUIDs for production)
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_sv
                },
                payload=meta # Store full metadata dictionary as payload
            ))
        
        # Batch upsert
        total_points = len(points)
        print(f"Total points to upsert: {total_points}")
        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                print(f"✅ Upserted batch {i // batch_size + 1}/{(total_points + batch_size - 1) // batch_size} ({len(batch)} points)")
            except Exception as e:
                print(f"❌ Error upserting batch {i}: {e}")
        print(f"✅ Upserted {len(points)} hybrid points with metadata successfully.")

    def search_integrated_hybrid(self, query_dense, query_sparse, metadata_filter=None, top_k=5, fusion="RRF"):
        """
        Hybrid search with metadata filtering.
        query_dense: Single 1D vector (sentence embedding).
        query_sparse: Single sparse dict.
        metadata_filter: Dict for exact match filtering (e.g., {"category": "news"}).
        """
        # Prepare Filter
        qdrant_filter = None
        if metadata_filter:
            must_conditions = []
            for key, value in metadata_filter.items():
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            qdrant_filter = Filter(must=must_conditions)

        # Prepare Dense Vector
        if hasattr(query_dense, 'cpu'):
            query_dense = query_dense.cpu().numpy()
        if hasattr(query_dense, 'tolist'):
            query_dense = query_dense.tolist()
            
        # Prepare Sparse Vector
        query_sparse_sv = self.convert_sparse_dict_to_qdrant_format(query_sparse)
        
        prefetch_limit = top_k * 3
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_sparse_sv,
                    using="sparse",
                    filter=qdrant_filter, # Apply filter to sparse
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=query_dense,
                    using="dense",
                    filter=qdrant_filter, # Apply filter to dense
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=getattr(Fusion, fusion)),
            limit=top_k
        )

        print(f"\n🔍 Top Results (Hybrid {fusion} + Filter: {metadata_filter}):")
        for r in results.points:
            print(f"→ Score: {r.score:.4f} | Payload: {r.payload}")
        return results