from embedder import TextEmbedder
from qdrantManager import QdrantManager

textEmbedder = TextEmbedder()
qdrantManager = QdrantManager()

chunks = [
    "Deep learning has transformed artificial intelligence.",
    "Neural networks are widely used in image and speech recognition.",
    "Python is the most popular programming language for AI projects.",
    "Machine learning algorithms improve automatically with data.",
    "Reinforcement learning allows agents to learn by trial and error.",
    "Natural Language Processing helps computers understand human language.",
    "Computer vision enables machines to interpret and analyze images.",
    "Data preprocessing is crucial for building effective machine learning models.",
    "Supervised learning uses labeled datasets to train models.",
    "Unsupervised learning identifies patterns in unlabeled data."
]

embeddings = textEmbedder.embedChunksInMultiVector(chunks)
spares = textEmbedder.getSparsEmbeddings(chunks)
qdrantManager.upsertPoints(embeddings, spares,chunks)
query = "which changed ai"
embeddings = textEmbedder.multiVectorEmbedder(query)
spares = textEmbedder.getSparsEmbeddings(query)
qdrantManager.hybrid_search(embeddings, spares)

print("\n" + "="*50)
print("TESTING NEW HYBRID SEARCH WITH METADATA")
print("="*50)

# 1. Prepare Data
sentences = [
    "The stock market crashed today.",
    "The new movie release broke box office records.",
    "The government passed a new healthcare bill.",
    "Scientists discovered a new planet in the solar system."
]
meta_list = [
    {"category": "finance", "source": "Bloomberg"},
    {"category": "entertainment", "source": "Variety"},
    {"category": "politics", "source": "Reuters"},
    {"category": "science", "source": "NASA"}
]

# 2. Generate Embeddings (1D sentence vectors)
dense_vecs = textEmbedder.embed_text(sentences) # Returns list of 1D arrays
sparse_vecs = textEmbedder.getSparsEmbeddings(sentences)

# 3. Upsert with Metadata
qdrantManager.upsert_integrated_hybrid(dense_vecs, sparse_vecs, meta_list)

# 4. Search with Metadata Filter
query_text = "news about money"
query_dense = textEmbedder.embed_text(query_text)[0] # Take first (and only) vector
query_sparse = textEmbedder.getSparsEmbeddings(query_text)[0]

print("\n--- Search: 'news about money' (Filter: category='finance') ---")
qdrantManager.search_integrated_hybrid(
    query_dense, 
    query_sparse, 
    metadata_filter={"category": "finance"}
)

print("\n--- Search: 'news about money' (Filter: category='science') [Should look irrelevant or empty] ---")
qdrantManager.search_integrated_hybrid(
    query_dense, 
    query_sparse, 
    metadata_filter={"category": "science"}
)