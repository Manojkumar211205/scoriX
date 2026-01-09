
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import torch
import gc


class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", use_gpu: bool = True):
        """
        Initialize the text embedder for manually chunked documents with GPU optimization

        Args:
            model_name: Name of the sentence transformer model to use
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        # Check GPU availability and set device
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        print(f"🚀 Initializing TextEmbedder on {self.device.upper()}")
        if self.use_gpu:
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load model once with optimizations
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.tokenizer = self.model.tokenizer  # Reuse tokenizer from SentenceTransformer to avoid extra loading

        # GPU-specific optimizations
        if self.use_gpu:
            # Enable mixed precision for faster inference
            self.model.half()  # Use FP16 for memory efficiency
            # Optimize for inference
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.enable_flash_sdp(True)

        print(f"✅ Model loaded successfully on {self.device}")

    def embed_text(self, text: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Embed a single text or list of texts with GPU optimization

        Args:
            text: Text string or list of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]

        # Use GPU-optimized encoding
        with torch.cuda.amp.autocast() if self.use_gpu else torch.no_grad():
            embeddings = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                device=self.device
            )

        # Clear GPU cache after processing
        if self.use_gpu:
            torch.cuda.empty_cache()

        return embeddings

    def embed_chunks(self, chunks: List[str], batch_size: int = None, show_progress: bool = True) -> List[np.ndarray]:
        """
        Embed a list of manually chunked documents with dynamic batch sizing
        """
        # Auto-calculate optimal batch size based on GPU memory
        if batch_size is None:
            if self.use_gpu:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory_gb >= 16:
                    batch_size = 34
                elif gpu_memory_gb >= 8:
                    batch_size = 12
                else:
                    batch_size = 5
            else:
                batch_size = 5

        print(f"🔄 Processing {len(chunks)} chunks with batch size {batch_size}")

        embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            if len(chunks) > 100 and show_progress:
                print(f"   Processing batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

            batch_embeddings = self.embed_text(batch, show_progress=False)

            # FIX: Properly handle 2D batch embeddings
            if batch_embeddings.ndim == 2:
                # Convert 2D batch to list of 1D embeddings
                for embedding in batch_embeddings:
                    embeddings.append(embedding)
            else:
                # Single embedding case
                embeddings.append(batch_embeddings)

            # Memory management for large datasets
            if self.use_gpu and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        print(f"✅ Completed embedding {len(chunks)} chunks")

        return embeddings
    def multiVectorEmbedder(self,texts):
        """get multi-vector embedder for list of texts"""
        if isinstance(texts, str):
            texts = [texts]

        dense_embeddings = self.model.encode(
            texts,
            output_value="token_embeddings",  # returns per-token embeddings
            convert_to_tensor=True
        )
        return dense_embeddings

    def embedChunksInMultiVector(self,chunks: List[str],batchSize=None,show_progress: bool = True) -> List[np.ndarray]:

        if batchSize is None:
            if self.use_gpu:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory_gb >= 16:
                    batchSize = 34
                elif gpu_memory_gb >= 8:
                    batchSize = 12
                else:
                    batchSize = 5
            else:
                batchSize = 5

        print(f"🔄 Processing {len(chunks)} chunks with batch size {batchSize}")

        embeddings = []

        for i in range(0, len(chunks), batchSize):
            batch = chunks[i:i + batchSize]

            if len(chunks) > 100 and show_progress:
                print(f"   Processing batch {i // batchSize + 1}/{(len(chunks) + batchSize - 1) // batchSize}")

            batch_embeddings = self.multiVectorEmbedder(batch)


            embeddings.append(batch_embeddings)

            # Memory management for large datasets
            if self.use_gpu and i % (batchSize * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        print(f"✅ Completed embedding {len(chunks)} chunks")

        return embeddings

    def getSparsEmbeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        sparse_vecs = []
        from collections import Counter
        for j in range(len(inputs['input_ids'])):
            token_ids = inputs['input_ids'][j].tolist()
            counts = Counter(token_ids)
            vec = {tid: float(counts[tid]) for tid in counts}
            sparse_vecs.append(vec)
        return sparse_vecs  # List of {int_token_id: float_count}
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.model.get_sentence_embedding_dimension()

    def get_gpu_info(self) -> dict:
        """Get GPU information and memory usage"""
        if not self.use_gpu:
            return {"gpu_available": False}

        return {
            "gpu_available": True,
            "device_name": torch.cuda.get_device_name(),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_memory_gb": torch.cuda.memory_reserved() / 1e9,
        }

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.use_gpu:
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 GPU cache cleared")

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


class HybridEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = True):
        """
        Initialize the HybridEmbedder with BAAI/bge-base-en-v1.5
        """
        # Check GPU availability and set device
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        print(f"🚀 Initializing HybridEmbedder on {self.device.upper()}")
        
        # Load model with timeout and local fallback
        try:
            # Try to load from cache first (local_files_only=True)
            print(f"   Attempting to load {model_name} from cache...")
            self.model = SentenceTransformer(
                model_name, 
                device=self.device, 
                trust_remote_code=True,
                local_files_only=True
            )
            print(f"   ✅ Loaded from cache")
        except Exception as e:
            # If not in cache, download with timeout
            print(f"   Not in cache, downloading from HuggingFace...")
            import os
            # Set shorter timeout for HTTP requests
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
            
            try:
                self.model = SentenceTransformer(
                    model_name, 
                    device=self.device, 
                    trust_remote_code=True
                )
                print(f"   ✅ Downloaded successfully")
            except Exception as download_error:
                print(f"   ❌ Failed to download model: {download_error}")
                raise RuntimeError(
                    f"Failed to load model {model_name}. "
                    f"Please check your internet connection or pre-download the model."
                ) from download_error
        
        self.tokenizer = self.model.tokenizer

        if self.use_gpu:
            self.model.half()
            
        print(f"✅ Hybrid Model {model_name} loaded successfully")

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text using the efficient bge-base model.
        Returns: Numpy array of embeddings.
        """
        if isinstance(text, str):
            text = [text]

        with torch.cuda.amp.autocast() if self.use_gpu else torch.no_grad():
            embeddings = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )

        if self.use_gpu:
            torch.cuda.empty_cache()
            
        return embeddings

    def get_sparse_embeddings(self, texts: Union[str, List[str]]):
        """
        Generate sparse embeddings (token frequency) for the given texts.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        sparse_vecs = []
        from collections import Counter
        
        for j in range(len(inputs['input_ids'])):
            token_ids = inputs['input_ids'][j].tolist()
            # Filter out special tokens if needed, but keeping simplistic for now matching previous logic
            counts = Counter(token_ids)
            vec = {tid: float(counts[tid]) for tid in counts}
            sparse_vecs.append(vec)
            
        return sparse_vecs

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    