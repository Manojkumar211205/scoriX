from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingAgent:
    """Batch embeds all questions using HuggingFace SentenceTransformers."""
    
    def __init__(self):
        # Use BAAI/bge-small-en-v1.5 for embeddings (local, fast, batch support)
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.dimension = 384  # bge-small-en-v1.5 dimension
    
    def embed_batch(self, questions: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Batch embed all questions.
        Returns: {question_id -> embedding_vector}
        """
        question_texts = [q["text"] for q in questions]
        question_ids = [q["qid"] for q in questions]
        
        try:
            # Batch embed all questions in one call (very fast!)
            embeddings = self.model.encode(
                question_texts, 
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to numpy arrays and create mapping
            result = {}
            for qid, embedding in zip(question_ids, embeddings):
                result[qid] = np.array(embedding, dtype=np.float32)
            
            return result
        except Exception as e:
            # Fallback: return zero embeddings
            print(f"Warning: Embedding failed: {str(e)}")
            return {qid: np.zeros(self.dimension, dtype=np.float32) for qid in question_ids}

