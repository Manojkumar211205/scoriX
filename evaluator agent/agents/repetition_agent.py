from typing import Dict, List
from config import Config
import numpy as np

class RepetitionAgent:
    """
    Deterministic repetition detection agent.
    Uses similarity scores from memory and embeddings, applies thresholds.
    NO LLM calls.
    """
    
    def __init__(self):
        self.exact_threshold = 0.90
        self.semantic_threshold = 0.80
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Vectors should already be normalized, but ensure it
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            dot_product = np.dot(vec1, vec2)
            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0
    
    def _is_zero_embedding(self, embedding: np.ndarray) -> bool:
        """Check if embedding is all zeros (failed embedding)."""
        return np.allclose(embedding, 0.0)
    
    def detect_repetition_batch(
        self, 
        questions: List[Dict], 
        similarity_scores: Dict[str, List[tuple]],
        question_embeddings: Dict[str, np.ndarray] = None
    ) -> List[Dict]:
        """
        Detect repetitions using similarity scores and embeddings.
        
        Args:
            questions: List of question dicts with qid and text
            similarity_scores: {question_id -> [(similar_qid, score), ...]} from memory
            question_embeddings: {question_id -> embedding_vector} for within-batch comparison
        
        Returns:
            List of repetition detection results
        """
        results = []
        
        # Check if embeddings are valid (not all zeros)
        use_embeddings = question_embeddings is not None and len(question_embeddings) > 0
        if use_embeddings:
            # Check if any embedding is non-zero
            first_emb = next(iter(question_embeddings.values()))
            use_embeddings = not self._is_zero_embedding(first_emb)
        
        for i, q in enumerate(questions):
            qid = q.get("qid")
            similarities = similarity_scores.get(qid, [])
            
            # Get max similarity from historical memory
            max_similarity = 0.0
            similar_to = None
            
            if similarities:
                similar_to, max_similarity = similarities[0]  # Top result
            
            # Check within current batch (compare with previous questions)
            for j in range(i):
                prev_q = questions[j]
                prev_qid = prev_q.get("qid")
                
                if use_embeddings and qid in question_embeddings and prev_qid in question_embeddings:
                    # Use embedding-based similarity
                    emb1 = question_embeddings[qid]
                    emb2 = question_embeddings[prev_qid]
                    
                    if not self._is_zero_embedding(emb1) and not self._is_zero_embedding(emb2):
                        similarity = self._cosine_similarity(emb1, emb2)
                    else:
                        similarity = 0.0
                else:
                    similarity = 0.0
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_to = prev_qid
            
            # Apply deterministic thresholds
            if max_similarity >= self.exact_threshold:
                repetition_type = "Exact Repetition"
                action = "Remove"
            elif max_similarity >= self.semantic_threshold:
                repetition_type = "Semantic Repetition"
                action = "Modify"
            else:
                repetition_type = "Unique"
                action = "Keep"
            
            results.append({
                "question_id": qid,
                "max_similarity": round(max_similarity, 3),
                "repetition_type": repetition_type,
                "action": action,
                "similar_to": similar_to
            })
        
        return results
