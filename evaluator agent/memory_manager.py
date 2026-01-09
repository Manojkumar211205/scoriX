from typing import Dict, List, Tuple
import numpy as np
import os
import json
import faiss

class MemoryManager:
    """FAISS-based vector store for question memory."""
    
    def __init__(self, memory_file: str = "data/question_memory.json", index_file: str = "data/question_index.faiss"):
        self.memory_file = memory_file
        self.index_file = index_file
        self.questions = self._load_memory()
        self.dimension = None  # Will be set from first embedding
        self.index = None
    
    def _load_memory(self) -> List[Dict]:
        """Load questions from JSON file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_memory(self):
        """Save questions to JSON file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.questions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save memory: {str(e)}")
    
    def _get_or_create_index(self, dimension: int):
        """Get or create FAISS index with specified dimension."""
        if self.dimension is None:
            self.dimension = dimension
        
        if self.dimension != dimension:
            # Dimension mismatch - recreate index
            self.index = None
            self.dimension = dimension
        
        if self.index is None:
            if os.path.exists(self.index_file) and len(self.questions) > 0:
                try:
                    index = faiss.read_index(self.index_file)
                    if index.d == dimension:
                        self.index = index
                        return self.index
                except:
                    pass
            
            # Create new index (L2 distance for cosine similarity)
            self.index = faiss.IndexFlatL2(dimension)
        
        return self.index
    
    def _save_index(self):
        """Save FAISS index to file."""
        try:
            faiss.write_index(self.index, self.index_file)
        except Exception as e:
            print(f"Warning: Could not save index: {str(e)}")
    
    def batch_similarity_search(self, query_embeddings: Dict[str, np.ndarray], k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch similarity search for all query embeddings.
        Returns: {question_id -> [(similar_qid, similarity_score), ...]}
        """
        if len(self.questions) == 0:
            return {qid: [] for qid in query_embeddings.keys()}
        
        # Get dimension from first embedding
        if not query_embeddings:
            return {}
        
        first_emb = next(iter(query_embeddings.values()))
        dimension = len(first_emb)
        
        # Get or create index
        index = self._get_or_create_index(dimension)
        
        if index.ntotal == 0:
            return {qid: [] for qid in query_embeddings.keys()}
        
        results = {}
        
        for qid, query_emb in query_embeddings.items():
            # Normalize for cosine similarity
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            query_norm = query_norm.reshape(1, -1).astype('float32')
            
            # Search
            distances, indices = index.search(query_norm, min(k, index.ntotal))
            
            # Convert distances to similarities (cosine similarity)
            similarities = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.questions):
                    # For L2 distance on normalized vectors: similarity ≈ 1 - (distance^2 / 2)
                    similarity = max(0.0, 1.0 - (dist / 2.0))
                    similar_qid = self.questions[idx].get("qid", "Unknown")
                    similarities.append((similar_qid, float(similarity)))
            
            results[qid] = similarities
        
        return results
    
    def add_questions_batch(self, questions: List[Dict], embeddings: Dict[str, np.ndarray]):
        """
        Add questions with embeddings to memory.
        Only stores unique questions (based on similarity threshold).
        """
        if not questions or not embeddings:
            return
        
        # Get dimension from first embedding
        first_emb = next(iter(embeddings.values()))
        dimension = len(first_emb)
        
        # Get or create index
        index = self._get_or_create_index(dimension)
        
        new_questions = []
        new_embeddings = []
        
        for q in questions:
            qid = q.get("qid")
            if qid not in embeddings:
                continue
            
            # Check if question already exists
            exists = any(stored_q.get("qid") == qid for stored_q in self.questions)
            if not exists:
                question_data = {
                    "qid": qid,
                    "text": q.get("text", ""),
                    "bloom_level": q.get("bloom_level", "Unknown"),
                    "difficulty": q.get("difficulty", "Unknown"),
                    "topic": q.get("topic", "Unknown"),
                    "subtopic": q.get("subtopic", "Unknown"),
                    "verdict": q.get("verdict", "Unknown")
                }
                new_questions.append(question_data)
                
                # Normalize embedding for cosine similarity
                emb = embeddings[qid]
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                new_embeddings.append(emb_norm.astype('float32'))
        
        if new_questions and new_embeddings:
            # Add to questions list
            self.questions.extend(new_questions)
            
            # Add to FAISS index
            if len(new_embeddings) > 0:
                embeddings_array = np.vstack(new_embeddings)
                index.add(embeddings_array)
                self.index = index  # Update reference
            
            # Save
            self._save_memory()
            self._save_index()
    
    def get_all_questions(self) -> List[Dict]:
        """Get all stored questions."""
        return self.questions
