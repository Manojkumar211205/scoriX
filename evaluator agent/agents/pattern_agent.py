from typing import Dict, List
from collections import Counter


class PatternAgent:
    def __init__(self):
        pass
    
    def analyze_patterns(self, question_inferences: List[Dict]) -> Dict:
        """Analyze patterns across all questions."""
        
        # Extract all inferences
        bloom_levels = [q.get("bloom_level", "Unknown") for q in question_inferences]
        difficulties = [q.get("difficulty", "Medium") for q in question_inferences]
        topics = [q.get("topic", "Unknown") for q in question_inferences]
        repetition_flags = [q.get("flag", "Unique") for q in question_inferences]
        
        # Bloom distribution
        bloom_dist = Counter(bloom_levels)
        bloom_balance = self._assess_bloom_balance(bloom_dist, len(question_inferences))
        
        # Difficulty distribution
        difficulty_dist = Counter(difficulties)
        difficulty_balance = self._assess_difficulty_balance(difficulty_dist, len(question_inferences))
        
        # Topic coverage
        unique_topics = len(set(topics))
        topic_coverage = self._assess_topic_coverage(unique_topics, len(question_inferences))
        
        # Repetition count
        repetition_count = sum(1 for flag in repetition_flags if flag == "Repeated")
        
        return {
            "bloom_balance": bloom_balance,
            "difficulty_balance": difficulty_balance,
            "topic_coverage": topic_coverage,
            "repetition_count": repetition_count,
            "bloom_distribution": dict(bloom_dist),
            "difficulty_distribution": dict(difficulty_dist),
            "topic_distribution": dict(Counter(topics))
        }
    
    def _assess_bloom_balance(self, bloom_dist: Counter, total: int) -> str:
        """Assess if Bloom levels are balanced."""
        if total == 0:
            return "Insufficient Data"
        
        # Ideal: distributed across levels, not all in one level
        max_count = max(bloom_dist.values()) if bloom_dist else 0
        max_ratio = max_count / total
        
        if max_ratio > 0.6:  # More than 60% in one level
            return "Skewed"
        elif max_ratio > 0.4:  # 40-60% in one level
            return "Moderately Balanced"
        else:
            return "Well Balanced"
    
    def _assess_difficulty_balance(self, difficulty_dist: Counter, total: int) -> str:
        """Assess if difficulty levels are balanced."""
        if total == 0:
            return "Insufficient Data"
        
        easy = difficulty_dist.get("Easy", 0)
        medium = difficulty_dist.get("Medium", 0)
        hard = difficulty_dist.get("Hard", 0)
        
        # Ideal: mix of all levels, not all easy or all hard
        if easy == total:
            return "All Easy"
        elif hard == total:
            return "All Hard"
        elif medium == total:
            return "All Medium"
        elif easy > total * 0.5:
            return "Too Easy"
        elif hard > total * 0.5:
            return "Too Hard"
        else:
            return "Well Balanced"
    
    def _assess_topic_coverage(self, unique_topics: int, total: int) -> str:
        """Assess topic coverage."""
        if total == 0:
            return "Insufficient Data"
        
        coverage_ratio = unique_topics / total
        
        if coverage_ratio < 0.3:  # Less than 30% unique topics
            return "Over-Concentrated"
        elif coverage_ratio < 0.5:
            return "Moderate Coverage"
        else:
            return "Good Coverage"










