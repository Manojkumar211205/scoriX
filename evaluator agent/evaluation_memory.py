"""Evaluation Memory Manager - Tracks issues, iterations, and resolution status."""

import json
import os
from typing import Dict, List
from datetime import datetime


class EvaluationMemory:
    """Maintains evaluation memory across iterations."""
    
    def __init__(self, memory_file: str = "data/evaluation_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict:
        """Load evaluation memory from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {
                    "iterations": [],
                    "question_issues": {},
                    "resolved_issues": [],
                    "unresolved_issues": []
                }
        return {
            "iterations": [],
            "question_issues": {},
            "resolved_issues": [],
            "unresolved_issues": []
        }
    
    def _save_memory(self):
        """Save evaluation memory to file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save evaluation memory: {str(e)}")
    
    def record_iteration(self, iteration_num: int, status: str, issues: List[Dict]):
        """Record an evaluation iteration."""
        iteration_record = {
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "issues_count": len(issues),
            "issues": issues
        }
        self.memory["iterations"].append(iteration_record)
        self._save_memory()
    
    def track_question_issue(self, question_id: str, issue: Dict):
        """Track an issue for a specific question."""
        if question_id not in self.memory["question_issues"]:
            self.memory["question_issues"][question_id] = []
        
        issue_record = {
            **issue,
            "timestamp": datetime.now().isoformat(),
            "resolved": False
        }
        self.memory["question_issues"][question_id].append(issue_record)
        self._save_memory()
    
    def mark_issue_resolved(self, question_id: str, issue_type: str):
        """Mark a specific issue as resolved."""
        if question_id in self.memory["question_issues"]:
            for issue in self.memory["question_issues"][question_id]:
                if issue.get("issue_type") == issue_type and not issue.get("resolved", False):
                    issue["resolved"] = True
                    issue["resolved_at"] = datetime.now().isoformat()
                    self._save_memory()
                    return True
        return False
    
    def get_question_issue_history(self, question_id: str) -> List[Dict]:
        """Get issue history for a question."""
        return self.memory["question_issues"].get(question_id, [])
    
    def detect_repeated_flaws(self, question_id: str, current_issue: Dict) -> bool:
        """Detect if the same flaw is repeated for a question."""
        history = self.get_question_issue_history(question_id)
        for past_issue in history:
            if (past_issue.get("issue_type") == current_issue.get("issue_type") and
                past_issue.get("question_id") == current_issue.get("question_id") and
                not past_issue.get("resolved", False)):
                return True
        return False
    
    def get_iteration_count(self) -> int:
        """Get total number of iterations."""
        return len(self.memory["iterations"])
    
    def clear(self):
        """Clear all evaluation memory (for testing)."""
        self.memory = {
            "iterations": [],
            "question_issues": {},
            "resolved_issues": [],
            "unresolved_issues": []
        }
        self._save_memory()






