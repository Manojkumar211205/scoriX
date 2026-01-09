import json
import re
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import Config


class EvaluatorAgent:
    """Sole authority for issue detection and refinement instructions."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,
            api_key=Config.NVIDIA_API_KEY,
            base_url=Config.NVIDIA_BASE_URL
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a question paper validation evaluator with sole authority for issue detection.

Your responsibilities:
1. Detect issues per question (repetition, Bloom mismatch, difficulty imbalance, topic over-concentration, structural flaws)
2. Output structured refinement instructions for flawed questions
3. Determine severity (LOW | MEDIUM | HIGH)
4. Decide evaluation status (REFINE_REQUIRED | ACCEPTABLE_WITH_WARNINGS | VALID)

Issue Types:
- REPETITION: Exact or semantic duplicates
- BLOOM_MISMATCH: Bloom level doesn't match question complexity
- DIFFICULTY_IMBALANCE: Difficulty doesn't match question content
- TOPIC_OVERCONCENTRATION: Too many questions on same topic
- STRUCTURAL_FLAW: Formatting, clarity, or policy violations

Return ONLY valid JSON:
{{
    "status": "<REFINE_REQUIRED | ACCEPTABLE_WITH_WARNINGS | VALID>",
    "issues": [
        {{
            "question_id": "<QID>",
            "issue_type": "<REPETITION | BLOOM_MISMATCH | DIFFICULTY_IMBALANCE | TOPIC_OVERCONCENTRATION | STRUCTURAL_FLAW>",
            "reason": "<detailed explanation>",
            "severity": "<LOW | MEDIUM | HIGH>",
            "required_action": "<Remove | Modify | Rewrite | Keep>",
            "constraints": "<specific constraints for refinement>"
        }}
    ]
}}
"""),
            ("human", """Question Paper Analysis:

Total Questions: {total_questions}

Question Details:
{question_details}

Pattern Analysis:
- Bloom Distribution: {bloom_distribution}
- Difficulty Distribution: {difficulty_distribution}
- Topic Distribution: {topic_distribution}
- Repetition Count: {repetition_count}

Provide structured evaluation:""")
        ])
    
    def _analyze_patterns(self, question_results: List[Dict]) -> Dict:
        """Analyze patterns from question results."""
        total = len(question_results)
        if total == 0:
            return {}
        
        bloom_dist = {}
        diff_dist = {}
        topic_dist = {}
        
        for q in question_results:
            bloom = q.get("bloom_level", "Unknown")
            diff = q.get("difficulty_level", "Unknown")
            topic = q.get("topic", "Unknown")
            
            bloom_dist[bloom] = bloom_dist.get(bloom, 0) + 1
            diff_dist[diff] = diff_dist.get(diff, 0) + 1
            topic_dist[topic] = topic_dist.get(topic, 0) + 1
        
        repetition_count = sum(
            1 for q in question_results 
            if q.get("repetition_type") != "Unique"
        )
        
        return {
            "bloom_distribution": bloom_dist,
            "difficulty_distribution": diff_dist,
            "topic_distribution": topic_dist,
            "repetition_count": repetition_count
        }
    
    def _format_question_details(self, question_results: List[Dict]) -> str:
        """Format question details for LLM prompt."""
        details = []
        for q in question_results:
            details.append(
                f"QID: {q.get('question_id')}\n"
                f"  Bloom: {q.get('bloom_level')}\n"
                f"  Difficulty: {q.get('difficulty_level')}\n"
                f"  Topic: {q.get('topic')}\n"
                f"  Repetition: {q.get('repetition_type')} (similarity: {q.get('max_similarity', 0.0):.2f})\n"
                f"  Action: {q.get('action', 'Keep')}"
            )
        return "\n\n".join(details)
    
    def _clean_json_response(self, response_text: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON object with status and issues
        json_match = re.search(r'\{[^{}]*"status"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            # Try to get full object including nested arrays
            full_match = re.search(r'\{.*?"status".*?"issues".*?\].*?\}', response_text, re.DOTALL)
            if full_match:
                return full_match.group(0)
            return json_match.group(0)
        return response_text
    
    def evaluate(self, question_results: List[Dict], evaluation_memory=None) -> Dict:
        """Generate structured evaluation with per-question issue detection."""
        pattern_analysis = self._analyze_patterns(question_results)
        question_details = self._format_question_details(question_results)
        
        chain = self.prompt | self.llm
        
        try:
            response = chain.invoke({
                "total_questions": len(question_results),
                "question_details": question_details,
                "bloom_distribution": str(pattern_analysis.get("bloom_distribution", {})),
                "difficulty_distribution": str(pattern_analysis.get("difficulty_distribution", {})),
                "topic_distribution": str(pattern_analysis.get("topic_distribution", {})),
                "repetition_count": pattern_analysis.get("repetition_count", 0)
            })
            
            cleaned_response = self._clean_json_response(response.content)
            result = json.loads(cleaned_response)
            
            # Validate and structure output
            status = result.get("status", "REFINE_REQUIRED")
            issues = result.get("issues", [])
            
            # Validate severity and required_action
            for issue in issues:
                if issue.get("severity") not in ["LOW", "MEDIUM", "HIGH"]:
                    issue["severity"] = "MEDIUM"
                if issue.get("required_action") not in ["Remove", "Modify", "Rewrite", "Keep"]:
                    issue["required_action"] = "Modify"
            
            # Determine final status based on issues
            if status == "VALID" and len(issues) == 0:
                return {
                    "status": "VALID",
                    "summary": {
                        "total_questions": len(question_results),
                        "iterations_used": evaluation_memory.get_iteration_count() if evaluation_memory else 0
                    }
                }
            else:
                # Check if only LOW severity issues
                high_medium_issues = [i for i in issues if i.get("severity") in ["HIGH", "MEDIUM"]]
                if len(high_medium_issues) == 0 and len(issues) > 0:
                    status = "ACCEPTABLE_WITH_WARNINGS"
                
                return {
                    "status": status if status != "VALID" else "REFINE_REQUIRED",
                    "issues": issues
                }
                
        except Exception as e:
            # Fallback: Deterministic issue detection
            issues = []
            for q in question_results:
                qid = q.get("question_id")
                
                # Repetition detection
                if q.get("repetition_type") != "Unique":
                    similarity = q.get("max_similarity", 0.0)
                    if similarity >= 0.90:
                        issues.append({
                            "question_id": qid,
                            "issue_type": "REPETITION",
                            "reason": f"Exact or near-exact repetition detected (similarity: {similarity:.2f})",
                            "severity": "HIGH",
                            "required_action": "Remove",
                            "constraints": "Remove duplicate question"
                        })
                    elif similarity >= 0.70:
                        issues.append({
                            "question_id": qid,
                            "issue_type": "REPETITION",
                            "reason": f"Semantic repetition detected (similarity: {similarity:.2f})",
                            "severity": "MEDIUM",
                            "required_action": "Modify",
                            "constraints": "Reword to reduce similarity while maintaining topic coverage"
                        })
            
            # Pattern-based issues
            bloom_dist = pattern_analysis.get("bloom_distribution", {})
            if len(bloom_dist) == 1:
                issues.append({
                    "question_id": "PAPER_LEVEL",
                    "issue_type": "BLOOM_MISMATCH",
                    "reason": f"All questions are at {list(bloom_dist.keys())[0]} level - no cognitive diversity",
                    "severity": "HIGH",
                    "required_action": "Modify",
                    "constraints": "Add questions across multiple Bloom levels"
                })
            
            topic_dist = pattern_analysis.get("topic_distribution", {})
            if len(topic_dist) == 1:
                issues.append({
                    "question_id": "PAPER_LEVEL",
                    "issue_type": "TOPIC_OVERCONCENTRATION",
                    "reason": f"All questions focus on {list(topic_dist.keys())[0]} - insufficient topic diversity",
                    "severity": "MEDIUM",
                    "required_action": "Modify",
                    "constraints": "Add questions from different topics"
                })
            
            if len(issues) == 0:
                return {
                    "status": "VALID",
                    "summary": {
                        "total_questions": len(question_results),
                        "iterations_used": evaluation_memory.get_iteration_count() if evaluation_memory else 0
                    }
                }
            else:
                high_medium_count = sum(1 for i in issues if i.get("severity") in ["HIGH", "MEDIUM"])
                status = "REFINE_REQUIRED" if high_medium_count > 0 else "ACCEPTABLE_WITH_WARNINGS"
                
                return {
                    "status": status,
                    "issues": issues
                }

