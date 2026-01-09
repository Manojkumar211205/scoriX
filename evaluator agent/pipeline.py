import json
import re
import concurrent.futures
from typing import Dict, List, Tuple
from agents import (
    SegmentationAgent,
    EmbeddingAgent,
    BloomAgent,
    DifficultyAgent,
    TopicAgent,
    RepetitionAgent,
    EvaluatorAgent,
    QuestionReasoningAgent
)
from memory_manager import MemoryManager
from evaluation_memory import EvaluationMemory


class QuestionPaperValidationPipeline:
    """High-performance Council of Task-Specialized Agents with Iterative Refinement Support."""
    
    def __init__(self, max_iterations: int = 5, enable_refinement_loop: bool = False):
        self.segmentation_agent = SegmentationAgent()
        self.embedding_agent = EmbeddingAgent()
        self.bloom_agent = BloomAgent()
        self.difficulty_agent = DifficultyAgent()
        self.topic_agent = TopicAgent()
        self.repetition_agent = RepetitionAgent()
        self.evaluator_agent = EvaluatorAgent()
        self.reasoning_agent = QuestionReasoningAgent()
        self.memory_manager = MemoryManager()
        self.evaluation_memory = EvaluationMemory()
        self.max_iterations = max_iterations
        self.enable_refinement_loop = enable_refinement_loop  # Set to True when refinement agent is added
    
    def _process_single_iteration(self, questions: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Process a single validation iteration. Returns (question_results, evaluation)."""
        
        # Step 1: Batch Embedding
        question_embeddings = self.embedding_agent.embed_batch(questions)
        
        # Step 2: Memory (Vector Store) - Batch Similarity Search
        similarity_scores = self.memory_manager.batch_similarity_search(
            question_embeddings, 
            k=5
        )
        
        # Step 3: Repetition Agent (Deterministic, No LLM)
        repetition_results = self.repetition_agent.detect_repetition_batch(
            questions, 
            similarity_scores,
            question_embeddings
        )
        
        # Step 4: LLM Agents (Batched) - RUN IN PARALLEL
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            bloom_future = executor.submit(self.bloom_agent.infer_bloom_batch, questions)
            difficulty_future = executor.submit(self.difficulty_agent.infer_difficulty_batch, questions)
            topic_future = executor.submit(self.topic_agent.infer_topic_batch, questions)
            
            bloom_results = bloom_future.result()
            difficulty_results = difficulty_future.result()
            topic_results = topic_future.result()
        
        # Step 5: Combine Results
        question_results = []
        for i, q in enumerate(questions):
            qid = q.get("qid")
            
            rep_result = next(
                (r for r in repetition_results if r.get("question_id") == qid),
                {}
            )
            
            question_results.append({
                "question_id": qid,
                "bloom_level": bloom_results[i].get("bloom_level", "Unknown"),
                "difficulty_level": difficulty_results[i].get("difficulty", "Unknown"),
                "topic": topic_results[i].get("topic", "Unknown"),
                "max_similarity": rep_result.get("max_similarity", 0.0),
                "repetition_type": rep_result.get("repetition_type", "Unique"),
                "action": rep_result.get("action", "Keep")
            })
        
        # Step 6: Evaluator Agent (Structured Evaluation)
        evaluation = self.evaluator_agent.evaluate(question_results, self.evaluation_memory)
        
        return question_results, evaluation
    
    def process(self, input_text: str) -> Dict:
        """Execute the complete validation pipeline - single iteration (ready for loop when refinement agent is added)."""
        
        # Step 0: Segmentation
        segmentation_result = self.segmentation_agent.segment_questions(input_text)
        questions = segmentation_result["questions"]
        
        if not questions:
            return {"error": "No questions found in input"}
        
        # Single iteration (no loop until refinement agent is added)
        iteration = 1
        question_results, evaluation = self._process_single_iteration(questions)
        
        # Create question map early (needed for both VALID and REFINE_REQUIRED paths)
        question_map = {q.get("qid"): q.get("text", "") for q in questions}
        
        # Record in evaluation memory
        issues = evaluation.get("issues", [])
        status = evaluation.get("status", "REFINE_REQUIRED")
        
        self.evaluation_memory.record_iteration(iteration, status, issues)
        
        # Track per-question issues
        for issue in issues:
            qid = issue.get("question_id")
            if qid and qid != "PAPER_LEVEL":
                self.evaluation_memory.track_question_issue(qid, issue)
        
        # Map issues to questions for reasoning generation
        issue_map = {}
        for issue in issues:
            qid = issue.get("question_id")
            if qid and qid != "PAPER_LEVEL":
                issue_map[qid] = issue
        
        # Prepare question data for reasoning agent
        questions_for_reasoning = []
        
        for q_result in question_results:
            qid = q_result.get("question_id")
            question_text = question_map.get(qid, "")
            
            # Get issue if exists
            issue = issue_map.get(qid)
            
            # Determine status
            if issue:
                status_for_reasoning = issue.get("required_action", "Modify")
                if status_for_reasoning == "Remove":
                    status_for_reasoning = "Remove"
                else:
                    status_for_reasoning = "Modify"
            else:
                status_for_reasoning = "Keep"
            
            questions_for_reasoning.append({
                "question": question_text,
                "question_text": question_text,
                "bloom_level": q_result.get("bloom_level", "Unknown"),
                "difficulty": q_result.get("difficulty_level", "Unknown"),
                "topic": q_result.get("topic", "Unknown"),
                "status": status_for_reasoning,
                "issue": issue
            })
        
        # Generate reasoning for all questions
        reasonings = self.reasoning_agent.generate_reasoning_batch(questions_for_reasoning)
        
        # Store unique questions in memory
        unique_questions = [
            q for i, q in enumerate(questions)
            if question_results[i].get("repetition_type") == "Unique"
        ]
        
        if unique_questions:
            unique_with_metadata = []
            for q in unique_questions:
                qid = q.get("qid")
                q_result = next((r for r in question_results if r.get("question_id") == qid), {})
                unique_with_metadata.append({
                    "qid": qid,
                    "text": q.get("text", ""),
                    "bloom_level": q_result.get("bloom_level", "Unknown"),
                    "difficulty": q_result.get("difficulty_level", "Unknown"),
                    "topic": q_result.get("topic", "Unknown"),
                    "subtopic": "Unknown",
                    "verdict": status
                })
            
            unique_embeddings = {
                q["qid"]: self.embedding_agent.embed_batch([q])[q["qid"]]
                for q in unique_questions
            }
            self.memory_manager.add_questions_batch(unique_with_metadata, unique_embeddings)
        
        # Format output based on status - return only paper_verdict
        # Calculate summary statistics for paper_verdict (needed for both VALID and REFINE_REQUIRED)
        bloom_dist = {}
        diff_dist = {}
        topic_dist = {}
        
        for q_result in question_results:
            bloom = q_result.get("bloom_level", "Unknown")
            diff = q_result.get("difficulty_level", "Unknown")
            topic = q_result.get("topic", "Unknown")
            
            bloom_dist[bloom] = bloom_dist.get(bloom, 0) + 1
            diff_dist[diff] = diff_dist.get(diff, 0) + 1
            topic_dist[topic] = topic_dist.get(topic, 0) + 1
        
        repetition_count = sum(
            1 for q in question_results 
            if q.get("repetition_type") != "Unique"
        )
        
        if status == "VALID":
            # For VALID status, build questions list with no issues
            all_questions = []
            for i, q_result in enumerate(question_results):
                qid = q_result.get("question_id")
                question_text = question_map.get(qid, "")
                
                # Get reasoning (should be None for perfect questions)
                reasoning = reasonings[i] if i < len(reasonings) else None
                
                question_obj = {
                    "question": question_text,
                    "bloom_level": q_result.get("bloom_level", "Unknown"),
                    "difficulty": q_result.get("difficulty_level", "Unknown"),
                    "topic": q_result.get("topic", "Unknown"),
                    "status": "Keep",
                    "issue": None
                }
                
                # Only add reasoning if it exists (shouldn't for perfect questions)
                if reasoning:
                    question_obj["reasoning"] = reasoning
                
                all_questions.append(question_obj)
            
            return {
                "summary": {
                    "verdict": "Valid",
                    "summary_reason": "Paper is valid",
                    "issues": [],
                    "questions": all_questions,
                    "questions_to_change": [],
                    "iterations_used": iteration
                }
            }
        else:
            # Map question_id to question text for refinement agent
            question_map = {q.get("qid"): q.get("text", "") for q in questions}
            
            # Replace question_id with question text in issues
            issues_with_text = []
            paper_level_issues = []
            for issue in issues:
                issue_copy = issue.copy()
                qid = issue_copy.pop("question_id", None)  # Remove question_id
                
                if qid == "PAPER_LEVEL":
                    issue_copy["question"] = None
                    paper_level_issues.append(issue_copy)
                else:
                    issue_copy["question"] = question_map.get(qid, "")
                    issues_with_text.append(issue_copy)
            
            # Generate summary issues for paper_verdict with readable explanations
            summary_issues = []
            
            # Check repetition (most common issue)
            if repetition_count > 0:
                if repetition_count == 1:
                    summary_issues.append(f"1 repeated question detected - some questions are duplicated")
                else:
                    summary_issues.append(f"{repetition_count} repeated questions detected - some questions are duplicated")
            
            # Check difficulty distribution
            total_questions = len(question_results)
            easy_count = diff_dist.get("Easy", 0)
            medium_count = diff_dist.get("Medium", 0)
            hard_count = diff_dist.get("Hard", 0)
            
            if len(diff_dist) == 1:
                only_level = list(diff_dist.keys())[0]
                if only_level == "Easy":
                    summary_issues.append("Paper is too easy - all questions are at Easy difficulty level")
                elif only_level == "Hard":
                    summary_issues.append("Paper is too hard - all questions are at Hard difficulty level")
                else:
                    summary_issues.append("Difficulty is not balanced - all questions are at Medium level")
            elif easy_count > total_questions * 0.7:
                summary_issues.append("Paper is too easy - more than 70% questions are Easy")
            elif hard_count > total_questions * 0.7:
                summary_issues.append("Paper is too hard - more than 70% questions are Hard")
            elif easy_count == 0:
                summary_issues.append("Paper lacks easy questions - no Easy difficulty questions found")
            elif hard_count == 0:
                summary_issues.append("Paper lacks challenging questions - no Hard difficulty questions found")
            
            # Check Bloom distribution
            if len(bloom_dist) == 1:
                only_level = list(bloom_dist.keys())[0]
                summary_issues.append(f"All questions test the same cognitive level ({only_level}) - needs more variety")
            elif len(bloom_dist) < 3:
                summary_issues.append("Limited cognitive diversity - questions cover fewer than 3 Bloom's taxonomy levels")
            
            # Check topic concentration
            if len(topic_dist) == 1:
                only_topic = list(topic_dist.keys())[0]
                summary_issues.append(f"All questions focus on one topic ({only_topic}) - needs broader coverage")
            elif len(topic_dist) < 3:
                summary_issues.append("Limited topic coverage - questions cover fewer than 3 different topics")
            
            # Add paper-level issues from evaluator
            for paper_issue in paper_level_issues:
                issue_type = paper_issue.get("issue_type", "")
                reason = paper_issue.get("reason", "")
                if issue_type == "BLOOM_MISMATCH":
                    summary_issues.append(f"Bloom taxonomy issue: {reason}")
                elif issue_type == "TOPIC_OVERCONCENTRATION":
                    summary_issues.append(f"Topic concentration issue: {reason}")
                elif issue_type == "DIFFICULTY_IMBALANCE":
                    summary_issues.append(f"Difficulty imbalance: {reason}")
            
            # Determine verdict
            verdict = "Valid" if status == "VALID" else "Invalid"
            
            # Generate overall summary reason
            summary_reason_parts = []
            if repetition_count > 0:
                summary_reason_parts.append(f"{repetition_count} question(s) are repeated")
            if len(diff_dist) == 1:
                only_level = list(diff_dist.keys())[0]
                if only_level == "Easy":
                    summary_reason_parts.append("paper is too easy")
                elif only_level == "Hard":
                    summary_reason_parts.append("paper is too hard")
            elif easy_count > total_questions * 0.7:
                summary_reason_parts.append("paper is too easy")
            elif hard_count > total_questions * 0.7:
                summary_reason_parts.append("paper is too hard")
            if len(bloom_dist) < 3:
                summary_reason_parts.append("limited cognitive diversity")
            if len(topic_dist) < 3:
                summary_reason_parts.append("limited topic coverage")
            
            summary_reason = "Paper needs changes because: " + ", ".join(summary_reason_parts) if summary_reason_parts else "Paper is valid"
            
            # Extract questions to change (only questions with issues, not PAPER_LEVEL)
            questions_to_change = []
            for issue in issues_with_text:
                issue_type = issue.get("issue_type", "")
                reason = issue.get("reason", "")
                
                # Make reason more readable
                readable_reason = reason
                if issue_type == "REPETITION":
                    # Extract similarity score from reason (format: "similarity: 0.84" or "similarity: 0.84)")
                    similarity_match = re.search(r'similarity:\s*([\d.]+)', reason, re.IGNORECASE)
                    similarity_score = similarity_match.group(1) if similarity_match else "high"
                    
                    if "Exact" in reason or "near-exact" in reason or float(similarity_score) >= 0.90:
                        readable_reason = f"This question is almost identical to another question (similarity: {similarity_score})"
                    elif "Semantic" in reason or float(similarity_score) >= 0.70:
                        readable_reason = f"This question is very similar to another question (similarity: {similarity_score})"
                elif issue_type == "BLOOM_MISMATCH":
                    readable_reason = f"Bloom level doesn't match question complexity: {reason}"
                elif issue_type == "DIFFICULTY_IMBALANCE":
                    readable_reason = f"Difficulty level doesn't match question content: {reason}"
                elif issue_type == "TOPIC_OVERCONCENTRATION":
                    readable_reason = f"Too many questions on this topic: {reason}"
                
                questions_to_change.append({
                    "question": issue.get("question", ""),
                    "reason": readable_reason,
                    "action": issue.get("required_action", "Modify"),
                    "issue_type": issue.get("issue_type", ""),
                    "severity": issue.get("severity", "MEDIUM")
                })
            
            # Build complete questions list with reasoning
            all_questions = []
            for i, q_result in enumerate(question_results):
                qid = q_result.get("question_id")
                question_text = question_map.get(qid, "")
                
                # Get issue if exists
                issue = issue_map.get(qid)
                
                # Determine status
                if issue:
                    question_status = issue.get("required_action", "Modify")
                    if question_status == "Remove":
                        question_status = "Remove"
                    else:
                        question_status = "Modify"
                else:
                    question_status = "Keep"
                
                # Get reasoning (only if not None)
                reasoning = reasonings[i] if i < len(reasonings) else None
                
                # Build issue object for output
                issue_obj = None
                if issue:
                    issue_type = issue.get("issue_type", "")
                    reason = issue.get("reason", "")
                    
                    # Make reason more readable
                    readable_reason = reason
                    if issue_type == "REPETITION":
                        similarity_match = re.search(r'similarity:\s*([\d.]+)', reason, re.IGNORECASE)
                        similarity_score = similarity_match.group(1) if similarity_match else "high"
                        
                        if "Exact" in reason or "near-exact" in reason or (similarity_match and float(similarity_score) >= 0.90):
                            readable_reason = f"This question is almost identical to another question (similarity: {similarity_score})"
                        elif "Semantic" in reason or (similarity_match and float(similarity_score) >= 0.70):
                            readable_reason = f"This question is very similar to another question (similarity: {similarity_score})"
                    
                    issue_obj = {
                        "type": issue_type,
                        "reason": readable_reason
                    }
                
                question_obj = {
                    "question": question_text,
                    "bloom_level": q_result.get("bloom_level", "Unknown"),
                    "difficulty": q_result.get("difficulty_level", "Unknown"),
                    "topic": q_result.get("topic", "Unknown"),
                    "status": question_status,
                    "issue": issue_obj
                }
                
                # Only add reasoning if it exists (not None)
                if reasoning:
                    question_obj["reasoning"] = reasoning
                
                all_questions.append(question_obj)
            
            return {
                "summary": {
                    "verdict": verdict,
                    "summary_reason": summary_reason,
                    "issues": summary_issues,
                    "questions": all_questions,
                    "questions_to_change": questions_to_change,
                    "iterations_used": iteration
                }
            }


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py '<question_paper_text>'")
        sys.exit(1)
    
    input_text = sys.argv[1]
    pipeline = QuestionPaperValidationPipeline()
    result = pipeline.process(input_text)
    
    # Output strict JSON only
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
