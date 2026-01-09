"""Question Reasoning Agent - Provides improvement suggestions for questions needing changes."""

import json
import re
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import Config


class QuestionReasoningAgent:
    """
    AI agent that analyzes questions and provides improvement suggestions.
    Only provides reasoning for questions that need changes.
    Perfect questions get no reasoning field.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.2,
            api_key=Config.NVIDIA_API_KEY,
            base_url=Config.NVIDIA_BASE_URL
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a question improvement advisor. Your role is to analyze questions and provide actionable improvement suggestions.

IMPORTANT RULES:
1. ONLY provide reasoning for questions that NEED CHANGES (status: Modify or Remove, or has issues)
2. DO NOT provide reasoning for perfect questions (status: Keep, no issues)
3. Your reasoning should be exactly 2 lines
4. Focus on actionable suggestions, not just descriptions

For questions needing changes, provide specific suggestions:
- If too easy: Suggest adding complexity, examples, or combining with other questions
- If too hard: Suggest breaking down or providing guidance
- If repeated: Suggest how to modify to make it unique
- If has other issues: Suggest specific improvements

Return ONLY valid JSON:
{{
    "reasoning": "<2-line improvement suggestion>" OR null
}}

If the question is perfect, return: {{"reasoning": null}}
"""),
            ("human", """Question: {question_text}
Bloom Level: {bloom_level}
Difficulty: {difficulty}
Topic: {topic}
Status: {status}
Issue Type: {issue_type}
Issue Reason: {issue_reason}

Analyze this question and provide improvement suggestions if needed:""")
        ])
    
    def _clean_json_response(self, response_text: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Try to extract JSON object
        json_match = re.search(r'\{[^{}]*"reasoning"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return response_text
    
    def generate_reasoning(
        self,
        question_text: str,
        bloom_level: str,
        difficulty: str,
        topic: str,
        status: str,
        issue_type: Optional[str] = None,
        issue_reason: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate reasoning for a question if it needs changes.
        Returns None for perfect questions.
        
        Args:
            question_text: The question text
            bloom_level: Bloom's taxonomy level
            difficulty: Difficulty level
            topic: Topic classification
            status: Keep, Modify, or Remove
            issue_type: Type of issue (REPETITION, BLOOM_MISMATCH, etc.) or None
            issue_reason: Reason for the issue or None
            
        Returns:
            Reasoning string if question needs changes, None if perfect
        """
        # If question is perfect (Keep status and no issues), return None
        if status == "Keep" and issue_type is None:
            return None
        
        # Prepare issue information
        issue_type_str = issue_type if issue_type else "None"
        issue_reason_str = issue_reason if issue_reason else "None"
        
        chain = self.prompt | self.llm
        
        try:
            response = chain.invoke({
                "question_text": question_text,
                "bloom_level": bloom_level,
                "difficulty": difficulty,
                "topic": topic,
                "status": status,
                "issue_type": issue_type_str,
                "issue_reason": issue_reason_str
            })
            
            cleaned_response = self._clean_json_response(response.content)
            result = json.loads(cleaned_response)
            
            reasoning = result.get("reasoning")
            
            # If reasoning is null or empty, return None
            if not reasoning or reasoning.lower() == "null":
                return None
            
            return reasoning.strip()
            
        except Exception as e:
            # If there's an error, return None (don't provide reasoning)
            return None
    
    def generate_reasoning_batch(
        self,
        question_results: List[Dict]
    ) -> List[Optional[str]]:
        """
        Generate reasoning for multiple questions in batch.
        Returns list of reasoning strings (or None for perfect questions).
        
        Args:
            question_results: List of question result dicts with:
                - question (or question_text)
                - bloom_level
                - difficulty (or difficulty_level)
                - topic
                - status (or action)
                - issue (dict with type and reason, or None)
        
        Returns:
            List of reasoning strings (or None for perfect questions)
        """
        reasonings = []
        
        for q_result in question_results:
            # Extract question text
            question_text = q_result.get("question") or q_result.get("question_text") or q_result.get("text", "")
            
            # Extract taxonomy
            bloom_level = q_result.get("bloom_level", "Unknown")
            difficulty = q_result.get("difficulty") or q_result.get("difficulty_level", "Unknown")
            topic = q_result.get("topic", "Unknown")
            
            # Extract status and issue
            status = q_result.get("status") or q_result.get("action", "Keep")
            issue = q_result.get("issue")
            
            issue_type = None
            issue_reason = None
            
            if issue and isinstance(issue, dict):
                # Handle both "type" and "issue_type" keys
                issue_type = issue.get("type") or issue.get("issue_type")
                issue_reason = issue.get("reason")
            
            # Generate reasoning
            reasoning = self.generate_reasoning(
                question_text=question_text,
                bloom_level=bloom_level,
                difficulty=difficulty,
                topic=topic,
                status=status,
                issue_type=issue_type,
                issue_reason=issue_reason
            )
            
            reasonings.append(reasoning)
        
        return reasonings

