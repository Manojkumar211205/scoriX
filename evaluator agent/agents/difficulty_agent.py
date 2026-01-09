import json
import re
import logging
from typing import Dict, Optional, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import Config

# Setup logging
logger = logging.getLogger(__name__)


class DifficultyAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.3,
            api_key=Config.NVIDIA_API_KEY,
            base_url=Config.NVIDIA_BASE_URL
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a difficulty assessment expert.
            Analyze the question and estimate difficulty:
            - Easy: Straightforward, basic concepts, direct recall
            - Medium: Requires understanding and some application
            - Hard: Complex reasoning, analysis, synthesis required
            
            Consider: question complexity, marks allocated (if mentioned), cognitive demand.
            
            Return ONLY valid JSON (no markdown, no code blocks):
            {{"difficulty": "<Easy/Medium/Hard>", "confidence": <0-1>}}
            """),
            ("human", "Question: {question_text}\nMarks: {marks}\n\nAssess difficulty:")
        ])
    
    def extract_marks(self, question_text: str) -> Optional[int]:
        """Extract marks from question text if mentioned."""
        patterns = [
            r'\((\d+)\s*marks?\)',
            r'\[(\d+)\s*marks?\]',
            r'(\d+)\s*marks?',
        ]
        for pattern in patterns:
            match = re.search(pattern, question_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None
    
    def _clean_json_response(self, response_text: str) -> str:
        """Extract JSON from LLM response, removing markdown code blocks if present."""
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Try to find JSON array first (expected format for batch)
        array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if array_match:
            return array_match.group(0)
        
        # Try to find JSON object (fallback for single responses)
        object_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if object_match:
            return object_match.group(0)
        
        return response_text
    
    def _validate_difficulty(self, difficulty: str) -> str:
        """Validate and normalize difficulty value."""
        difficulty_lower = difficulty.lower().strip()
        difficulty_map = {
            'easy': 'Easy',
            'medium': 'Medium',
            'hard': 'Hard'
        }
        return difficulty_map.get(difficulty_lower, 'Medium')  # Default to Medium if invalid
    
    def infer_difficulty(self, qid: str, question_text: str) -> Dict:
        """Infer difficulty level for a question."""
        marks = self.extract_marks(question_text)
        marks_str = str(marks) if marks else "Not specified"
        
        chain = self.prompt | self.llm
        
        try:
            response = chain.invoke({
                "question_text": question_text,
                "marks": marks_str
            })
            
            cleaned_response = self._clean_json_response(response.content)
            result = json.loads(cleaned_response)
            
            difficulty = self._validate_difficulty(result.get("difficulty", "Medium"))
            
            return {
                "qid": qid,
                "difficulty": difficulty,
                "confidence": float(result.get("confidence", 0.5))
            }
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for QID {qid}: {str(e)}")
            return {
                "qid": qid,
                "difficulty": "Medium",
                "confidence": 0.0
            }
        except Exception as e:
            logger.error(f"Error inferring difficulty for QID {qid}: {str(e)}")
            return {
                "qid": qid,
                "difficulty": "Medium",
                "confidence": 0.0
            }
    
    def infer_difficulty_batch(self, questions: List[Dict]) -> List[Dict]:
        """Infer difficulty for multiple questions in one API call."""
        # Extract marks for all questions
        questions_with_marks = []
        for q in questions:
            marks = self.extract_marks(q["text"])
            marks_str = str(marks) if marks else "Not specified"
            questions_with_marks.append({
                "qid": q["qid"],
                "text": q["text"],
                "marks": marks_str
            })
        
        # Build batch prompt
        questions_text = "\n\n".join([
            f"Question {i+1} (QID: {q['qid']}): {q['text']}\nMarks: {q['marks']}"
            for i, q in enumerate(questions_with_marks)
        ])
        
        batch_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a difficulty assessment expert.
            Analyze each question and estimate difficulty:
            - Easy: Straightforward, basic concepts, direct recall
            - Medium: Requires understanding and some application
            - Hard: Complex reasoning, analysis, synthesis required
            
            Return ONLY valid JSON array (no markdown, no code blocks):
            [
                {{"qid": "Q1", "difficulty": "<Easy/Medium/Hard>", "confidence": <0-1>}},
                {{"qid": "Q2", "difficulty": "<Easy/Medium/Hard>", "confidence": <0-1>}},
                ...
            ]"""),
            ("human", "Questions:\n{questions_text}\n\nAssess difficulty for all questions:")
        ])
        
        chain = batch_prompt | self.llm
        
        try:
            response = chain.invoke({"questions_text": questions_text})
            cleaned_response = self._clean_json_response(response.content)
            results = json.loads(cleaned_response)
            
            # Ensure results is a list
            if not isinstance(results, list):
                logger.warning(f"Batch response is not a list, attempting to convert. Response type: {type(results)}")
                logger.debug(f"Raw response: {response.content[:500]}")  # Log first 500 chars
                if isinstance(results, dict):
                    # If single dict, check if it has qid - might be single result
                    if "qid" in results:
                        results = [results]
                    else:
                        # Try to extract array from response if it exists
                        raise ValueError("Invalid response format: single dict without qid")
                else:
                    raise ValueError(f"Invalid response format: {type(results)}")
            
            # Ensure results match questions order
            result_dict = {r.get("qid"): r for r in results}
            batch_results = []
            for q in questions:
                qid = q["qid"]
                if qid in result_dict:
                    raw_difficulty = result_dict[qid].get("difficulty", "Medium")
                    validated_difficulty = self._validate_difficulty(raw_difficulty)
                    batch_results.append({
                        "qid": qid,
                        "difficulty": validated_difficulty,
                        "confidence": float(result_dict[qid].get("confidence", 0.5))
                    })
                else:
                    logger.warning(f"Missing result for QID {qid} in batch response")
                    batch_results.append({
                        "qid": qid,
                        "difficulty": "Medium",
                        "confidence": 0.0
                    })
            return batch_results
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed in batch: {str(e)}")
            return [
                {"qid": q["qid"], "difficulty": "Medium", "confidence": 0.0}
                for q in questions
            ]
        except Exception as e:
            logger.error(f"Error in batch difficulty inference: {str(e)}")
            return [
                {"qid": q["qid"], "difficulty": "Medium", "confidence": 0.0}
                for q in questions
            ]
