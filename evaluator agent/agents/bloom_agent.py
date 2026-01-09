import json
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import Config


class BloomInference(BaseModel):
    qid: str
    bloom_level: str
    confidence: float


class BloomAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.3,
            api_key=Config.NVIDIA_API_KEY,
            base_url=Config.NVIDIA_BASE_URL
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Bloom's Taxonomy classification expert.
            Analyze the question and determine its cognitive level:
            - Remember: Recall facts, definitions
            - Understand: Explain concepts, interpret
            - Apply: Use knowledge in new situations
            - Analyze: Break down, compare, contrast
            - Evaluate: Judge, critique, justify
            - Create: Design, construct, produce
            
            Return ONLY valid JSON:
            {{"bloom_level": "<level>", "confidence": <0-1>}}
            """),
            ("human", "Question: {question_text}\n\nHistorical Bloom Level: {historical_bloom}\n\nClassify this question:")
        ])
    
    def infer_bloom_level(self, qid: str, question_text: str, historical_bloom: str = "Unknown") -> Dict:
        """Infer Bloom's taxonomy level for a question."""
        chain = self.prompt | self.llm
        
        response = chain.invoke({
            "question_text": question_text,
            "historical_bloom": historical_bloom
        })
        
        try:
            result = json.loads(response.content)
            return {
                "qid": qid,
                "bloom_level": result.get("bloom_level", "Unknown"),
                "confidence": float(result.get("confidence", 0.5))
            }
        except Exception as e:
            # Fallback
            return {
                "qid": qid,
                "bloom_level": "Unknown",
                "confidence": 0.0
            }
    
    def infer_bloom_batch(self, questions: List[Dict], historical_blooms: List[str] = None) -> List[Dict]:
        """Infer Bloom levels for multiple questions in one API call."""
        if historical_blooms is None:
            historical_blooms = ["Unknown"] * len(questions)
        
        # Build batch prompt
        questions_text = "\n\n".join([
            f"Question {i+1} (QID: {q['qid']}): {q['text']}\nHistorical Bloom: {historical_blooms[i]}"
            for i, q in enumerate(questions)
        ])
        
        batch_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Bloom's Taxonomy classification expert.
            Analyze each question and determine its cognitive level:
            - Remember: Recall facts, definitions
            - Understand: Explain concepts, interpret
            - Apply: Use knowledge in new situations
            - Analyze: Break down, compare, contrast
            - Evaluate: Judge, critique, justify
            - Create: Design, construct, produce
            
            Return ONLY valid JSON array:
            [
                {{"qid": "Q1", "bloom_level": "<level>", "confidence": <0-1>}},
                {{"qid": "Q2", "bloom_level": "<level>", "confidence": <0-1>}},
                ...
            ]"""),
            ("human", "Questions:\n{questions_text}\n\nClassify all questions:")
        ])
        
        chain = batch_prompt | self.llm
        
        try:
            response = chain.invoke({"questions_text": questions_text})
            results = json.loads(response.content)
            
            # Ensure results match questions order
            result_dict = {r.get("qid"): r for r in results}
            batch_results = []
            for q in questions:
                qid = q["qid"]
                if qid in result_dict:
                    batch_results.append({
                        "qid": qid,
                        "bloom_level": result_dict[qid].get("bloom_level", "Unknown"),
                        "confidence": float(result_dict[qid].get("confidence", 0.5))
                    })
                else:
                    batch_results.append({
                        "qid": qid,
                        "bloom_level": "Unknown",
                        "confidence": 0.0
                    })
            return batch_results
        except Exception as e:
            # Fallback: return Unknown for all
            return [
                {"qid": q["qid"], "bloom_level": "Unknown", "confidence": 0.0}
                for q in questions
            ]

