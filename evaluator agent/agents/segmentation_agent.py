import re
import json
from typing import List, Dict
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from pydantic import BaseModel, Field

from config import Config


class QuestionSegment(BaseModel):
    qid: str = Field(description="Question identifier like Q1, Q2")
    text: str = Field(description="Full question text")


class SegmentationOutput(BaseModel):
    questions: List[QuestionSegment]


class SegmentationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0,
            api_key=Config.NVIDIA_API_KEY,
            base_url=Config.NVIDIA_BASE_URL
        )
        
    def segment_questions(self, input_text: str) -> Dict:
        """Split input text into individual questions using numbering patterns."""
        
        # Pattern to match numbered questions: 1), 2), 3) or 1. 2. 3.
        pattern = r'(\d+)[\)\.]\s*(.+?)(?=\d+[\)\.]\s*|$)'
        matches = re.findall(pattern, input_text, re.DOTALL)
        
        questions = []
        for idx, (num, text) in enumerate(matches, 1):
            qid = f"Q{idx}"
            clean_text = text.strip()
            questions.append({
                "qid": qid,
                "text": clean_text
            })
        
        # If regex fails, use LLM as fallback
        if not questions:
            questions = self._llm_segment(input_text)
        
        return {"questions": questions}
    
    def _llm_segment(self, input_text: str) -> List[Dict]:
        """Fallback LLM-based segmentation."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a question segmentation expert. 
            Extract numbered questions from the input text.
            Return ONLY valid JSON in this exact format:
            {{"questions": [{{"qid": "Q1", "text": "question text"}}, ...]}}
            """),
            ("human", "Segment these questions:\n{input}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"input": input_text})
        
        try:
            result = json.loads(response.content)
            return result.get("questions", [])
        except:
            return []

