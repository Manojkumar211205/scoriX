import json
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import Config


class TopicAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.2,
            api_key=Config.NVIDIA_API_KEY,
            base_url=Config.NVIDIA_BASE_URL
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a topic classification expert.
            Identify the main topic and subtopic from the question.
            Topics should be specific and domain-relevant (e.g., "Computer Architecture", "Operating Systems").
            Subtopic should be more granular (e.g., "Cache Memory", "Process Scheduling").
            
            Return ONLY valid JSON:
            {{"topic": "<main topic>", "subtopic": "<subtopic>"}}
            """),
            ("human", "Question: {question_text}\n\nIdentify topic and subtopic:")
        ])
    
    def infer_topic(self, qid: str, question_text: str) -> Dict:
        """Infer topic and subtopic for a question."""
        chain = self.prompt | self.llm
        
        response = chain.invoke({
            "question_text": question_text
        })
        
        try:
            result = json.loads(response.content)
            return {
                "qid": qid,
                "topic": result.get("topic", "Unknown"),
                "subtopic": result.get("subtopic", "Unknown")
            }
        except Exception as e:
            return {
                "qid": qid,
                "topic": "Unknown",
                "subtopic": "Unknown"
            }
    
    def infer_topic_batch(self, questions: List[Dict]) -> List[Dict]:
        """Infer topics for multiple questions in one API call."""
        # Build batch prompt
        questions_text = "\n\n".join([
            f"Question {i+1} (QID: {q['qid']}): {q['text']}"
            for i, q in enumerate(questions)
        ])
        
        batch_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a topic classification expert.
            Identify the main topic and subtopic for each question.
            Topics should be specific and domain-relevant (e.g., "Computer Architecture", "Operating Systems").
            Subtopic should be more granular (e.g., "Cache Memory", "Process Scheduling").
            
            Return ONLY valid JSON array:
            [
                {{"qid": "Q1", "topic": "<main topic>", "subtopic": "<subtopic>"}},
                {{"qid": "Q2", "topic": "<main topic>", "subtopic": "<subtopic>"}},
                ...
            ]"""),
            ("human", "Questions:\n{questions_text}\n\nIdentify topic and subtopic for all questions:")
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
                        "topic": result_dict[qid].get("topic", "Unknown"),
                        "subtopic": result_dict[qid].get("subtopic", "Unknown")
                    })
                else:
                    batch_results.append({
                        "qid": qid,
                        "topic": "Unknown",
                        "subtopic": "Unknown"
                    })
            return batch_results
        except Exception as e:
            # Fallback: return Unknown for all
            return [
                {"qid": q["qid"], "topic": "Unknown", "subtopic": "Unknown"}
                for q in questions
            ]

