from services.prompt.prompt import agentPrompts, questionGeneratorPrompt
from services.LLMServices import LLMInterface
import json
import re
from typing import Any, Dict, Union

class taskProcessor:
    def __init__(self):
        self.prompts = agentPrompts()
        self.Interfaces = LLMInterface()


    def questionPaperHandler(self , imagePath):
        prompt = self.prompts.questionPaperHandelingPrompt()
        response = self.Interfaces.geminiLLMInterface(prompt=prompt, imagePath=imagePath)
        response = extract_json_from_llm_response(response)

        return response

    def tableSummaryProcessor(self,content):
        prompt = self.prompts.tableSummaryPrompt()
        response = self.Interfaces.nvidiaResponse(prompt=prompt)
        response = extract_clean_answer(
            response,
        )
        return response

    def ragAnswerChecker(self,question_text,marks,retrieved_context):
        prompt = self.prompts.ragAnswerCheckerPrompt(question_text,marks,retrieved_context)
        response = self.Interfaces.nvidiaResponse(prompt=prompt)
        print(response)
        response = parseLLMJson(response)

        return response

    def answerFullFiller(self,question_text,partial_answers,marks):
        prompt = self.prompts.answerFullFillPrompt(question_text,partial_answers,marks)
        response = self.Interfaces.nvidiaResponse(prompt=prompt)
        response = parseLLMJson(response)
        return response
    def finalAnswerGenerator(self,question_text,marks,retrieved_context):
        prompt = self.prompts.finalAnsweringPrompt(question_text,marks,retrieved_context)
        print(prompt)
        response = self.Interfaces.nvidiaResponse(prompt=prompt)
        response = extract_clean_answer(response)
        return  response
    def mathAnswerGenerator(self,question_text,retrieved_context):
        prompt = self.prompts.generate_math_prompt(question_text,retrieved_context)
        response = self.Interfaces.geminiLLMInterface(prompt=prompt)
        response = extract_clean_answer(
            response, )
        return response

    def simpleEvaluator(self,imagePath):
        prompt = self.prompts.simpleEvaluatorPrompt()
        response = self.Interfaces.geminiLLMInterface(prompt=prompt, imagePath=imagePath)
        response = extract_json_from_llm_response(response)
        return response
    def finalAnswerProvider(self,question_json,imagepath):
        prompt = self.prompts.finalMarkProvidingPrompt(question_json)
        response = self.Interfaces.geminiLLMInterface(prompt=prompt, imagePath=imagepath)
        print(response)
        response = extract_json_from_llm_response(response)
        return response



class dataProcessor():
    def __init__(self):
        self.interface = LLMInterface()
        self.prompts = questionGeneratorPrompt()
        
    def hierarchicalDataCreator(self,text):
        prompt = self.prompts.hierarchyMakingPrompt(text)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)

    def topicEnhancer(self,content,chunk):
        prompt = self.prompts.topicEnhansingPrompt(content,chunk)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)

    def webSearchSelector(self,subject,topic):
        prompt = self.prompts.webSearchSelectorPrompt(subject,topic)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)
    
    def baseQuestionCreator(self,content):
        prompt = self.prompts.baseQuestionpaperGeneratorPrompt(content)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response) 
    
    def questionEnhancer(self,content,reference):
        prompt = self.prompts.questionEnhancerPrompt(content,reference=reference    )
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)

    def extractTopics(self, content,element):
        prompt = self.prompts.topicExtractionPrompt(content,element)
        response = self.interface.nvidiaResponse(prompt=prompt, model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)

    def questionEvaluatorMainLoop(self,content,questions,verdict,memory):
        prompt = self.prompts.questionPaperEvaluatorLoopPrompt(content,questions,verdict,memory)
        response = self.interface.nvidiaResponse(prompt=prompt,model="moonshotai/kimi-k2-thinking")
        return extract_json_from_llm_response(response)
    
    def ragQueryGenerator(self,content,prompt):
        prompt = self.prompts.ragQueryGeneratorPrompt(content,prompt)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)

    def contentSummarizer(self,query,retrivedContent):
        prompt = self.prompts.contentSummarizerPrompt(query,retrivedContent)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return response

    def ragFinalizer(self,requirement_prompt,content_summary):
        prompt = self.prompts.ragFinalizerPrompt(requirement_prompt,content_summary)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)

    def stepReasoningSummarizer(self,stepReasoning):
        prompt = self.prompts.stepReasoningSummarizerPrompt(stepReasoning)
        response = self.interface.nvidiaResponse(prompt=prompt,model="mistralai/mixtral-8x7b-instruct-v0.1")
        return response

    def questionValidator(self, questions):
        """Validates and cleans questions to remove metadata and formatting issues"""
        prompt = self.prompts.questionValidatorPrompt(questions)
        response = self.interface.nvidiaResponse(prompt=prompt, model="mistralai/mixtral-8x7b-instruct-v0.1")
        return extract_json_from_llm_response(response)

    
def parseLLMJson(llm_output):
    """
    Extract and parse JSON from LLM output, ignoring extra text or <think> tags.
    """
    if isinstance(llm_output, dict):
        return llm_output

    # Try to find the first JSON code block
    match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, flags=re.DOTALL)
    if match:
        llm_output = match.group(1)
    else:
        # Fallback: try to find any standalone JSON-like object
        match = re.search(r"(\{.*\})", llm_output, flags=re.DOTALL)
        if match:
            llm_output = match.group(1)

    try:
        parsed = json.loads(llm_output)
    except json.JSONDecodeError:
        parsed = {
            "answer": llm_output.strip(),
            "is_fulfilled": False,
            "require_diagram": False,
            "diagram_search_queries": [],
            "error": "Could not parse JSON from LLM output"
        }

    # Ensure all expected keys exist
    defaults = {
        "answer": "",
        "is_fulfilled": False,
        "require_diagram": False,
        "diagram_search_queries": []
    }
    for key, value in defaults.items():
        parsed.setdefault(key, value)

    return parsed


def extract_clean_answer(llm_output: str) -> str:
    """
    Extract the final answer from LLM output by removing <think> tags.
    Works whether the output is plain text or includes JSON/code blocks.
    """
    if not llm_output:
        return ""

    # Remove <think>...</think> sections
    cleaned = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL).strip()

    # Remove JSON code block markers if any
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    
    return cleaned

def extract_json_from_llm_response(text: str):
    """
    Extracts JSON object or list from LLM output.
    Handles ```json blocks and raw JSON.
    Returns a Python dict or list.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Remove ```json or ``` markers
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()

    # Try to find JSON array or object using regex
    # Using non-greedy match might be safer if there is trailing text, 
    # but for now let's stick to simple cleaning and parsing.
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)

    if not match:
        # If regex fails, try parsing the cleaned text directly in case it's just a number or string
        json_str = cleaned
    else:
        json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            import ast
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError) as e:
            interface = LLMInterface()
            content = json_str    
            reformationPrompt = f"""You will be given text that is intended to be valid JSON but may be malformed, truncated, or contain extra characters due to token limits.
            input: {content}
            Your task:
            1. Recover and reconstruct the JSON so that it is valid and syntactically correct (but don't change the json schema).
            2. Preserve all fields and values that are present.
            3. If a value is clearly truncated, fill it with a reasonable placeholder such as:
            - "" for strings
            - 0 for numbers
            - [] for arrays
            - {{}} for objects
            4. Do NOT remove any fields unless they are completely unrecoverable.
            5. Do NOT add extra fields that were not present in the original structure.
            6. Output ONLY valid JSON — no explanations, no markdown, no code fences.
            7. return the output in the same format as the input.

            If the input is not JSON-like at all, return:
            {{}}
            """
            response = interface.nvidiaResponse(prompt=reformationPrompt, model="mistralai/mixtral-8x7b-instruct-v0.1")
            return extract_clean_answer(response)
             
            raise ValueError(f"Extracted content is not valid JSON. Content: {json_str[:100]}... Error: {e}")


