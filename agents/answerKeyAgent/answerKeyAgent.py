import redis
import json
import uuid


from torch.ao.ns.fx.utils import compute_normalized_l2_error

from services.prompt.promptProcessor import taskProcessor
from ragSystems.ragProcessor import ragProcessor
from agents.answerKeyAgent.documentProcessor import documentProcessor
from typing import Dict, List, Optional, Any
from ragSystems.imageRag import ClipSearchService
from dotenv import load_dotenv
from serpapi.google_search import GoogleSearch
import requests, os
import wikipediaapi
from duckduckgo_search import DDGS

# r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Store question paper JSON
# qp_data = {"subject": "Maths", "questions": [{"qno": 1, "text": "2+2?", "marks": 2}]}
# r.set("qp_2025_maths", json.dumps(qp_data))
#
# # Retrieve later
# qp = json.loads(r.get("qp_2025_maths"))
# print(qp["subject"])  # Maths

class answerKeyAgent:
    def __init__(self,uniqID,notesProvided):
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.taskProcessors = taskProcessor()
        if  notesProvided:
           self.documentProcessors = documentProcessor(collection_name=uniqID)
        self.uniqID = uniqID
        self.wiki_api = wikipediaapi.Wikipedia(
            language="en",
            user_agent="MyWebSearchAgent/1.0"
        )
        self.notesProvided = notesProvided
        # Initialize the DuckDuckGo Search API wrapper
        self.ddgs_api = DDGS()

    def questionPaperHandler(self,imagePath):
        response = self.taskProcessors.questionPaperHandler(imagePath=imagePath)
        # self.r.set(self.uniqID, json.dumps(qp_data))

    def storeAnswerKey(self,answerKeyPath):
        #store documents in vector db
        if isinstance(answerKeyPath, list):
            for path in answerKeyPath:
                self.documentProcessors.processor(path)
        elif isinstance(answerKeyPath, str):
             self.documentProcessors.processor(answerKeyPath)

    def _search_wikipedia(self, topic: str) -> Optional[Dict[str, str]]:

        page = self.wiki_api.page(topic)

        if not page.exists():

            return None


        summary_sentences = page.summary.split('.')
        short_summary = ".".join(summary_sentences[0:3]) + "."

        return {
            "title": page.title,
            "summary": short_summary,
            "url": page.fullurl
        }

    def _search_duckduckgo(self, query: str, max_results=5) -> List[Dict[str, str]]:

        results = self.ddgs_api.text(keywords=query, max_results=max_results)

        if not results:
            print("[Agent] No DuckDuckGo results found.")
            return []

        # Format results into a cleaner list of dictionaries
        formatted_results = [
            {
                "title": result['title'],
                "snippet": result['body'],
                "url": result['href']
            }
            for result in results
        ]
        return formatted_results



    def answerGenerator(self,question,marksAllocated):
        #search for answer
        if self.notesProvided:
            print("using rag")
            ragAnswer = self.documentProcessors.search(question)
            print("retrived content from rag :")
            print(ragAnswer)
            response = self.taskProcessors.ragAnswerChecker(question,marksAllocated, ragAnswer)
        else:
            print("not using rag")
            ragAnswer = "nothing retrieved"
            response = self.taskProcessors.ragAnswerChecker(question,marksAllocated, ragAnswer)
            print("all done")
        print("rag full fill check answer")
        print(response)
        imgReturnPath = None
        finalAnswer = response["answer"]
        if response["require_diagram"]:

            clipRag = ClipSearchService(str(self.uniqID) + "img")
            imgReturnPath = []

            for img_query in response["diagram_search_queries"]:
                score = 0
                imagePaths = clipRag.search_by_text(img_query, limit=1)

                for _,imagePath in enumerate(imagePaths):
                    imageDetails = imagePath.payload.get('filename')
                    score = imagePath.score

                    imgReturnPath.append(imageDetails)
                if score < 0.2:
                    print("getting image from google")
                    imageDetails = get_images_for_question(img_query)
                    imgReturnPath.append(imageDetails)

        if not response["is_fulfilled"]:
            router = self.taskProcessors.answerFullFiller(question,response["answer"],marksAllocated)
            print("answer not full filled")
            if not router['duckduckgo_search'] == "not_required":
                print("answer by duckduckgo")
                searchResults = self._search_duckduckgo(router["duckduckgo_search"])
                for result in searchResults:
                    finalAnswer += "\n"
                    finalAnswer += str(result)
            elif not router['wikipedia_search'] == "not_required":
                print("answer by wikipedia")
                searchResults = self._search_wikipedia(router["wikipedia_search"])
                for result in searchResults or []:
                    finalAnswer += "\n"
                    finalAnswer += str(result)
            elif not router['math_answer_generator'] == "not_required":
                print("answer by math")
                finalAnswer = self.taskProcessors.mathAnswerGenerator(question,ragAnswer)

        finalAnswer = self.taskProcessors.finalAnswerGenerator(question,marksAllocated,finalAnswer)
        output = {"Answer": finalAnswer, "imageRequired": response["require_diagram"]}
        if imgReturnPath:
            output["imageDetails"] = imgReturnPath

        return output

    def calculate_total_marks(self,data_json) -> float:
        try:
            data = data_json
            total_marks = sum(item.get("marks", 0) for item in data)
            return total_marks
        except Exception as e:
            print("Error:", e)
            return 0.0

    def answerKeyAgent(self,questionPaperPaths,answerKeyPath = None,simpleAnswerSheet=False):
        if simpleAnswerSheet:
            print(simpleAnswerSheet)
            llmResponse = []
            print("manoj")
            for questionPaperPath in questionPaperPaths:
                llmResponse.extend(self.taskProcessors.simpleEvaluator(imagePath=questionPaperPath))
            finalAnswer = []
            for questionPaperPath in questionPaperPaths:
                finalAnswer.extend(self.taskProcessors.finalAnswerProvider(imagepath=questionPaperPath,question_json=llmResponse))


            print(llmResponse)
            totalMarks = self.calculate_total_marks(finalAnswer)
            return {"mark": totalMarks}
        else:
            if answerKeyPath:
                if isinstance(answerKeyPath, str):
                    self.documentProcessors.processor(answerKeyPath)
                elif isinstance(answerKeyPath, list):
                    for path in answerKeyPath:
                        self.documentProcessors.processor(path)
            if questionPaperPaths:
                if isinstance(questionPaperPaths, str):
                    questionPaper = self.taskProcessors.questionPaperHandler(questionPaperPaths)
                elif isinstance(questionPaperPaths, list):
                    partialQuestionPaper = []
                    for questionPaperPath in questionPaperPaths:
                        partialQuestionPaper.extend(self.taskProcessors.questionPaperHandler(questionPaperPath))
                        questionPaper = partialQuestionPaper
                else:
                    questionPaper = None
            else:
                return []



            answerKey = []
            for component in questionPaper:

                question = component["question"]
                marksAllocated = component["marks"]
                response = self.answerGenerator(question,marksAllocated)
                component["answer"]= response["Answer"]
                component["imageRequired"] = response["imageRequired"]
                component["imageDetails"] = response.get("imageDetails", {})

                print("component")
                print(component)
                answerKey.append(component)

            return answerKey


def get_images_for_question(question, num_images=1, output_folder="D:\\scoriX_agent\\data\\referenceImages"):
    load_dotenv()
    filepath = ""
    params = {
        "engine": "google_images",
        "q": question,
        "api_key": os.getenv("SERPAPI")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "images_results" not in results:
        print("No images found.")
        return

    for i, image_info in enumerate(results["images_results"][:num_images]):
        image_url = image_info.get("original")
        if not image_url:
            continue
        image_data = requests.get(image_url).content
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(output_folder, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        print(f"✅ Saved: {filename}")

    return filepath





