from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.webSearcher.scrapers import *
import pdfplumber
from services.prompt.promptProcessor import dataProcessor
from ragSystems.ragProcessor import HybridRagProcessor
import concurrent.futures
import logging
import os
from collections import defaultdict
from threading import Lock
from data.events import emit

try:
    from keybert import KeyBERT
except ImportError:
    KeyBERT = None

load_dotenv()

# Global model cache
_kw_model = None


# event_store = defaultdict(list)
# lock = Lock()

# def emit(task_id: str, msg: str):
#     with lock:
#         event_store[task_id].append({
#             "msg": msg
#         })

def get_kw_model():
    global _kw_model
    if _kw_model is None and KeyBERT is not None:
        try:
             # Use a smaller model for speed
            _kw_model = KeyBERT('all-MiniLM-L6-v2')
        except Exception as e:
            pass
    return _kw_model
def _extract_additional_topics(self, content: str) -> list:
        """Extracts keywords/topics from content using KeyBERT with safe chunking and multi-threading."""
        kw_model = get_kw_model()
        if not kw_model or not content:
            return []

        try:
            # -------- Chunk into ~400-word blocks --------
            words = content.split()
            chunk_size = 400
            chunks = [
                " ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]

            all_keywords = []

            def process_chunk(chunk):
                try:
                    keywords = kw_model.extract_keywords(
                        chunk,
                        keyphrase_ngram_range=(1, 2),  # 1–2 word topics
                        stop_words='english',
                        top_n=5                         # topics per chunk
                    )
                    return [kw[0] for kw in keywords]
                except Exception:
                    return []

            # Use ThreadPoolExecutor for parallel processing
            # Using threads because KeyBERT/PyTorch might release GIL for heavy ops
            max_workers = os.cpu_count() or 4
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_chunk, chunks))
            
            for res in results:
                all_keywords.extend(res)

            # -------- Deduplicate while keeping order --------
            seen = set()
            unique_topics = []
            for kw in all_keywords:
                kw_lower = kw.lower()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    unique_topics.append(kw)

            return unique_topics

        except Exception as e:
            # print(f"KeyBERT extraction failed: {e}") # Keeping clean logs
            return []


class QuestionPaperGenerator():
    def __init__(self,collectionName):
        # Store collection name for emit calls
        self.collectionName = collectionName
        
        # Setup logging
        logging.basicConfig(
            filename='agent.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing QuestionPaperGenerator Agent...")
        
        emit(task_id=collectionName, msg="🚀 Initializing Question Paper Generator...")

        self.currentQuestionPaper = []
        self.currentContent = []
        
        emit(task_id=collectionName, msg="📊 Loading data processor...")
        self.dataProcessor = dataProcessor()
        
        emit(task_id=collectionName, msg="🔍 Initializing RAG system...")
        self.ragSystem = HybridRagProcessor(collectionName)
        
        emit(task_id=collectionName, msg="✅ Question Paper Generator initialized successfully!")
        
        self.mainMemory = {
            "toolMemory" : [],
            "questionMemory" : set(),
            "stepReasoning" : []
        }
        return

    def hierarchicalDataCreator(self,text,filePath):
        emit(task_id=self.collectionName, msg="📝 Creating hierarchical structure from input text...")
        
        hierarchyJson = self.dataProcessor.hierarchicalDataCreator(text)
        emit(task_id=self.collectionName, msg="✅ Hierarchical structure created successfully!")


        if(filePath):

            fileContent = self._pdfProcessor(filePath)
            for i, chunk in enumerate(fileContent):

                hierarchyJson = self.dataProcessor.topicEnhansingPrompt(content=hierarchyJson,chunk=chunk)

        self.logger.info(f"Initial hierarchy created: {hierarchyJson}")

        emit(task_id=self.collectionName, msg=f"Initial hierarchy created: {hierarchyJson}")

        return hierarchyJson

    def ContentExtractor(self, content):
        mySet = set()
        finalContent = []
        # Mapping tool names to scraper classes
        scraper_map = {
            "W3SchoolsScraper": W3SchoolsScraper,
            "GeeksForGeeksScraper": GeeksForGeeksScraper,
            "NPTELScraper": NPTELScraper,
            "MITOCWScraper": MITOCWScraper,
            "OpenStaxScraper": OpenStaxScraper,
            "UniversityEDUScraper": UniversityEDUScraper
        }

        for element in content:
            extractedToolContent = ""

            
            for topic in element["topics"]:

                toolSelector = self.dataProcessor.webSearchSelector(f"{element['co']} {element['po']}", topic)
                emit(task_id=self.collectionName, msg=f"Tool Selection Plan: {toolSelector}")

                # Check for "plans" (plural) as defined in prompt, or "plan" (singular) as fallback
                plans = toolSelector.get("plans", toolSelector.get("plan", []))
                self.logger.info(f"Tool Selection Plan: {plans}")
                
                if plans:
                    for plan in plans:
                        tool_name = plan.get("tool")
                        query = plan.get("query")
                        emit(task_id=self.collectionName, msg=f"Tools selected: {tool_name}, Query: {query}")
                        
                        if tool_name in scraper_map:
                            scraper_class = scraper_map[tool_name]
                            scraper = scraper_class()
                            
                            try:
                                # Fetch content using the scraper

                                result = scraper.fetch(query)
                                scraped_text = result.get('content', '')
                                emit(task_id=self.collectionName, msg=f"Scraped content: {scraped_text}")
                                if scraped_text not in mySet:
                                    extractedToolContent += f"Content:\n{scraped_text}\n"

                                    # Process topics
                                    raw_topic = result.get('topic')
                                    emit(task_id=self.collectionName, msg=f"Raw topic: {raw_topic}")
                                    raw_topic .extend(scraper._extract_additional_topics(scraped_text))
                                    raw_topic = self.dataProcessor.extractTopics(raw_topic,f"{element['co']+element['po']}")
                                    topics_list = []
                                    emit(task_id=self.collectionName, msg=f"Topics list: {topics_list}")
                                    if raw_topic and isinstance(raw_topic, list) and len(raw_topic) > 0:
                                        print("-------- raw topic working")
                                        topics_list.extend(raw_topic)

                                        
                                    elif raw_topic and isinstance(raw_topic, str) and len(raw_topic.strip()) > 0:
                                        # Backward compatibility
                                        topics_list = [raw_topic]
                                        topics_list = self.dataProcessor.extractTopics(scraped_text[:2000])
                                        emit(task_id=self.collectionName, msg=f"Topics list: {topics_list}")
                                    else:
                                        # Fallback: Extract from content
                                        try:
                                            topics_list = self.dataProcessor.extractTopics(scraped_text[:2000]) # Limit context
                                        except Exception as e:
                                            emit(task_id=self.collectionName, msg=f"Error extracting topics: {e}")
                                            topics_list = ["Unknown Topic"]

                                    if "scraped_data" not in element:
                                        element["scraped_data"] = []
                                    
                                    # Store as a flat list of topics or list of lists? 
                                    # User said "returned as list of topics [t1,t2]"
                                    # Let's extend the main list
                                    if isinstance(topics_list, list):
                                        self.logger.info(f"Extending topics: {element}")
                                        element["scraped_data"].extend(topics_list)
                                    else:
                                        self.logger.info(f"Appending topic: {element}")
                                        element["scraped_data"].append(str(topics_list))
                                
                                else:
                                    mySet.add(scraped_text)
                                    continue
                                                       
                            except Exception as e:
                                pass
                        else:
                            pass

            # Clean up and deduplicate extracted topics
            if "scraped_data" in element and isinstance(element["scraped_data"], list):
                 element["scraped_data"] = list(set(element["scraped_data"]))
                 print("scraped topics")
                 print(element["scraped_data"])
                 element["topics"].extend(element["scraped_data"])
                 del element["scraped_data"]

            # Upsert code moved outside the inner loop to chunk all gathered content for this element
            if extractedToolContent:
                # self.logger.info(f"scraped content from tools: {extractedToolContent}")
                chunks = self._recursiveChunker(extractedToolContent)
                emit(task_id=self.collectionName, msg=f"Chunks content: {chunks}")

                # self.logger.info(f"Chunks content: {chunks}") # Optional: Uncomment to log full chunks if needed
                
                # metadata must be a list of dicts, one for each chunk
                base_metadata = {"CO": element["co"]}
                # Add text content to metadata as well, as HybridRagProcessor methods usually expect 
                # payload to contain text or store it differently. 
                # Checking ragProcessor.py:44 call -> qdrantManager.upsert_integrated_hybrid
                # QdrantManager usually stores metadata as payload.
                # Just ensuring 'text' key is present if required, but chunks are passed separately.
                # However, usually payload needs the text content to be retrieved later.
                # Let's check ragProcessor.py again... 
                # It does: `qdrantManager.upsert_integrated_hybrid(dense_vecs, sparse_vecs, metadata)`
                # It does NOT verify if 'text' is in metadata.
                # So I should probably add the text chunk to the metadata for each chunk.
                
                metadata_list = []
                for chunk in chunks:
                     meta = base_metadata.copy()
                     meta["text"] = chunk
                     metadata_list.append(meta)
                
                self.ragSystem.process_and_store(chunks=chunks, metadata=metadata_list)
            
            finalContent.append(element)    
            self.logger.info(f"Final content of scraped data: {finalContent}")

        
        return finalContent

    def baseQuestionpaperGenerator(self,content):
        finalQuestions = []
        for i, element in enumerate(content):
            print(content)
            questions = self.dataProcessor.baseQuestionCreator(f"{element}")
            print(questions)
            self.logger.info(f"base question generation input: {element}")
            self.logger.info(f"Generated base questions: {questions}")
            
            questionsDict = {}
            questionsDict["CO"]=element["co"]
            questionsDict["PO"]=element["po"]
            questionsDict["topics"]=element["topics"]
            questionsDict["questions"] = []
            
            for j, q in enumerate(questions):
                self.logger.info(f"Enhancing question with RAG search query: {q}")
                retrivalResult = self.ragSystem.search(query=q,metadata_filter={"CO": element["co"]})
                emit(task_id=self.collectionName, msg=f"RAG Retrieval Result: {retrivalResult}")
                inputText = []
                for data in retrivalResult:
                    inputText.append(data["text"])
                emit(task_id=self.collectionName, msg=f"RAG retrieval result: {inputText}")
                if not inputText:
                    inputText.extend([q,"no content found create question on your own but dont add irrelavent questions"])

                enhancedQuestions = self.dataProcessor.questionEnhancer(inputText,reference=f"{questionsDict["CO"]}+ {questionsDict["PO"]} + {q}")
                emit(task_id=self.collectionName, msg=f"Enhanced questions: {enhancedQuestions}")
                
                # Fix: Ensure enhancedQuestions is always a list
                if isinstance(enhancedQuestions, str):
                    # If it's a string, wrap it in a list to prevent character-by-character splitting
                    self.logger.warning(f"enhancedQuestions returned as string instead of list. Wrapping in list.")
                    questionsDict["questions"].append(enhancedQuestions)
                elif isinstance(enhancedQuestions, list):
                    questionsDict["questions"].extend(enhancedQuestions)
                else:
                    # Handle unexpected types (dict, None, etc.)
                    self.logger.error(f"Unexpected type for enhancedQuestions: {type(enhancedQuestions)}. Skipping.")
              
            
            finalQuestions.append(questionsDict)
        
        return content,finalQuestions

    def demoQuestionpaperGenerator(self,text,filePath):
        emit(task_id=self.collectionName, msg="📋 Starting question paper generation pipeline...")
        
        emit(task_id=self.collectionName, msg="🏗️ Step 1/4: Creating hierarchical structure...")
        hierarchyJson = self.hierarchicalDataCreator(text,filePath)
        
        emit(task_id=self.collectionName, msg="🌐 Step 2/4: Extracting content from web sources...")
        content = self.ContentExtractor(hierarchyJson)
        
        emit(task_id=self.collectionName, msg="❓ Step 3/4: Generating base questions...")
        baseQP = self.baseQuestionpaperGenerator(content)
        self.logger.info(f"Base Question Paper: {baseQP}")
        
        # Validate and clean questions asynchronously
        emit(task_id=self.collectionName, msg="✨ Step 4/4: Validating and enhancing questions...")
        self.logger.info("Starting async question validation...")
        content_data, finalQuestions = baseQP
        
        def validate_co_questions(co_data):
            """Validate questions for a single CO - processes entire list or in batches of 10"""
            co_index, co_dict = co_data
            questions_list = co_dict.get("questions", [])
            
            if not questions_list:
                return co_index, co_dict
            
            self.logger.info(f"Validating {len(questions_list)} questions for CO: {co_dict.get('CO', 'Unknown')[:50]}...")
            
            # Call validator with batching for token efficiency
            try:
                # If questions list is large (>10), process in batches of 10
                batch_size = 10
                all_cleaned_questions = []
                
                if len(questions_list) <= batch_size:
                    # Process all questions in one call (token efficient)
                    cleaned_questions = self.dataProcessor.questionValidator(questions_list)
                    
                    if isinstance(cleaned_questions, list):
                        all_cleaned_questions = cleaned_questions
                    else:
                        self.logger.warning(f"Validator returned non-list: {type(cleaned_questions)}")
                        all_cleaned_questions = questions_list
                else:
                    # Process in batches of 10 for very large lists
                    emit(self.collectionName, msg=f"Processing {len(questions_list)} questions in batches of {batch_size}")
                    
                    for i in range(0, len(questions_list), batch_size):
                        batch = questions_list[i:i + batch_size]
                        emit(self.collectionName, msg=f"Validating batch {i//batch_size + 1} ({len(batch)} questions)")
                        
                        cleaned_batch = self.dataProcessor.questionValidator(batch)
                        
                        if isinstance(cleaned_batch, list):
                            all_cleaned_questions.extend(cleaned_batch)
                        else:
                            all_cleaned_questions.extend(batch)  # Keep original batch
                
                co_dict["questions"] = all_cleaned_questions
                self.logger.info(f"✅ Successfully validated {len(all_cleaned_questions)} questions for CO")
                    
            except Exception as e:
                self.logger.error(f"❌ Validation failed for CO: {e}")
                co_dict["questions"] = questions_list  # Keep original on error
            
            return co_index, co_dict
        
        # Use ThreadPoolExecutor for async validation
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all CO validations
            future_to_co = {
                executor.submit(validate_co_questions, (i, co_dict)): i 
                for i, co_dict in enumerate(finalQuestions)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_co):
                co_index, validated_co = future.result()
                finalQuestions[co_index] = validated_co
                emit(self.collectionName, msg=f"Question validation complete for CO: {validated_co['CO']}") 
        
        emit(self.collectionName, msg="Question validation complete")
        print("done processing questions")
            
        return content_data, finalQuestions

        
    def mainEvaluatorLoop(self,verdict):
        for i,content in enumerate(self.currentContent):
            generatedQuestions = self.currentQuestionPaper[i]["questions"]

    def mainQuestionPaperEvaluator(self,content,generatedQuestions,verdict):
    
        while (True):
            if len(self.mainMemory["stepReasoning"]) > 6:
                summary = self.dataProcessor.stepReasoningGenerator(self.mainMemory["stepReasoning"]) 
                self.mainMemory["stepReasoning"] = []
                self.mainMemory["stepReasoning"].append(summary)
               
            mainLoop = self.dataProcessor.questionEvaluatorMainLoop(content,generatedQuestions,verdict,self.mainMemory)
            if "tool" in mainLoop and mainLoop["tool"] == "ragAgent":
                #call rag agent
                callDetails = {"tool":"ragAgent","prompt":mainLoop["prompt"]}
                self.mainMemory["toolMemory"].append(callDetails)
                ragOutput = self.__ragAgent(content,mainLoop["prompt"])
                if "questions" in ragOutput:
                    self.mainMemory["questionMemory"].update(ragOutput["questions"])
                if "stepReasoning" in mainLoop:
                    self.mainMemory["stepReasoning"].extend(mainLoop["stepSummary"])

            elif "questions" in mainLoop and mainLoop["taskOver"] == False:
                self.mainMemory["questionMemory"].update(mainLoop["questions"])
                if "stepReasoning" in mainLoop:
                    self.mainMemory["stepReasoning"].extend(mainLoop["stepSummary"])

            else:
                return content,mainLoop["questions"]
                
    def __ragAgent(self,content,prompt):
        ragQuery = self.dataProcessor.ragQueryGenerator(content,prompt)
        retrivedContent = []
        duplicatesAnalyser = set()
        for query in ragQuery:
            retrived = self.ragSystem.search(query=query,metadata_filter={"CO": content["co"]})
            self.logger.info(f"Retrieved content: {retrived}")
            for data in retrived:
                if data["text"] not in duplicatesAnalyser:
                    duplicatesAnalyser.add(data["text"])
                else:
                    retrived.remove(data)
            contentSummary = self.dataProcessor.contentSummarizer(query,retrived)
            retrivedContent.append(contentSummary)

        finalContent = self.dataProcessor.ragFinalizer(requirement_prompt=prompt,content_summary=retrivedContent)
        return finalContent

        


        
            
            

            


            
            
                
       
    def _pdfProcessor(self, pdf_path, chunk_size=500, chunk_overlap=50):
        """
        Extracts text from a PDF and splits it into token-efficient chunks.
        """
        text_content = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text_content += (page.extract_text() or "") + "\n"
        except Exception as e:
            self.logger.error(f"Error reading PDF: {e}")
            return []

        # Fix: call self._recursiveChunker and pass args
        chunks = self._recursiveChunker(text_content, chunk_size, chunk_overlap)
        return chunks

    def _recursiveChunker(self, textContent, chunk_size=500, chunk_overlap=50):
        if not textContent.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(textContent)
        return chunks

    # def questionPaperGeneratorAgent(self):

        return