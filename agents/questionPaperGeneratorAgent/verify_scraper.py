
import sys
import os
from unittest.mock import MagicMock

# Mock langchain_text_splitters before importing the module that uses it
sys.modules["langchain_text_splitters"] = MagicMock()
sys.modules["pdfplumber"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["services"] = MagicMock()
sys.modules["services.prompt"] = MagicMock()
sys.modules["services.prompt.promptProcessor"] = MagicMock()

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agents.questionPaperEvaluatorAgent.questionPaperGenerator import questionPaperGenerator

def test_content_extractor():
    print("Initializing test...")
    generator = questionPaperGenerator()
    
    # Mock dataProcessor to avoid API calls or complex setups
    mock_processor = MagicMock()
    
    # Mock return value for webSearchSelector
    # We will test with a couple of tools
    mock_plan = {
        "plan": [
            {
                "tool": "W3Schools",
                "query": "https://www.w3schools.com/python/python_intro.asp"
            },
            {
                "tool": "GeeksForGeeks",
                "query": "https://www.geeksforgeeks.org/python-programming-language/"
            }
        ]
    }
    mock_processor.webSearchSelector.return_value = mock_plan
    
    generator.dataProcessor = mock_processor
    
    # Mock input content
    test_content = [
        {
            "co": "CO1",
            "po": "PO1",
            "topic": "Python Introduction"
        }
    ]
    
    print("Running ContentExtractor...")
    extracted_content = generator.ContentExtractor(test_content)
    
    print("\n--- Extracted Content ---")
    print(extracted_content[:500] + "...\n[Truncated for brevity]")
    print("-------------------------\n")
    
    if "Content from W3Schools" in extracted_content and "Content from GeeksForGeeks" in extracted_content:
        print("SUCCESS: Content from both tools found.")
    else:
        print("FAILURE: Content missing.")

if __name__ == "__main__":
    test_content_extractor()
