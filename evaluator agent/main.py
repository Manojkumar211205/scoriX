"""Production entry point for Question Paper Validation Pipeline."""

from pipeline import QuestionPaperValidationPipeline
import sys
import json

def main():
    """Main entry point for production use."""
    if len(sys.argv) < 2:
        print("Usage: python main.py '<question_paper_text>'")
        print("\nExample:")
        print('python main.py "1) Define cache memory.\\n2) Explain cache coherence protocol."')
        sys.exit(1)
    
    # Get input from command line
    input_text = sys.argv[1]
    
    # Initialize pipeline
    pipeline = QuestionPaperValidationPipeline()
    
    # Process
    result = pipeline.process(input_text)
    
    # Output strict JSON only
    if "error" in result:
        print(json.dumps({"error": result["error"]}, indent=2))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()


