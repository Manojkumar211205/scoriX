"""
Simple script to generate question paper
Run this separately, then view results in Streamlit
"""
import json
import sys
from datetime import datetime
from agents.questionPaperGeneratorAgent.questionPaperGenerator import QuestionPaperGenerator

def generate_qp(text, output_file="latest_qp.json"):
    """Generate question paper and save to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection = f"qp_{timestamp}"
    
    print(f"Generating question paper...")
    print(f"Collection: {collection}")
    
    qpgen = QuestionPaperGenerator(collectionName=collection)
    output = qpgen.demoQuestionpaperGenerator(text=text, filePath="")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to: {output_file}")
    return output

if __name__ == "__main__":
    # Read input from file or use default
    try:
        with open("input.txt", 'r', encoding='utf-8') as f:
            text = f.read()
    except:
        text = """Course: Programming Fundamentals and Data Structures

Course Outcomes:
CO1: Understand the basic concepts of programming, including variables, data types, and control structures.
CO2: Apply programming constructs such as loops, functions, and conditionals to solve computational problems.
CO3: Analyze and implement fundamental data structures to solve real-world problems efficiently.

Program Outcomes:
PO1: Engineering knowledge – Apply knowledge of mathematics and computing fundamentals.
PO2: Problem analysis – Identify, formulate, and analyze computational problems.
PO3: Design/development of solutions – Design and implement efficient algorithms.

Syllabus Content:
Unit 1: Introduction to programming, variables, data types, input/output operations.
Unit 2: Control structures – conditional statements, loops, and functions.
Unit 3: Arrays, strings, and basic operations.
Unit 4: Data structures – stacks, queues, linked lists.
Unit 5: Searching and sorting algorithms, time and space complexity.
"""
    
    generate_qp(text)
