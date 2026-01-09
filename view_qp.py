import streamlit as st
import json
import os
import subprocess
from datetime import datetime

st.set_page_config(page_title="ScoriX", page_icon="📝", layout="wide")
st.title("📝 ScoriX - Question Paper Generator")

st.info("💡 **How to use:** Enter your content below, save it, then click 'Generate' to run the generator in a separate process.")

# Input
input_text = st.text_area("Course Content:", height=300, value="""Course: Programming Fundamentals and Data Structures

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
""")

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Save Input", use_container_width=True):
        with open("input.txt", 'w', encoding='utf-8') as f:
            f.write(input_text)
        st.success("✅ Input saved to input.txt")

with col2:
    if st.button("🚀 Generate (Run Separately)", use_container_width=True):
        st.warning("⚠️ Please run: `python generate_qp.py` in your terminal")
        st.code("python generate_qp.py", language="bash")

st.markdown("---")

# Display latest result
if os.path.exists("latest_qp.json"):
    st.success("✅ Found generated question paper!")
    
    with open("latest_qp.json", 'r', encoding='utf-8') as f:
        qp_data = json.load(f)
    
    if isinstance(qp_data, list) and len(qp_data) >= 2:
        questions_data = qp_data[1]
        
        for section in questions_data:
            st.markdown(f"### {section.get('CO', 'N/A')}")
            st.markdown(f"**PO:** {', '.join(section.get('PO', []))}")
            
            for i, q in enumerate(section.get('questions', []), 1):
                st.markdown(f"{i}. {q}")
            
            st.markdown("---")
        
        st.download_button(
            "💾 Download JSON",
            data=json.dumps(qp_data, indent=2),
            file_name=f"qp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.json(qp_data)
else:
    st.info("📄 No question paper generated yet. Run `python generate_qp.py` to generate one.")
