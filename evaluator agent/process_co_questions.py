"""Process questions from JSON file, running pipeline separately for each CO."""

import json
import re
import sys
import os
from datetime import datetime
from typing import List, Set, Any, Dict
from pipeline import QuestionPaperValidationPipeline


def normalize_text(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces/newlines to single space."""
    if not text:
        return ""
    # Replace newlines, tabs, and multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_valid_question(text: str, min_length: int = 15) -> bool:
    """Check if text looks like a valid question."""
    if not text or len(text) < min_length:
        return False
    
    # Filter out common non-question patterns
    invalid_patterns = [
        r'^\[.*\]$',  # Just brackets
        r'^\{.*\}$',  # Just braces
        r'^PO\d+:',  # Just PO reference
        r'^CO\d+:',  # Just CO reference
        r'^Unit \d+:',  # Just unit reference
        r'^In the context of the previous question',  # Reference only
        r'^Retrieved Content',  # Metadata
        r'^Original Question:',  # Label only
        r'^Higher-order thinking',  # Label only
        r'^MCQ:',  # Just label
        r'^\[A\]',  # Just option
    ]
    
    text_lower = text.lower()
    for pattern in invalid_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False
    
    # Should have at least one question word or end with ?
    question_indicators = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 
                          'describe', 'explain', 'analyze', 'compare', 'define',
                          'identify', 'list', 'discuss', 'evaluate', 'create',
                          'design', 'implement', 'solve', 'calculate', 'write',
                          'given', 'consider', 'suppose', 'imagine']
    
    has_question_word = any(word in text_lower for word in question_indicators)
    ends_with_qmark = text.strip().endswith('?')
    
    return has_question_word or ends_with_qmark


def extract_from_string(value: str, seen: Set[str]) -> List[str]:
    """Extract questions from a string value."""
    questions = []
    
    # Normalize the string
    normalized = normalize_text(value)
    
    # Strategy 1: Try parsing as JSON
    try:
        parsed = json.loads(value)
        questions.extend(extract_from_any(parsed, seen))
        return questions
    except:
        pass
    
    # Strategy 2: Check if it's a valid question directly
    if is_valid_question(normalized):
        if normalized not in seen:
            questions.append(normalized)
            seen.add(normalized)
        return questions
    
    # Strategy 3: Extract from JSON-like structures without parsing
    # Look for patterns like "question": "...", "text": "...", etc.
    json_patterns = [
        r'["\']question["\']\s*:\s*["\']([^"\']+)["\']',
        r'["\']text["\']\s*:\s*["\']([^"\']+)["\']',
        r'["\']Question["\']\s*:\s*["\']([^"\']+)["\']',
        r'["\']description["\']\s*:\s*["\']([^"\']+)["\']',
        r'["\']task["\']\s*:\s*["\']([^"\']+)["\']',
        r'["\']CO1["\']\s*:\s*["\']([^"\']+)["\']',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, value, re.IGNORECASE)
        for match in matches:
            match_clean = normalize_text(match)
            if is_valid_question(match_clean) and match_clean not in seen:
                questions.append(match_clean)
                seen.add(match_clean)
    
    # Strategy 4: Split by common delimiters and check each part
    parts = re.split(r'[\.\?]\s+(?=[A-Z])|\\n|\n', value)
    for part in parts:
        part_clean = normalize_text(part)
        if is_valid_question(part_clean) and part_clean not in seen:
            questions.append(part_clean)
            seen.add(part_clean)
    
    return questions


def extract_from_dict(obj: Dict, seen: Set[str]) -> List[str]:
    """Extract questions from a dictionary."""
    questions = []
    
    # Priority keys that likely contain questions
    priority_keys = ['question', 'text', 'Question', 'Text', 'description', 
                     'Description', 'task', 'Task', 'CO1', 'CO', 'prompt', 'Prompt',
                     'MCQ', 'mcq']
    
    # First, check priority keys
    for key in priority_keys:
        if key in obj:
            value = obj[key]
            if isinstance(value, str):
                extracted = extract_from_string(value, seen)
                questions.extend(extracted)
            elif isinstance(value, dict) and key == 'MCQ' and 'question' in value:
                # Handle MCQ question
                mcq_q = value.get('question', '')
                if is_valid_question(mcq_q) and mcq_q not in seen:
                    questions.append(mcq_q)
                    seen.add(mcq_q)
            elif isinstance(value, (dict, list)):
                questions.extend(extract_from_any(value, seen))
    
    # Then, recursively check all other values
    for key, value in obj.items():
        if key not in priority_keys:
            if isinstance(value, (dict, list)):
                questions.extend(extract_from_any(value, seen))
            elif isinstance(value, str):
                extracted = extract_from_string(value, seen)
                questions.extend(extracted)
    
    return questions


def extract_from_list(items: List, seen: Set[str]) -> List[str]:
    """Extract questions from a list."""
    questions = []
    
    for item in items:
        if isinstance(item, str):
            # Check if entire string is a question
            normalized = normalize_text(item)
            if is_valid_question(normalized) and normalized not in seen:
                questions.append(normalized)
                seen.add(normalized)
            else:
                # Try extracting from string
                extracted = extract_from_string(item, seen)
                questions.extend(extracted)
        elif isinstance(item, (dict, list)):
            questions.extend(extract_from_any(item, seen))
    
    return questions


def extract_from_any(value: Any, seen: Set[str] = None) -> List[str]:
    """Main extraction function that handles any data type."""
    if seen is None:
        seen = set()
    
    questions = []
    
    if isinstance(value, str):
        questions.extend(extract_from_string(value, seen))
    elif isinstance(value, dict):
        questions.extend(extract_from_dict(value, seen))
    elif isinstance(value, list):
        questions.extend(extract_from_list(value, seen))
    
    return questions


def extract_questions_from_co_data(co_data: dict) -> List[str]:
    """Extract questions from a CO data structure. Handles both clean arrays and complex nested formats."""
    questions = []
    seen = set()
    
    # Primary path: Direct questions array (case-insensitive)
    questions_list = co_data.get("questions") or co_data.get("Questions") or []
    
    if questions_list:
        # If questions are already in a clean array format
        for item in questions_list:
            if isinstance(item, str):
                # Clean string question
                normalized = normalize_text(item)
                if normalized and normalized not in seen:
                    questions.append(normalized)
                    seen.add(normalized)
            elif isinstance(item, (dict, list)):
                # Fallback: complex nested structure
                extracted = extract_from_any(item, seen)
                questions.extend(extracted)
            else:
                # Convert to string and try
                item_str = str(item)
                normalized = normalize_text(item_str)
                if is_valid_question(normalized) and normalized not in seen:
                    questions.append(normalized)
                    seen.add(normalized)
    
    # Fallback: If no direct questions array, search all fields
    if not questions:
        for key, value in co_data.items():
            key_lower = key.lower()
            if key_lower not in ['co', 'po', 'topics', 'topics'] and isinstance(value, (str, dict, list)):
                extracted = extract_from_any(value, seen)
                questions.extend(extracted)
    
    # Final cleanup: remove very similar questions (fuzzy deduplication)
    unique_questions = []
    for q in questions:
        if not q or len(q.strip()) < 10:  # Skip very short items
            continue
            
        # Check if this question is too similar to existing ones
        is_duplicate = False
        q_lower = q.lower().strip()
        for existing in unique_questions:
            existing_lower = existing.lower().strip()
            # If 80% similar, consider duplicate
            q_words = set(q_lower.split())
            existing_words = set(existing_lower.split())
            if len(q_words) > 0 and len(existing_words) > 0:
                similarity = len(q_words & existing_words) / max(len(q_words), len(existing_words))
                if similarity > 0.8:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_questions.append(q.strip())
    
    return unique_questions


def format_questions_for_pipeline(questions: List[str]) -> str:
    """Format questions as numbered list for pipeline input."""
    formatted = []
    for i, q in enumerate(questions, 1):
        formatted.append(f"{i}) {q}")
    return "\n".join(formatted)


def process_co_questions(json_file_path: str):
    """Process questions for each CO separately."""
    
    # Load JSON
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": f"File not found: {json_file_path}"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}
    
    # Extract CO definitions and questions
    co_definitions = data[0] if len(data) > 0 else []
    co_questions_data = data[1] if len(data) > 1 else []
    
    if not co_questions_data:
        return {"error": "No CO questions data found in JSON"}
    
    # Initialize pipeline once (reusable)
    pipeline = QuestionPaperValidationPipeline()
    
    results = {
        "co_results": []
    }
    
    # Process each CO
    for idx, co_data in enumerate(co_questions_data):
        # Handle case-insensitive key access
        co_name = co_data.get("CO") or co_data.get("co") or f"CO{idx+1}"
        po_list = co_data.get("PO") or co_data.get("po") or []
        topics_list = co_data.get("topics") or co_data.get("Topics") or []
        
        # Extract questions (handles both clean arrays and complex formats)
        questions = extract_questions_from_co_data(co_data)
        
        if not questions:
            results["co_results"].append({
                "co": co_name,
                "po": po_list,
                "topics": topics_list,
                "total_questions": 0,
                "error": "No valid questions extracted"
            })
            continue
        
        # Format as input text for pipeline (numbered format)
        formatted_input = format_questions_for_pipeline(questions)
        
        # Run pipeline
        try:
            co_result = pipeline.process(formatted_input)
            
            if "error" in co_result:
                results["co_results"].append({
                    "co": co_name,
                    "po": po_list,
                    "topics": topics_list,
                    "total_questions": len(questions),
                    "error": co_result["error"]
                })
                continue
            
            # Store result
            co_summary = {
                "co": co_name,
                "po": po_list,
                "topics": topics_list,
                "total_questions": len(questions),
                "summary": co_result.get("summary", {})
            }
            
            results["co_results"].append(co_summary)
            
        except Exception as e:
            results["co_results"].append({
                "co": co_name,
                "po": po_list,
                "topics": topics_list,
                "total_questions": len(questions) if questions else 0,
                "error": str(e)
            })
            continue
    
    return results


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python process_co_questions.py <json_file_path>")
        print("\nExample:")
        print('python process_co_questions.py "question_paper_output_20260101_132923.json"')
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    results = process_co_questions(json_file)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"co_validation_results_{timestamp}.json"
    
    # Create output folder if it doesn't exist
    output_folder = os.path.join(os.getcwd(), "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save to file in output folder
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also print to stdout
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Print file location
    print(f"\nOutput saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

