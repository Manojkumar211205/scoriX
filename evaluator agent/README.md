# Question Paper Validation Pipeline

Production-ready agentic AI system for validating and analyzing question papers.

## Project Structure

```
agent/
├── agents/                 # AI Agent modules
│   ├── bloom_agent.py      # Bloom's Taxonomy classification
│   ├── difficulty_agent.py # Difficulty assessment
│   ├── embedding_agent.py  # Text embeddings
│   ├── evaluator_agent.py  # Issue detection and evaluation
│   ├── question_reasoning_agent.py # Improvement suggestions
│   ├── repetition_agent.py # Repetition detection
│   ├── segmentation_agent.py # Question segmentation
│   └── topic_agent.py      # Topic classification
│
├── data/                   # Data storage
│   ├── evaluation_memory.json  # Evaluation history
│   ├── question_memory.json    # Question database
│   └── question_index.faiss    # FAISS vector index
│
├── output/                 # Generated results
│   └── co_validation_results_*.json
│
├── config.py               # Configuration settings
├── evaluation_memory.py     # Evaluation memory manager
├── memory_manager.py        # FAISS memory manager
├── pipeline.py              # Main pipeline orchestration
├── process_co_questions.py  # CO-based processing script
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
└── setup_env.py            # Environment setup
```

## Features

- **Multi-Agent System**: Specialized agents for different validation tasks
- **Bloom's Taxonomy Classification**: Automatic cognitive level detection
- **Difficulty Assessment**: Easy/Medium/Hard classification
- **Repetition Detection**: Semantic similarity-based duplicate detection
- **Topic Classification**: Automatic topic and subtopic identification
- **AI-Powered Reasoning**: Improvement suggestions for questions needing changes
- **FAISS Memory**: Efficient vector storage for historical questions
- **CO-Based Processing**: Process questions per Course Outcome

## Usage

### Process CO Questions from JSON

```bash
python process_co_questions.py "path/to/input.json"
```

### Process Single Question Set

```bash
python main.py "1) Question 1\n2) Question 2..."
```

## Output

Results are saved to `output/co_validation_results_TIMESTAMP.json` with:
- Verdict (Valid/Invalid)
- All questions with taxonomy classification
- Reasoning for questions needing changes
- Issues summary
- Questions to change

## Requirements

See `requirements.txt` for dependencies.

## Configuration

Edit `config.py` for API keys and model settings.

