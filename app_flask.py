import json
import os
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agents.questionPaperGeneratorAgent.questionPaperGenerator import QuestionPaperGenerator
from data.events import get_events
from question_paper_formatter import QuestionPaperFormatter

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize Question Paper Generator
qpgen = QuestionPaperGenerator(collectionName="test_ai_collection_v1")

# Initialize Question Paper Formatter
formatter = QuestionPaperFormatter()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_formatted_outputs(output, timestamp, course_name="Course"):
    """
    Save question paper in multiple formats
    
    Args:
        output: Question paper data
        timestamp: Timestamp string for filenames
        course_name: Name of the course
        
    Returns:
        dict: Paths to saved files
    """
    base_filename = f"question_paper_{timestamp}"
    saved_files = {}
    
    # Save JSON (original)
    json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        saved_files['json'] = f"{base_filename}.json"
    except Exception as e:
        print(f"Error saving JSON: {e}")
    
    # Save HTML (formatted)
    try:
        html_content = formatter.format_to_html(output, course_name)
        html_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        saved_files['html'] = f"{base_filename}.html"
    except Exception as e:
        print(f"Error saving HTML: {e}")
    
    # Save TXT (formatted)
    try:
        txt_content = formatter.format_to_text(output, course_name)
        txt_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        saved_files['txt'] = f"{base_filename}.txt"
    except Exception as e:
        print(f"Error saving TXT: {e}")
    
    # Save Markdown (formatted)
    try:
        md_content = formatter.format_to_markdown(output, course_name)
        md_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        saved_files['md'] = f"{base_filename}.md"
    except Exception as e:
        print(f"Error saving Markdown: {e}")
    
    return saved_files


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'ScoriX Question Paper Generator API is running',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/generate': 'Generate question paper from text',
            'POST /api/generate-from-file': 'Generate question paper from uploaded file',
            'GET /api/outputs': 'List all generated outputs',
            'GET /api/outputs/<filename>': 'Download a specific output file',
            'GET /health': 'Health check endpoint'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/generate', methods=['POST'])
def generate_question_paper():
    """
    Generate question paper from text content
    
    Request Body (JSON):
    {
        "text": "Course content text...",
        "collection_name": "optional_collection_name"
    }
    
    Returns:
    {
        "status": "success",
        "data": {...},
        "output_file": "filename.json",
        "timestamp": "..."
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Extract text content
        text = data.get('text', '')
        if not text or not text.strip():
            return jsonify({
                'status': 'error',
                'message': 'Text content is required'
            }), 400
        
        # Optional: Use custom collection name
        collection_name = data.get('collection_name', 'test_ai_collection_v1')
        
        # Initialize generator with specified collection
        generator = QuestionPaperGenerator(collectionName=collection_name)
        
        # Generate question paper
        output = generator.demoQuestionpaperGenerator(text=text, filePath="")
        
        # Save output in multiple formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        course_name = data.get('course_name', 'Course')
        saved_files = save_formatted_outputs(output, timestamp, course_name)
        
        return jsonify({
            'status': 'success',
            'message': 'Question paper generated successfully in multiple formats',
            'data': output,
            'output_files': saved_files,
            'timestamp': timestamp
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating question paper: {str(e)}'
        }), 500


@app.route('/api/generate-from-file', methods=['POST'])
def generate_from_file():
    """
    Generate question paper from uploaded file
    
    Form Data:
    - file: The file to upload
    - collection_name: (optional) Custom collection name
    
    Returns:
    {
        "status": "success",
        "data": {...},
        "output_file": "filename.json",
        "timestamp": "..."
    }
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Optional: Use custom collection name
        collection_name = request.form.get('collection_name', 'test_ai_collection_v1')
        
        # Initialize generator with specified collection
        generator = QuestionPaperGenerator(collectionName=collection_name)
        
        # Generate question paper from file
        output = generator.demoQuestionpaperGenerator(text="", filePath=file_path)
        
        # Save output to file
        output_filename = f"question_paper_output_{timestamp}.json"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # If output is not JSON serializable, save as text
            output_filename = f"question_paper_output_{timestamp}.txt"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(output))
        
        return jsonify({
            'status': 'success',
            'message': 'Question paper generated successfully from file',
            'data': output,
            'uploaded_file': unique_filename,
            'output_file': output_filename,
            'timestamp': timestamp
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating question paper from file: {str(e)}'
        }), 500


@app.route('/api/outputs', methods=['GET'])
def list_outputs():
    """
    List all generated output files
    
    Returns:
    {
        "status": "success",
        "files": [...]
    }
    """
    try:
        files = []
        for filename in os.listdir(app.config['OUTPUT_FOLDER']):
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.isfile(file_path):
                files.append({
                    'filename': filename,
                    'size': os.path.getsize(file_path),
                    'created': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                })
        
        return jsonify({
            'status': 'success',
            'count': len(files),
            'files': files
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error listing outputs: {str(e)}'
        }), 500


@app.route('/api/outputs/<filename>', methods=['GET'])
def download_output(filename):
    """
    Download a specific output file
    
    Returns: File download
    """
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error downloading file: {str(e)}'
        }), 500


@app.route('/api/generate-stream', methods=['POST'])
def generate_question_paper_stream():
    """
    Generate question paper from text content with SSE streaming
    
    Request Body (JSON):
    {
        "text": "Course content text...",
        "collection_name": "optional_collection_name"
    }
    
    Returns: Server-Sent Events stream
    """
    def generate():
        try:
            # Get JSON data from request
            data = request.get_json()
            
            if not data:
                yield f"data: {json.dumps({'event': 'error', 'message': 'No JSON data provided'})}\n\n"
                return
            
            # Extract text content
            text = data.get('text', '')
            if not text or not text.strip():
                yield f"data: {json.dumps({'event': 'error', 'message': 'Text content is required'})}\n\n"
                return
            
            # Optional: Use custom collection name
            collection_name = data.get('collection_name', f'stream_{int(time.time())}')
            
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'message': 'Starting question paper generation', 'task_id': collection_name})}\n\n"
            
            # Start generation in a separate thread
            result_container = {'output': None, 'error': None}
            
            def run_generation():
                try:
                    generator = QuestionPaperGenerator(collectionName=collection_name)
                    output = generator.demoQuestionpaperGenerator(text=text, filePath="")
                    result_container['output'] = output
                except Exception as e:
                    result_container['error'] = str(e)
            
            generation_thread = threading.Thread(target=run_generation)
            generation_thread.start()
            
            # Stream events while generation is running
            last_index = 0
            while generation_thread.is_alive() or result_container['output'] is not None or result_container['error'] is not None:
                # Get new events
                events, next_index = get_events(collection_name, last_index)
                
                # Send each event
                for event in events:
                    yield f"data: {json.dumps({'event': 'thinking', 'message': event['msg']})}\n\n"
                
                last_index = next_index
                
                # Check if generation is complete
                if result_container['output'] is not None:
                    # Save output to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"question_paper_output_{timestamp}.json"
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(result_container['output'], f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        output_filename = f"question_paper_output_{timestamp}.txt"
                        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(str(result_container['output']))
                    
                    yield f"data: {json.dumps({'event': 'complete', 'message': 'Question paper generated successfully', 'output_file': output_filename, 'data': result_container['output']})}\n\n"
                    break
                
                if result_container['error'] is not None:
                    yield f"data: {json.dumps({'event': 'error', 'message': result_container['error']})}\n\n"
                    break
                
                # Small delay to avoid overwhelming the client
                time.sleep(0.5)
            
            generation_thread.join(timeout=1)
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')


@app.route('/api/generate-from-file-stream', methods=['POST'])
def generate_from_file_stream():
    """
    Generate question paper from uploaded file with SSE streaming
    
    Form Data:
    - file: The file to upload
    - collection_name: (optional) Custom collection name
    
    Returns: Server-Sent Events stream
    """
    def generate():
        try:
            # Check if file is present
            if 'file' not in request.files:
                yield f"data: {json.dumps({'event': 'error', 'message': 'No file provided'})}\n\n"
                return
            
            file = request.files['file']
            
            if file.filename == '':
                yield f"data: {json.dumps({'event': 'error', 'message': 'No file selected'})}\n\n"
                return
            
            if not allowed_file(file.filename):
                allowed_types = ", ".join(ALLOWED_EXTENSIONS)
                yield f"data: {json.dumps({'event': 'error', 'message': f'File type not allowed. Allowed types: {allowed_types}'})}\n\n"
                return
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Optional: Use custom collection name
            collection_name = request.form.get('collection_name', f'stream_{int(time.time())}')
            
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'message': 'Starting question paper generation from file', 'task_id': collection_name})}\n\n"
            
            # Start generation in a separate thread
            result_container = {'output': None, 'error': None}
            
            def run_generation():
                try:
                    generator = QuestionPaperGenerator(collectionName=collection_name)
                    output = generator.demoQuestionpaperGenerator(text="", filePath=file_path)
                    result_container['output'] = output
                except Exception as e:
                    result_container['error'] = str(e)
            
            generation_thread = threading.Thread(target=run_generation)
            generation_thread.start()
            
            # Stream events while generation is running
            last_index = 0
            while generation_thread.is_alive() or result_container['output'] is not None or result_container['error'] is not None:
                # Get new events
                events, next_index = get_events(collection_name, last_index)
                
                # Send each event
                for event in events:
                    yield f"data: {json.dumps({'event': 'thinking', 'message': event['msg']})}\n\n"
                
                last_index = next_index
                
                # Check if generation is complete
                if result_container['output'] is not None:
                    # Save output to file
                    output_filename = f"question_paper_output_{timestamp}.json"
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(result_container['output'], f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        output_filename = f"question_paper_output_{timestamp}.txt"
                        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(str(result_container['output']))
                    
                    yield f"data: {json.dumps({'event': 'complete', 'message': 'Question paper generated successfully', 'output_file': output_filename, 'uploaded_file': unique_filename, 'data': result_container['output']})}\n\n"
                    break
                
                if result_container['error'] is not None:
                    yield f"data: {json.dumps({'event': 'error', 'message': result_container['error']})}\n\n"
                    break
                
                # Small delay to avoid overwhelming the client
                time.sleep(0.5)
            
            generation_thread.join(timeout=1)
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("🚀 Starting ScoriX Question Paper Generator API...")
    print("📍 API will be available at: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
