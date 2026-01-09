import uuid
import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

from agents.answerKeyAgent.answerKeyAgent import answerKeyAgent
from agents.answerKeyAgent.documentProcessor import documentProcessor  # If needed for processing docs
from ragSystems.imageRag import ClipSearchService  # If needed for indexing images

app = Flask(__name__)
CORS(app)
# Configuration
UPLOAD_FOLDER = 'D:\scoriX_agent\data'
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'question_papers')
DOC_FOLDER = os.path.join(UPLOAD_FOLDER, 'answer_keys')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(DOC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
ALLOWED_DOC_EXTENSIONS = {'pdf', 'docx', 'pptx', 'txt'}

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_doc(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DOC_EXTENSIONS

def save_images(files):
    """
    Save image files from request and return list of paths.
    Assumes 'images' key in request.files with multiple files.
    """
    if 'images' not in request.files:
        return []
    image_files = request.files.getlist('images')
    image_paths = []
    for file in image_files:
        if file.filename == '':
            continue
        if not allowed_image(file.filename):
            continue  # Or raise error
        filename = secure_filename(file.filename)
        filepath = os.path.join(IMAGE_FOLDER, filename)
        file.save(filepath)
        image_paths.append(filepath)
    return image_paths

def save_docs(files):
    """
    Save document files from request and return list of paths.
    Assumes 'docs' key in request.files with multiple files.
    Processes them using documentProcessor if needed.
    """
    if 'docs' not in request.files:
        return []
    doc_files = request.files.getlist('docs')
    doc_paths = []
    # Optional: Initialize processor for indexing if needed
    # processor = documentProcessor(collection_name="some_collection")
    for file in doc_files:
        if file.filename == '':
            continue
        if not allowed_doc(file.filename):
            continue  # Or raise error
        filename = secure_filename(file.filename)
        filepath = os.path.join(DOC_FOLDER, filename)
        file.save(filepath)
        doc_paths.append(filepath)
        # Optional: Process immediately
        # processor.processor(filepath)
    return doc_paths

def save_ordered_images(image_files):
    """
    Save image files in the order received, with indexed filenames to preserve order.
    Returns list of paths in the original order.
    """
    image_paths = []
    for idx, file in enumerate(image_files):
        if file.filename == '':
            continue
        if not allowed_image(file.filename):
            continue  # Skip invalid, but keep order for others
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        # Append index to preserve order, even if names collide
        new_filename = f"{name}_{idx}{ext}"
        filepath = os.path.join("D:\\scoriX_agent\\data\\answer_sheets", new_filename)
        file.save(filepath)
        image_paths.append(filepath)
    return image_paths
@app.route('/process', methods=['POST'])
def process():
    """
    Flask endpoint to receive images and docs, save them, process with answerKeyAgent,
    and return response as JSON.
    Expects multipart form with 'images' and/or 'docs' fields containing files.
    """
    print("hi")
    try:
        info_json = request.form.get('info_json')
        extra_data = None
        session_id = uuid.uuid4().hex
        image_paths = save_images(request.files)
        if info_json:
            try:
                extra_data = json.loads(info_json)
                agent = answerKeyAgent(uniqID=session_id,notesProvided=False)
                response = agent.answerKeyAgent(questionPaperPaths=image_paths,simpleAnswerSheet=True)
                return jsonify(response)

            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in info_json"}), 400


        doc_paths = save_docs(request.files)

        if not image_paths and not doc_paths:
            return jsonify({"error": "No valid files provided"}), 400


        has_docs = len(doc_paths) > 0

        # Initialize agent based on doc availability


        # Run processing accordingly
        if has_docs:
            agent = answerKeyAgent(session_id, has_docs)
            response = agent.answerKeyAgent(image_paths, doc_paths)
        else:
            agent = answerKeyAgent(session_id,False)
            response = agent.answerKeyAgent(image_paths)

        return jsonify({
            "session_id": session_id,
            "has_docs": has_docs,
            "response": response,
            "image_paths": image_paths,
            "doc_paths": doc_paths
        }), 200

    except Exception as e:
        print("Error processing:", e)
        return jsonify({
            "error": str(e),

        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
