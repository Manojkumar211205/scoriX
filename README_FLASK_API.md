# ScoriX Question Paper Generator - Flask API

A Flask REST API for generating question papers from course content using the ScoriX Question Paper Generator.

## Features

- ✅ Generate question papers from text content
- ✅ Generate question papers from uploaded files (txt, pdf, doc, docx)
- ✅ List all generated outputs
- ✅ Download generated question papers
- ✅ CORS enabled for cross-origin requests
- ✅ Automatic file saving with timestamps

## Installation

1. Install dependencies:
```bash
pip install -r requirements_flask.txt
```

2. Run the API:
```bash
python app_flask.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Health Check
```
GET /
GET /health
```

**Response:**
```json
{
  "status": "success",
  "message": "ScoriX Question Paper Generator API is running",
  "version": "1.0.0"
}
```

### 2. Generate Question Paper from Text
```
POST /api/generate
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Course: Programming Fundamentals\n\nCourse Outcomes:\nCO1: Understand basic programming concepts...",
  "collection_name": "optional_collection_name"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Question paper generated successfully",
  "data": { ... },
  "output_file": "question_paper_output_20260108_093000.json",
  "timestamp": "20260108_093000"
}
```

### 3. Generate Question Paper from File
```
POST /api/generate-from-file
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: The file to upload (txt, pdf, doc, docx)
- `collection_name`: (optional) Custom collection name

**Response:**
```json
{
  "status": "success",
  "message": "Question paper generated successfully from file",
  "data": { ... },
  "uploaded_file": "20260108_093000_course_content.txt",
  "output_file": "question_paper_output_20260108_093000.json",
  "timestamp": "20260108_093000"
}
```

### 4. List All Generated Outputs
```
GET /api/outputs
```

**Response:**
```json
{
  "status": "success",
  "count": 5,
  "files": [
    {
      "filename": "question_paper_output_20260108_093000.json",
      "size": 12345,
      "created": "2026-01-08T09:30:00"
    }
  ]
}
```

### 5. Download Output File
```
GET /api/outputs/<filename>
```

**Response:** File download

## Usage Examples

### Using cURL

#### Generate from text:
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Course: Programming Fundamentals\n\nCourse Outcomes:\nCO1: Understand basic programming concepts..."
  }'
```

#### Generate from file:
```bash
curl -X POST http://localhost:5000/api/generate-from-file \
  -F "file=@course_content.txt" \
  -F "collection_name=my_collection"
```

#### List outputs:
```bash
curl http://localhost:5000/api/outputs
```

#### Download output:
```bash
curl -O http://localhost:5000/api/outputs/question_paper_output_20260108_093000.json
```

### Using Python Requests

```python
import requests

# Generate from text
url = "http://localhost:5000/api/generate"
data = {
    "text": """
    Course: Programming Fundamentals
    
    Course Outcomes:
    CO1: Understand basic programming concepts...
    """
}
response = requests.post(url, json=data)
print(response.json())

# Generate from file
url = "http://localhost:5000/api/generate-from-file"
files = {'file': open('course_content.txt', 'rb')}
response = requests.post(url, files=files)
print(response.json())

# List outputs
url = "http://localhost:5000/api/outputs"
response = requests.get(url)
print(response.json())
```

### Using JavaScript (Fetch API)

```javascript
// Generate from text
fetch('http://localhost:5000/api/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Course: Programming Fundamentals\n\nCourse Outcomes:\nCO1: ...'
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Generate from file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/api/generate-from-file', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Configuration

- **Max File Size:** 16MB
- **Allowed File Types:** txt, pdf, doc, docx
- **Upload Folder:** `uploads/`
- **Output Folder:** `outputs/`
- **Default Collection:** `test_ai_collection_v1`

## Error Handling

All endpoints return standardized error responses:

```json
{
  "status": "error",
  "message": "Error description"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (missing or invalid data)
- `404`: Not Found
- `500`: Internal Server Error

## Notes

- All generated files are saved with timestamps to prevent overwrites
- The API automatically creates `uploads/` and `outputs/` folders if they don't exist
- CORS is enabled for all origins (configure in production as needed)
- Files are saved as JSON when possible, otherwise as text files
