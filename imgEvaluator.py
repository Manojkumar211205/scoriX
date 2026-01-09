import os
import uuid
import logging
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union, Optional
from PIL import Image  # Ensure this import is at the top if not already present
import google.generativeai as genai
# Install required package for OpenCLIP
# pip install open_clip_torch

import open_clip
from openpyxl.styles.builtins import output

# --- Configuration ---
OPENCLIP_MODEL_NAME = "ViT-B-16"
OPENCLIP_PRETRAINED = "laion2b_s34b_b88k"
VECTOR_SIZE = 512
DISTANCE_METRIC = "cosine"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Component: Powerful OpenCLIP Embedder ---

class PowerfulCLIPEmbedder:
    """
    Uses OpenCLIP ViT-B/16 trained on LAION-2B for better performance
    """

    def __init__(self, model_name: str = OPENCLIP_MODEL_NAME, pretrained: str = OPENCLIP_PRETRAINED):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading OpenCLIP model: {model_name} with {pretrained}...")

        # Load OpenCLIP model - better than original CLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()  # Set to evaluation mode

        logger.info("✅ OpenCLIP ViT-B/16 loaded successfully! (Better than original CLIP)")

    def get_image_embedding(self, image_path: str):
        """Get a single image embedding using OpenCLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_embedding = F.normalize(image_features, p=2, dim=1)

            return image_embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def batch_process_images(self, image_paths: List[str]):
        """Process multiple images in batch for efficiency"""
        try:
            # Process images individually to handle different sizes
            processed_images = []
            for path in image_paths:
                image = Image.open(path).convert('RGB')
                processed_image = self.preprocess(image)
                processed_images.append(processed_image)

            # Stack all images into a batch
            image_batch = torch.stack(processed_images).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_batch)
                image_embeddings = F.normalize(image_features, p=2, dim=1)

            return image_embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error batch processing images: {str(e)}")
            raise

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        import numpy as np
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


# --- Main Function: Find Most Similar Image ---
def find_most_similar_image(reference_image_path: str, image_list: List[str], embedder: Optional[PowerfulCLIPEmbedder] = None) :
    """
    Find the image from the list that has the highest similarity to the reference image.

    Args:
        reference_image_path: Path to the reference image.
        image_list: List of image paths to compare against.
        embedder: Optional pre-initialized embedder instance. If None, creates a new one.

    Returns:
        The path of the most similar image from the list.
    """
    if not image_list:
        raise ValueError("Image list cannot be empty.")

    # Initialize embedder if not provided
    if embedder is None:
        embedder = PowerfulCLIPEmbedder()

    # Get reference embedding
    reference_embedding = embedder.get_image_embedding(reference_image_path)
    logger.info(f"✅ Reference image embedding generated: {reference_image_path}")

    # Get embeddings for all images in the list
    candidate_embeddings = embedder.batch_process_images(image_list)
    logger.info(f"✅ Candidate embeddings generated for {len(image_list)} images")

    # Calculate similarities
    similarities = []
    for i, candidate_emb in enumerate(candidate_embeddings):
        sim_score = embedder.cosine_similarity(reference_embedding, candidate_emb)
        similarities.append((image_list[i], sim_score))
        logger.info(f"Similarity for {image_list[i]}: {sim_score:.4f}")

    # Find the maximum similarity
    most_similar_image, max_similarity = max(similarities, key=lambda x: x[1])
    if max_similarity < 0.4:
        return None
    logger.info(f"🎯 Most similar image: {most_similar_image} (score: {max_similarity:.4f})")

    return most_similar_image


def get_images_for_questions(question_numbers: set,questions) -> list:
    """
    Retrieve all image paths from imageDetails for the given set of question numbers.

    Args:
        question_numbers: Set of integers representing question numbers.

    Returns:
        List of all image paths (flattened from all matching questions).
    """
    all_images = []
    for q in questions:
        if q["question_number"] in question_numbers:
            all_images.extend(list(q["imageDetails"]))
    return all_images


def find_question_for_image(image_path: str,questions) -> int:
    """
    Find the question number that contains the given image path in its imageDetails.

    Args:
        image_path: The image file path (e.g., 'gsgfs.jpg').

    Returns:
        The question number (int) if found, else None.
    """
    for q in questions:
        if image_path in q["imageDetails"]:
            return q["question_number"]
    return None


def any_question_requires_image(question_numbers: set,questions) -> bool:
    """
    Check if any of the given question numbers require an image based on the 'imageRequired' key.

    Args:
        question_numbers: Set of integers representing question numbers.

    Returns:
        True if any question requires an image, else False.
    """
    for q in questions:
        if q["question_number"] in question_numbers and q["imageRequired"]:
            return True
    return False
# --- Example Usage ---


def geminiLLMInterface(handwritten_image_path: str, reference_image_path: str) -> str:
    """
    Generates LLM response for evaluating handwritten/drawn exam answer against reference.
    Uses a generalized static prompt for comparison and verdict across any subject.
    :param handwritten_image_path: Path to the student's handwritten/drawn exam image.
    :param reference_image_path: Path to the reference answer image for comparison.
    :return: Cleaned text output (evaluation verdict from Gemini).
    """
    GOOGLE_GEMINI_API_KEY = "AIzaSyAoCDiwD_N6ftUhXzfeXZCT9n5jyxA6Au4"

    if not GOOGLE_GEMINI_API_KEY:
        raise ValueError("❌ GOOGLE_GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")

    # Configure Gemini API once during initialization
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

    # Load a multimodal model (handles both text + image)
    model = genai.GenerativeModel("gemini-2.5-flash")  # Stable, current version
    static_prompt = """
    You are an expert evaluator for handwritten or manually drawn exam answers across any subject.

    Image 1 is the student's handwritten/drawn answer from the exam paper.
    Image 2 is the reference/expected correct answer for comparison.

    Compare the handwritten/drawn answer in Image 1 with the reference answer in Image 2.

    Evaluate based on:
    - Content accuracy: Does it correctly capture the key concepts, facts, or elements?
    - Completeness: Are all essential points, steps, or components included?
    - Structure and clarity: Is it well-organized, legible, and visually clear (for drawings/diagrams)?
    - Depth and correctness: Does it demonstrate proper understanding and execution?

    Provide:
    - A score out of the full marks (assume 10 marks unless specified otherwise).
    - Brief justification for the score.
    - Suggestions for improvement if score < 8.

    Be fair, objective, and constructive. Focus on both textual and visual elements in drawings.
    Output in this format:
    Score: X/10
    Justification: [Explanation]
    Improvements: [If applicable, otherwise "N/A"]
    """

    try:
        handwritten_image = Image.open(handwritten_image_path)
        reference_image = Image.open(reference_image_path)

        contents = [static_prompt, handwritten_image, reference_image]
        response = model.generate_content(contents)

        return response.text.strip()

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return ""


def geminiDiagramInference(diagram_image_path: str) -> str:
    """
    Generates LLM response to explain and analyze a diagram image for flaws.
    Uses a static prompt for description, explanation, and verdict on correctness.
    :param diagram_image_path: Path to the diagram image to analyze.
    :return: Cleaned text output (explanation and verdict from Gemini).
    """
    GOOGLE_GEMINI_API_KEY = "AIzaSyAoCDiwD_N6ftUhXzfeXZCT9n5jyxA6Au4"

    if not GOOGLE_GEMINI_API_KEY:
        raise ValueError("❌ GOOGLE_GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")

    # Configure Gemini API once during initialization
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

    # Load a multimodal model (handles both text + image)
    model = genai.GenerativeModel("gemini-2.5-flash")  # Stable, current version
    static_prompt = """
    You are an expert diagram analyzer for educational content, such as exam answers or technical illustrations.

    Analyze the diagram in the provided image.

    Provide:
    - A clear explanation/description of what the diagram depicts.
    - Key components and how they interconnect or function.
    - Evaluation of correctness: Is it accurate, complete, and well-structured?
    - Any flaws or errors identified: Mention missing elements, inaccuracies, poor labeling, or conceptual mistakes.
    - Overall verdict: A score out of 10, with justification.

    Be objective, detailed, and constructive. If no flaws, state so explicitly.
    Output in this format:
    Description: [Detailed explanation of the diagram]
    Key Components: [List or describe main parts]
    Correctness Evaluation: [Assessment of accuracy and completeness]
    Flaws/Errors: [List any issues, or "None identified"]
    Verdict: Score X/10 - [Brief justification]
    """

    try:
        diagram_image = Image.open(diagram_image_path)

        contents = [static_prompt, diagram_image]
        response = model.generate_content(contents)

        return response.text.strip()

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return ""


def get_questions_requiring_image_given_set(question_numbers: set,questions) -> list:
    """
    Retrieve a list of question numbers from the given set that require an image based on the 'imageRequired' key.

    Args:
        question_numbers: Set of integers representing question numbers.

    Returns:
        List of integers representing question numbers from the input set that require images.
    """
    requiring_image = []
    for q in questions:
        if q["question_number"] in question_numbers and q["imageRequired"]:
            requiring_image.append(q["question_number"])
    return requiring_image


def imgEvaluator(question_numbers, questions,image_path ):
    if not any_question_requires_image(question_numbers, questions):
        return
    else :
        referenceImagePaths = get_images_for_questions(question_numbers, questions)
        if referenceImagePaths:
            referenceImagePath = find_most_similar_image(image_path ,referenceImagePaths)
            if (referenceImagePath):
                questionNumber = find_question_for_image(referenceImagePath,questions)
                verdict = geminiLLMInterface(image_path, referenceImagePath)
                response = [{"question_number": questionNumber,"diagram_evaluation" : verdict}]
                return response
            else:
                verdict = geminiDiagramInference(image_path)
                questionNeedImage = get_questions_requiring_image_given_set(question_numbers,questions)
                output = []
                for q in questionNeedImage:
                   response = {"question_number": q , "diagram_evaluation" : verdict}
                   output.append(response)
                return output

        else :
            verdict = geminiDiagramInference(image_path)
            questionNeedImage = get_questions_requiring_image_given_set(question_numbers, questions)
            output = []
            for q in questionNeedImage:
                response = {"question_number": q, "diagram_evaluation": verdict}
                output.append(response)
            return  output

questions = [
  {
    "question_number": 1,
    "question": "Define the Perceptron in the context of neural networks.",
    "marks": 2,
    "type": "Descriptive",
    "answer": "The Perceptron is a foundational single-layer neural network model used for supervised binary classification. It processes inputs by weighting them, summing the results, and applying an activation function, typically a step function, to generate an output. The Perceptron learning rule adjusts weights to minimize errors, updating them to reduce classification mistakes. As a linear classifier, it can only handle linearly separable problems but serves as the basis for more complex neural networks.",
    "imageRequired": False,
    "imageDetails": {}
  },
  {
    "question_number": 15,
    "question": "Discuss the fundamental idea behind the Actor-Critic method in reinforcement learning.",
    "marks": 5,
    "type": "Descriptive",
    "answer": "The Actor-Critic method in reinforcement learning combines two key components: the actor and the critic. The actor is responsible for selecting actions based on the current state, with the goal of maximizing the expected cumulative reward. It represents a policy that determines the optimal action to take. The critic, on the other hand, evaluates these actions by estimating the expected future rewards, providing feedback to the actor. This feedback helps the actor refine its policy over time. Simultaneously, the critic uses the information from the actor's actions to improve its value function, which assesses the desirability of states or actions. Together, the actor and critic iteratively enhance both the policy and the value function, leading to improved decision-making and learning in the environment. This collaborative process allows the system to optimize its actions and predictions effectively.",
    "imageRequired": True,
    "imageDetails": {
        "D:\scoriX_agent\images\Draw_the_structure_of_a_plant__0.jpg",
        "D:\scoriX_agent\images\Draw_the_structure_of_a_plant__1.jpg",
        "D:\scoriX_agent\images\Draw_the_structure_of_a_plant__2.jpg",
        "D:\scoriX_agent\images\img.png",
        "D:\scoriX_agent\images\img_1.png"
        # Add more paths as needed
    }
  },]
out = imgEvaluator({1,15},questions,"D:\scoriX_agent\img_2.png")
print(out)
