# import os
# import uuid
# import logging
# import torch
# import torch.nn.functional as F
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from typing import List, Union, Optional
#
# # --- Configuration ---
# # (Combined from your config.py)
#
# # Qdrant Configuration
# QDRANT_HOST = "localhost"
# QDRANT_PORT = 6333
# COLLECTION_NAME = "clip_embeddings"
#
# # CLIP Model Configuration
# CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# VECTOR_SIZE = 512  # CLIP base patch32 embedding size
# DISTANCE_METRIC = models.Distance.COSINE
#
# # --- Logging Setup ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
#
# # --- Component 1: CLIP Embedder ---
# # (From your clip_embedder.py)
#
# class UnifiedCLIPEmbedder:
#     """
#     Manages loading the CLIP model and creating embeddings
#     for both images and text.
#     """
#
#     def __init__(self, model_name: str = CLIP_MODEL_NAME):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {self.device}")
#
#         logger.info(f"Loading CLIP model: {model_name}...")
#         self.model = CLIPModel.from_pretrained(model_name).to(self.device)
#         self.processor = CLIPProcessor.from_pretrained(model_name)
#         self.model.eval()  # Set to evaluation mode
#         logger.info("CLIP model loaded successfully!")
#
#     def get_image_embedding(self, image_path: str):
#         """Get a single image embedding"""
#         try:
#             image = Image.open(image_path).convert('RGB')
#             inputs = self.processor(images=image, return_tensors="pt").to(self.device)
#
#             with torch.no_grad():
#                 image_features = self.model.get_image_features(**inputs)
#                 image_embedding = F.normalize(image_features, p=2, dim=1)
#
#             return image_embedding.cpu().numpy().flatten()
#         except Exception as e:
#             logger.error(f"Error processing image {image_path}: {str(e)}")
#             raise
#
#     def get_text_embedding(self, text: str):
#         """Get a single text embedding"""
#         try:
#             inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
#
#             with torch.no_grad():
#                 text_features = self.model.get_text_features(**inputs)
#                 text_embedding = F.normalize(text_features, p=2, dim=1)
#
#             return text_embedding.cpu().numpy().flatten()
#         except Exception as e:
#             logger.error(f"Error processing text: {str(e)}")
#             raise
#
#     def batch_process_images(self, image_paths: List[str]):
#         """Process multiple images in batch for efficiency"""
#         try:
#             images = [Image.open(path).convert('RGB') for path in image_paths]
#             inputs = self.processor(images=images, return_tensors="pt").to(self.device)
#
#             with torch.no_grad():
#                 image_features = self.model.get_image_features(**inputs)
#                 image_embeddings = F.normalize(image_features, p=2, dim=1)
#
#             return image_embeddings.cpu().numpy()  # Returns 2D numpy array
#         except Exception as e:
#             logger.error(f"Error batch processing images: {str(e)}")
#             raise
#
#
# # --- Component 2: Qdrant Manager ---
# # (From your qdrant_manager.py)
#
# class QdrantManager:
#     """
#     Manages all interactions with the Qdrant vector database.
#     """
#
#     def __init__(self, host: str, port: int, collection_name: str, vector_size: int, distance: str):
#         self.host = host
#         self.port = port
#         self.collection_name = collection_name
#         self.vector_size = vector_size
#         self.distance = distance
#         self.client = QdrantClient(host=self.host, port=self.port)
#
#     def create_collection(self):
#         """Create collection if it doesn't exist"""
#         try:
#             self.client.get_collection(self.collection_name)
#             logger.info(f"Collection '{self.collection_name}' already exists")
#         except Exception:
#             self.client.create_collection(
#                 collection_name=self.collection_name,
#                 vectors_config=models.VectorParams(
#                     size=self.vector_size,
#                     distance=self.distance
#                 )
#             )
#             logger.info(f"Collection '{self.collection_name}' created successfully")
#
#     def store_embedding(self, vector: List[float], payload: dict) -> str:
#         """Store a single vector with its payload"""
#         point_id = str(uuid.uuid4())
#
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=[
#                 models.PointStruct(
#                     id=point_id,
#                     vector=vector,
#                     payload=payload
#                 )
#             ]
#         )
#         return point_id
#
#     def batch_store_embeddings(self, vectors: List[List[float]], payloads: List[dict]) -> List[str]:
#         """Store a batch of vectors and payloads"""
#         point_ids = [str(uuid.uuid4()) for _ in vectors]
#
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=[
#                 models.PointStruct(
#                     id=point_id,
#                     vector=vector,
#                     payload=payload
#                 )
#                 for point_id, vector, payload in zip(point_ids, vectors, payloads)
#             ]
#         )
#         return point_ids
#
#     def search_similar(self, query_embedding: List[float], limit: int = 5):
#         """Search for similar vectors"""
#         search_result = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=query_embedding,
#             limit=limit,
#             with_payload=True,
#             with_vectors=False
#         )
#         return search_result
#
#     def get_total_points(self) -> int:
#         """Get total number of points in collection"""
#         try:
#             info = self.client.get_collection(self.collection_name)
#             return info.vectors_count if info else 0
#         except Exception:
#             return 0
#
#
# # --- Main Service Class ---
# # (This is the new "single class" you requested)
#
# class ClipSearchService:
#     """
#     A unified service for indexing images and searching them
#     using text or image queries.
#     """
#
#     def __init__(self,collection_name):
#         logger.info("Initializing ClipSearchService...")
#         self.embedder = UnifiedCLIPEmbedder(model_name=CLIP_MODEL_NAME)
#         self.db_manager = QdrantManager(
#             host=QDRANT_HOST,
#             port=QDRANT_PORT,
#             collection_name=collection_name,
#             vector_size=VECTOR_SIZE,
#             distance=DISTANCE_METRIC
#         )
#
#         # Ensure the collection exists on startup
#         self.db_manager.create_collection()
#         logger.info("ClipSearchService initialized successfully.")
#
#     def index_images(self, image_paths: Union[str, List[str]], metadata: Optional[List[dict]] = None):
#         """
#         Embeds and stores one or more images in the vector database.
#
#         Args:
#             image_paths: A single file path (str) or a list of file paths.
#             metadata: (Optional) A list of metadata dictionaries, one for each image.
#                       If not provided, basic metadata will be created.
#         """
#         if isinstance(image_paths, str):
#             image_paths = [image_paths]  # Handle single image case
#
#         logger.info(f"Indexing {len(image_paths)} images...")
#
#         # 1. Embed all images in a batch
#         embeddings = self.embedder.batch_process_images(image_paths)
#
#         # 2. Prepare payloads
#         payloads = []
#         for i, path in enumerate(image_paths):
#             # Use provided metadata if available, else create default
#             meta = (metadata[i] if metadata and len(metadata) == len(image_paths) else {}).copy()
#
#             meta.setdefault("image_path", path)
#             meta.setdefault("source", "CLIP_embedding")
#             payloads.append(meta)
#
#         # 3. Store in Qdrant in a batch
#         # Note: Qdrant client expects vectors as standard lists, not numpy arrays
#         vectors_list = [emb.tolist() for emb in embeddings]
#         point_ids = self.db_manager.batch_store_embeddings(vectors_list, payloads)
#
#         return point_ids
#
#     def search_by_text(self, text_query: str, limit: int = 5):
#         """
#         Embeds a text query and retrieves the most similar images.
#         """
#         logger.info(f"Searching for text: '{text_query}'")
#
#         # 1. Embed the text query
#         query_embedding = self.embedder.get_text_embedding(text_query)
#
#         # 2. Search in Qdrant
#         # Convert numpy array to list for Qdrant
#         results = self.db_manager.search_similar(query_embedding.tolist(), limit=limit)
#
#         logger.info(f"Found {len(results)} results for text query.")
#         return results
#
#     def search_by_image(self, image_path: str, limit: int = 5):
#         """
#         Embeds an image query and retrieves the most similar images.
#         """
#         logger.info(f"Searching by image: '{image_path}'")
#
#         # 1. Embed the image query
#         query_embedding = self.embedder.get_image_embedding(image_path)
#
#         # 2. Search in Qdrant
#         # Convert numpy array to list for Qdrant
#         results = self.db_manager.search_similar(query_embedding.tolist(), limit=limit)
#
#         logger.info(f"Found {len(results)} results for image query.")
#         return results
#
#     def get_index_stats(self):
#         """Returns the total number of items in the index."""
#         count = self.db_manager.get_total_points()
#         return {"total_images": count}
#
#
# # --- Example Usage ---
#
# if __name__ == "__main__":
#     # 1. Create test image
#
#
#     # 2. Initialize service
#     service = ClipSearchService("testDB")
#
#     # 3. Store the image
#     service.index_images(["img.png"])
#     logger.info("Image stored in vector database")
#
#     # 4. Search with text query
#     test_query = "water bottel"
#     results = service.search_by_text(test_query, limit=3)
#
#     # 5. Display results
#     print(f"\n🔍 Search: '{test_query}'")
#     for i, result in enumerate(results):
#         print(f"#{i+1}: Score={result.score:.3f}, File={result.payload.get('image_path')}")

import os
import uuid
import logging
import torch
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Union, Optional

# Install required package for OpenCLIP
# pip install open_clip_torch

import open_clip

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "clip_embeddings"

# Updated to OpenCLIP ViT-B/16 - More powerful model
OPENCLIP_MODEL_NAME = "ViT-B-16"
OPENCLIP_PRETRAINED = "laion2b_s34b_b88k"  # Trained on 2B high-quality images
VECTOR_SIZE = 512  # Same size but much better quality
DISTANCE_METRIC = models.Distance.COSINE

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Component 1: Powerful OpenCLIP Embedder ---

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

    def get_text_embedding(self, text: str):
        """Get a single text embedding using OpenCLIP"""
        try:
            text_tokens = self.tokenizer([text]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_embedding = F.normalize(text_features, p=2, dim=1)

            return text_embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
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

    def get_multimodal_similarity(self, image_path: str, text: str):
        """Get direct similarity score between image and text"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            text_tokens = self.tokenizer([text]).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                # Calculate cosine similarity
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                similarity = (image_features @ text_features.T).item()

            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise


# --- Component 2: Qdrant Manager ---

class QdrantManager:
    """
    Manages all interactions with the Qdrant vector database.
    """

    def __init__(self, host: str, port: int, collection_name: str, vector_size: int, distance: str):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.client = QdrantClient(host=self.host, port=self.port)

    def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")

    def store_embedding(self, vector: List[float], payload: dict) -> str:
        """Store a single vector with its payload"""
        point_id = str(uuid.uuid4())

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        return point_id

    def batch_store_embeddings(self, vectors: List[List[float]], payloads: List[dict]) -> List[str]:
        """Store a batch of vectors and payloads"""
        point_ids = [str(uuid.uuid4()) for _ in vectors]

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(point_ids, vectors, payloads)
            ]
        )
        return point_ids

    def search_similar(self, query_embedding: List[float], limit: int = 5):
        """Search for similar vectors"""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return search_result

    def get_total_points(self) -> int:
        """Get total number of points in collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.vectors_count if info else 0
        except Exception:
            return 0


# --- Main Service Class ---

class ClipSearchService:
    """
    A unified service for indexing images and searching them
    using text or image queries with the more powerful OpenCLIP model.
    """

    def __init__(self, collection_name: str = COLLECTION_NAME):
        logger.info("🚀 Initializing ClipSearchService with OpenCLIP ViT-B/16...")
        self.embedder = PowerfulCLIPEmbedder()
        self.db_manager = QdrantManager(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=collection_name,
            vector_size=VECTOR_SIZE,
            distance=DISTANCE_METRIC
        )

        # Ensure the collection exists on startup
        self.db_manager.create_collection()
        logger.info("✅ ClipSearchService with OpenCLIP initialized successfully.")

    def index_images(self, image_paths: Union[str, List[str]], metadata: Optional[List[dict]] = None):
        """
        Embeds and stores one or more images in the vector database.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]  # Handle single image case

        logger.info(f"Indexing {len(image_paths)} images with OpenCLIP...")

        # 1. Embed all images in a batch
        embeddings = self.embedder.batch_process_images(image_paths)

        # 2. Prepare payloads
        payloads = []
        for i, path in enumerate(image_paths):
            # Use provided metadata if available, else create default
            meta = (metadata[i] if metadata and len(metadata) == len(image_paths) else {}).copy()

            meta.setdefault("image_path", path)
            meta.setdefault("filename", os.path.basename(path))
            meta.setdefault("source", "OpenCLIP_embedding")
            meta.setdefault("model", "OpenCLIP-ViT-B-16")
            payloads.append(meta)

        # 3. Store in Qdrant in a batch
        vectors_list = [emb.tolist() for emb in embeddings]
        point_ids = self.db_manager.batch_store_embeddings(vectors_list, payloads)

        logger.info(f"✅ Successfully indexed {len(point_ids)} images")
        return point_ids

    def search_by_text(self, text_query: str, limit: int = 5):
        """
        Embeds a text query and retrieves the most similar images.
        """
        logger.info(f"🔍 Searching for text: '{text_query}'")

        # 1. Embed the text query using OpenCLIP
        query_embedding = self.embedder.get_text_embedding(text_query)

        # 2. Search in Qdrant
        results = self.db_manager.search_similar(query_embedding.tolist(), limit=limit)

        logger.info(f"✅ Found {len(results)} results for text query.")
        return results

    def search_by_image(self, image_path: str, limit: int = 5):
        """
        Embeds an image query and retrieves the most similar images.
        """
        logger.info(f"🔍 Searching by image: '{image_path}'")

        # 1. Embed the image query using OpenCLIP
        query_embedding = self.embedder.get_image_embedding(image_path)

        # 2. Search in Qdrant
        results = self.db_manager.search_similar(query_embedding.tolist(), limit=limit)

        logger.info(f"✅ Found {len(results)} results for image query.")
        return results

    def get_direct_similarity(self, image_path: str, text: str) -> float:
        """
        Get direct similarity score between an image and text without database search.
        Useful for testing model understanding.
        """
        return self.embedder.get_multimodal_similarity(image_path, text)

    def get_index_stats(self):
        """Returns the total number of items in the index."""
        count = self.db_manager.get_total_points()
        return {"total_images": count, "model": "OpenCLIP-ViT-B-16"}


# --- Example Usage ---
#
# if __name__ == "__main__":
#     # 1. Initialize service with powerful OpenCLIP
#     service = ClipSearchService("testDB")
#
#     # 2. Store the image
#     service.index_images(["img_1.png"])
#     logger.info("Image stored in vector database")
#
#     # 3. Search with text query
#     test_query = "butterfly life cycle"
#     results = service.search_by_text(test_query, limit=3)
#
#     # 4. Display results
#     print(f"\n🔍 Search: '{test_query}'")
#     print("=" * 50)
#     for i, result in enumerate(results):
#         print(f"#{i + 1}: Score={result.score:.3f}, File={result.payload.get('filename')}")
#
#     # 5. Test direct similarity (optional)
#     print(f"\n🎯 Testing direct similarity:")
#     similarity = service.get_direct_similarity("img.png", test_query)
#     print(f"Direct similarity score: {similarity:.3f}")
#
#     # 6. Show stats
#     stats = service.get_index_stats()
#     print(f"\n📊 Database stats: {stats}")