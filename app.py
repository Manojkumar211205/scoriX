import os
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from openai import OpenAI
import time
from openai import OpenAIError, APIError, AuthenticationError, RateLimitError

class LLMInterface:
    def __init__(self):
        # Load .env file to read API keys
        load_dotenv()

        # Fetch Gemini API Key
        self.GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

        if not self.GOOGLE_GEMINI_API_KEY:
            raise ValueError("❌ GOOGLE_GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")

        # Configure Gemini API once during initialization
        genai.configure(api_key=self.GOOGLE_GEMINI_API_KEY)

        # Load a multimodal model (handles both text + image)
        self.key_manager = APIKeyManager(
            [os.getenv("nvidiaKey1"), os.getenv("nvidiaKey2"), os.getenv("nvidiaKey3"), os.getenv("nvidiaKey4")])

        self.model = genai.GenerativeModel("gemini-2.5-flash")  # Stable, current version
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key= self.key_manager.get_key()
        )

        self.client = None
        self._update_client()

    # Initialize the client once

    def _update_client(self):
        self.client = OpenAI(api_key=self.key_manager.get_key(), base_url="https://integrate.api.nvidia.com/v1")

    def nvidiaResponse(self, prompt: str,
                       model: str = "deepseek-ai/deepseek-r1-distill-qwen-14b",
                       temperature: float = 0.6, top_p: float = 0.7,
                       max_tokens: int = 4096, max_retries: int = 3) -> str:

        for attempt in range(max_retries * len(self.key_manager.keys)):
            try:
                response_text = ""

                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=True
                )

                for chunk in completion:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        response_text += delta.content

                return response_text  # ✅ Success: return output

            except RateLimitError:
                print(f"Rate limit hit! Rotating API key...")
                self.key_manager.rotate_key()
                self._update_client()
                time.sleep(1)

            except AuthenticationError:
                print(f"Bad key, removing it.")
                self.key_manager.keys.remove(self.key_manager.get_key())
                if not self.key_manager.keys:
                    raise
                self._update_client()

            except APIError as e:
                print(f"General API error: {e}. Retrying...")
                time.sleep(2)

        raise RuntimeError("All API keys failed after retries.")

    def geminiLLMInterface(self, prompt: str, imagePath: str = None) -> str:
        """
        Generates LLM response with or without image input.
        :param prompt: The text prompt to send to Gemini.
        :param imagePath: Optional path to an image (for multimodal reasoning).
        :return: Cleaned text output.
        """

        try:
            if imagePath:
                image = Image.open(imagePath)
                response = self.model.generate_content([prompt, image])
            else:
                response = self.model.generate_content(prompt)

            return response.text.strip()

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return ""

class APIKeyManager:
    def __init__(self, keys: list):
        self.keys = keys
        self.index = 0

    def get_key(self):
        return self.keys[self.index]

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.keys)
