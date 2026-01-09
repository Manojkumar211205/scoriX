import os
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from openai import OpenAI
from openai import RateLimitError  # Import for handling rate limit exceptions

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
        self.model = genai.GenerativeModel("gemini-2.5-flash")  # Stable, current version

        # Load multiple NVIDIA API keys dynamically from env vars like NVAPI_KEY_1, NVAPI_KEY_2, etc.
        self.nvapi_keys = [os.getenv("nvidiaKey1"), os.getenv("nvidiaKey2"),os.getenv("nvidiaKey3"),os.getenv("nvidiaKey4")]
        i = 1
        while True:
            key = os.getenv(f"NVAPI_KEY_{i}")
            if not key:
                break
            self.nvapi_keys.append(key)
            i += 1

        if not self.nvapi_keys:
            raise ValueError("❌ No NVAPI keys found. Please set at least one in your .env file as NVAPI_KEY_1, NVAPI_KEY_2, etc.")

        self.current_key_index = 0
        self.client = None  # Will be created dynamically in nvidiaResponse

    def nvidiaResponse(self, prompt: str, model: str = "meta/llama-3.3-70b-instruct",
                          temperature: float = 0.6, top_p: float = 0.7, max_tokens: int = 4096) -> str:
        response_text = ""
        max_retries = len(self.nvapi_keys)

        for attempt in range(max_retries):
            # Rotate to the next key in a round-robin fashion
            key_index = (self.current_key_index + attempt) % len(self.nvapi_keys)
            api_key = self.nvapi_keys[key_index]

            try:
                # Create a new client for this key (avoids baking in a single key)
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=api_key
                )

                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=True
                )

                for chunk in completion:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        response_text += delta.content

                # Success: Update current index to this key for future starts
                self.current_key_index = key_index
                return response_text

            except RateLimitError as e:
                print(f"⚠️ Rate limit hit with key {key_index + 1}. Trying next key... (Error: {e})")
                # Continue to next key
                continue
            except Exception as e:
                print(f"❌ Unexpected error with key {key_index + 1}: {e}")
                # For non-rate-limit errors, re-raise to avoid silent failures
                raise

        # All keys exhausted
        raise ValueError("❌ All NVIDIA API keys exhausted due to rate limits. Please wait or add more keys.")

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