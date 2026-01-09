"""Setup script to create .env file with NVIDIA API key."""

import os

NVIDIA_API_KEY = "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"

env_content = f"""# NVIDIA API Configuration
NVIDIA_API_KEY={NVIDIA_API_KEY}
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Model Configuration
LLM_MODEL=meta/llama-3.1-70b-instruct
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
"""

if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write(env_content)
    print("[SUCCESS] Created .env file with NVIDIA API configuration")
else:
    print("[WARNING] .env file already exists. Please update it manually with:")
    print(f"NVIDIA_API_KEY={NVIDIA_API_KEY}")
    print("NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1")

