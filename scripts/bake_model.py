"""Download the base model into the image's HF cache at build time.

Run from the Dockerfile. Baking the weights into the image removes the runtime
dependency on huggingface.co, which is rate-limited from shared serving-host
egress IPs (observed as a 429 on the app's first cold start). Retries cover
the same throttling striking the build farm.
"""

from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
WEIGHT_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja"]
ATTEMPTS = 5


def bake() -> None:
    """Fetch the base model with exponential backoff between attempts."""
    for attempt in range(ATTEMPTS):
        try:
            snapshot_download(BASE_MODEL_ID, allow_patterns=WEIGHT_PATTERNS)
            print(f"baked {BASE_MODEL_ID} into the image cache")
            return
        except Exception as error:  # noqa: BLE001 -- any network failure retries
            if attempt == ATTEMPTS - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"attempt {attempt + 1} failed ({error}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


if __name__ == "__main__":
    bake()
