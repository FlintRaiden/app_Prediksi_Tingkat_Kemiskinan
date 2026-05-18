# Root-level entry point for Hugging Face Spaces
# Hugging Face expects app.py at the root to expose a `app` WSGI object.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
from app import app  # noqa: F401

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
