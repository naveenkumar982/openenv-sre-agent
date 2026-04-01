"""
OpenEnv multi-mode deployment entry point.

This module provides the main() function required by the OpenEnv
multi-mode deployment standard. It starts the FastAPI server
configured in the root app.py.
"""

import sys
import os

# Ensure the parent directory is on the path so we can import the root app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main(host: str = "0.0.0.0", port: int = 7860):
    """Start the OpenEnv HTTP server."""
    import uvicorn
    from app import app

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
