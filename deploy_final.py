"""
Deploy all project files to Hugging Face Spaces using requests.
Handles Windows WinError 10054 with retries.
Fixes cp1252 encoding issues and uses correct HF API format.
Usage: python deploy_final.py <HF_TOKEN>
"""
import sys
import os
import json
import time
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if len(sys.argv) < 2:
    print("Usage: python deploy_final.py <HF_TOKEN>")
    sys.exit(1)

TOKEN = sys.argv[1]
REPO_ID = "naveen982/cloud-sre-simulator"
BASE = "https://huggingface.co"

FILES_TO_UPLOAD = [
    "README.md", "Dockerfile", "requirements.txt", "openenv.yaml",
    "models.py", "env.py", "tasks.py", "app.py",
    "baseline.py", "test_smoke.py", "validate_openenv.py",
]

project_dir = os.path.dirname(os.path.abspath(__file__))

# Session with retries
session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=3,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.headers.update({"Authorization": f"Bearer {TOKEN}"})


def api_call(method, path, **kwargs):
    """Make API call with connection retry."""
    url = f"{BASE}{path}"
    for attempt in range(5):
        try:
            if method == "GET":
                return session.get(url, timeout=30, **kwargs)
            else:
                return session.post(url, timeout=60, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 3 * (attempt + 1)
            print(f" [retry {attempt+1}/5, wait {wait}s]", end="", flush=True)
            time.sleep(wait)
    return None


def upload_single_file(filename):
    """Upload using the HF Hub create_commit style API."""
    filepath = os.path.join(project_dir, filename)
    with open(filepath, "rb") as f:
        file_content = f.read()

    # Use the commit API with proper multipart
    url_path = f"/api/spaces/{REPO_ID}/commit/main"

    # The HF commit API expects:
    # Part 1: "header" - JSON with commit info
    # Part 2: "file" - the actual file content
    header_data = json.dumps({
        "summary": f"Deploy {filename}",
        "files": [{"path": filename, "sample": "file"}],
        "lfsFiles": [],
        "deletedFiles": []
    })

    # Use requests multipart form
    files = {
        'header': ('header', header_data, 'application/json'),
        'file': (filename, file_content, 'application/octet-stream'),
    }

    resp = api_call("POST", url_path, files=files)
    if resp is None:
        return None, "Connection failed after 5 retries"
    return resp.status_code, resp.text[:300]


def main():
    print("=" * 60)
    print("  HF Spaces Final Deployer")
    print("=" * 60)

    # Auth
    print("\n[1/3] Authenticating...", end=" ", flush=True)
    r = api_call("GET", "/api/whoami-v2")
    if r and r.status_code == 200:
        name = r.json().get("name", "unknown")
        print(f"OK - logged in as: {name}")
    else:
        print(f"FAILED")
        sys.exit(1)

    # Ensure space exists
    print("[2/3] Checking space...", end=" ", flush=True)
    r = api_call("GET", f"/api/spaces/{REPO_ID}")
    if r and r.status_code == 200:
        print("EXISTS")
    else:
        print("creating...", end=" ", flush=True)
        r = api_call("POST", "/api/repos/create", json={
            "type": "space",
            "sdk": "docker",
            "name": REPO_ID.split("/")[1],
        })
        if r and r.status_code in (200, 201):
            print("CREATED")
        else:
            status = r.status_code if r else "N/A"
            print(f"WARNING ({status}) - continuing anyway")

    # Upload files
    print(f"[3/3] Uploading {len(FILES_TO_UPLOAD)} files...\n")
    ok = 0
    fail = 0

    for i, fname in enumerate(FILES_TO_UPLOAD, 1):
        fpath = os.path.join(project_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [{i:2d}/{len(FILES_TO_UPLOAD)}] SKIP {fname}")
            fail += 1
            continue

        size = os.path.getsize(fpath)
        print(f"  [{i:2d}/{len(FILES_TO_UPLOAD)}] {fname} ({size:,}B)...", end=" ", flush=True)

        try:
            status, msg = upload_single_file(fname)
            if status and status in (200, 201):
                print("OK")
                ok += 1
            else:
                # Sanitize msg for Windows console
                safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                print(f"FAIL ({status}): {safe_msg[:80]}")
                fail += 1
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
            fail += 1

        time.sleep(3)  # Rate limit

    print(f"\n{'=' * 60}")
    print(f"  Results: {ok} uploaded, {fail} failed")
    if ok > 0:
        print(f"  Live: https://huggingface.co/spaces/{REPO_ID}")
        print(f"  (Build takes ~2 min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
