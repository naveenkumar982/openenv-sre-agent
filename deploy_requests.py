"""
Deploy all project files to Hugging Face Spaces using requests with aggressive retries.
Handles Windows WinError 10054 connection resets.
Usage: python deploy_requests.py <HF_TOKEN>
"""
import sys
import os
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if len(sys.argv) < 2:
    print("Usage: python deploy_requests.py <HF_TOKEN>")
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

# Session with aggressive retries
session = requests.Session()
retry = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.headers.update({"Authorization": f"Bearer {TOKEN}"})


def upload_file(filename, max_attempts=5):
    filepath = os.path.join(project_dir, filename)
    if not os.path.exists(filepath):
        return None, "not found"

    with open(filepath, "rb") as f:
        content = f.read()

    boundary = "----HFBoundary987654"
    ops = json.dumps({
        "summary": f"Deploy {filename}",
        "files": [{"path": filename, "sample": "file"}],
        "lfsFiles": [],
        "deletedFiles": []
    })

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="header"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f"{ops}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    url = f"{BASE}/api/spaces/{REPO_ID}/commit/main"

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.post(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                timeout=30,
            )
            return resp.status_code, resp.text[:200]
        except requests.exceptions.ConnectionError as e:
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"    retry {attempt}/{max_attempts} (waiting {wait}s)...", end=" ", flush=True)
                time.sleep(wait)
            else:
                return None, str(e)[:200]


def main():
    print("=" * 60)
    print("  HF Spaces Deployer (requests + retries)")
    print("=" * 60)

    # Auth
    print("\nVerifying auth...", end=" ", flush=True)
    for attempt in range(5):
        try:
            r = session.get(f"{BASE}/api/whoami-v2", timeout=15)
            if r.status_code == 200:
                name = r.json().get("name", "unknown")
                print(f"OK! Logged in as: {name}")
                break
            else:
                print(f"failed ({r.status_code})")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            if attempt < 4:
                print(f"retry...", end=" ", flush=True)
                time.sleep(2 ** (attempt + 1))
            else:
                print("FAILED after 5 attempts. Network issue.")
                sys.exit(1)

    # Upload
    print(f"\nUploading {len(FILES_TO_UPLOAD)} files...\n")
    ok = 0
    fail = 0

    for i, fname in enumerate(FILES_TO_UPLOAD, 1):
        fpath = os.path.join(project_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [{i}/{len(FILES_TO_UPLOAD)}] SKIP {fname}")
            fail += 1
            continue

        size = os.path.getsize(fpath)
        print(f"  [{i}/{len(FILES_TO_UPLOAD)}] {fname} ({size:,}B)...", end=" ", flush=True)

        status, msg = upload_file(fname)
        if status and status in (200, 201):
            print(f"OK")
            ok += 1
        else:
            print(f"FAIL ({status}): {msg}")
            fail += 1

        time.sleep(3)

    print(f"\n{'=' * 60}")
    print(f"  Results: {ok} uploaded, {fail} failed")
    if ok > 0:
        print(f"  Live: https://huggingface.co/spaces/{REPO_ID}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
