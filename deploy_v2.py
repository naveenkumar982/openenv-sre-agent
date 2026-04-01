"""
Deploy to HF Spaces - fixed multipart commit format.
The HF commit API needs specific multipart boundary formatting.
Usage: python deploy_v2.py <HF_TOKEN>
"""
import sys
import os
import io
import json
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if len(sys.argv) < 2:
    print("Usage: python deploy_v2.py <HF_TOKEN>")
    sys.exit(1)

TOKEN = sys.argv[1]
REPO_ID = "naveen982/cloud-sre-simulator"
BASE = "https://huggingface.co"

FILES = [
    "README.md", "Dockerfile", "requirements.txt", "openenv.yaml",
    "models.py", "env.py", "tasks.py", "app.py",
    "baseline.py", "test_smoke.py", "validate_openenv.py",
]

project_dir = os.path.dirname(os.path.abspath(__file__))

session = requests.Session()
retry = Retry(total=5, backoff_factor=3)
session.mount("https://", HTTPAdapter(max_retries=retry))
session.headers.update({"Authorization": f"Bearer {TOKEN}"})


def api(method, path, **kw):
    url = f"{BASE}{path}"
    for attempt in range(5):
        try:
            r = getattr(session, method)(url, timeout=60, **kw)
            return r
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def upload_all_files():
    """Upload all files in a single commit using manually constructed multipart."""
    boundary = "----HFCommitBoundary"
    
    # Build the header JSON - all files in one commit
    file_entries = []
    file_parts = []
    
    for i, fname in enumerate(FILES):
        fpath = os.path.join(project_dir, fname)
        if not os.path.exists(fpath):
            continue
        
        with open(fpath, "rb") as f:
            content = f.read()
        
        file_entries.append({"path": fname, "sample": f"file_{i}"})
        file_parts.append((f"file_{i}", fname, content))
    
    header = json.dumps({
        "summary": "Deploy Cloud SRE Simulator - Final Hackathon Submission",
        "files": file_entries,
        "lfsFiles": [],
        "deletedFiles": []
    })
    
    # Build multipart body manually
    parts = []
    
    # Header part
    parts.append(f"--{boundary}\r\n")
    parts.append(f'Content-Disposition: form-data; name="header"\r\n')
    parts.append(f"Content-Type: application/json\r\n\r\n")
    parts.append(header)
    parts.append(f"\r\n")
    
    # File parts - each with matching sample name
    body_prefix = "".join(parts).encode("utf-8")
    
    file_bodies = []
    for sample_name, fname, content in file_parts:
        part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{sample_name}"; filename="{fname}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + content + b"\r\n"
        file_bodies.append(part)
    
    ending = f"--{boundary}--\r\n".encode("utf-8")
    
    full_body = body_prefix + b"".join(file_bodies) + ending
    
    url = f"{BASE}/api/spaces/{REPO_ID}/commit/main"
    
    for attempt in range(5):
        try:
            r = session.post(
                url,
                data=full_body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                timeout=120,
            )
            return r.status_code, r.text[:500]
        except Exception as e:
            print(f"  retry {attempt+1}...", flush=True)
            time.sleep(5 * (attempt + 1))
    
    return None, "All retries failed"


def main():
    print("=" * 60)
    print("  HF Spaces Deployer v2 (All-in-one commit)")
    print("=" * 60)

    # Auth
    print("\n[1/3] Auth...", end=" ", flush=True)
    r = api("get", "/api/whoami-v2")
    if r and r.status_code == 200:
        print(f"OK ({r.json().get('name', '?')})")
    else:
        print("FAILED")
        sys.exit(1)

    # Check space
    print("[2/3] Space...", end=" ", flush=True)
    r = api("get", f"/api/spaces/{REPO_ID}")
    if r and r.status_code == 200:
        print("EXISTS")
    else:
        print("MISSING - creating...", end=" ", flush=True)
        r = api("post", "/api/repos/create", json={
            "type": "space", "sdk": "docker",
            "name": REPO_ID.split("/")[1],
        })
        print("OK" if r and r.status_code in (200, 201) else "WARN")

    # Upload all files in one commit
    existing = [f for f in FILES if os.path.exists(os.path.join(project_dir, f))]
    print(f"[3/3] Uploading {len(existing)} files in single commit...", flush=True)
    
    status, body = upload_all_files()
    
    if status and status in (200, 201):
        print(f"\n  SUCCESS! ({status})")
        print(f"\n  Live: https://huggingface.co/spaces/{REPO_ID}")
        print(f"  (~2 min to build)")
    else:
        safe = body.encode('ascii', errors='replace').decode('ascii') if body else "?"
        print(f"\n  FAILED ({status}): {safe[:200]}")

    print("=" * 60)


if __name__ == "__main__":
    main()
