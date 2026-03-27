"""
Upload a single file to HF Spaces using only urllib (bypasses httpx/requests issues).
Usage: python upload_to_hf.py <HF_TOKEN> <filename>
"""
import sys
import os
import json
import urllib.request
import urllib.error

if len(sys.argv) < 3:
    print("Usage: python upload_to_hf.py <HF_TOKEN> <filename>")
    print("Example: python upload_to_hf.py hf_xxx validate_openenv.py")
    sys.exit(1)

token = sys.argv[1]
filename = sys.argv[2]
repo_id = "naveen982/cloud-sre-simulator"

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
if not os.path.exists(filepath):
    print(f"ERROR: {filepath} not found")
    sys.exit(1)

with open(filepath, "rb") as f:
    content = f.read()

# HF Hub API: Upload file via commit
url = f"https://huggingface.co/api/spaces/{repo_id}/commit/main"

# Build multipart payload
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

operations = json.dumps({
    "summary": f"Add {filename}",
    "files": [{"path": filename, "sample": "file"}],
    "lfsFiles": [],
    "deletedFiles": []
})

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="header"\r\n'
    f"Content-Type: application/json\r\n\r\n"
    f"{operations}\r\n"
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
    f"Content-Type: application/octet-stream\r\n\r\n"
).encode() + content + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    url,
    data=body,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    },
    method="POST",
)

print(f"Uploading {filename} to {repo_id}...")
try:
    resp = urllib.request.urlopen(req, timeout=30)
    result = resp.read().decode()
    print(f"SUCCESS: {resp.status}")
    print(result)
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print(e.read().decode())
except Exception as e:
    print(f"Error: {e}")
