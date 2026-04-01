"""
Deploy all project files to Hugging Face Spaces using raw sockets.
Bypasses urllib/httpx connection reset issues on Windows.
Usage: python deploy_raw.py <HF_TOKEN>
"""
import sys
import os
import ssl
import socket
import json
import time

if len(sys.argv) < 2:
    print("Usage: python deploy_raw.py <HF_TOKEN>")
    sys.exit(1)

TOKEN = sys.argv[1]
REPO_ID = "naveen982/cloud-sre-simulator"
HOST = "huggingface.co"

FILES_TO_UPLOAD = [
    "README.md",
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "models.py",
    "env.py",
    "tasks.py",
    "app.py",
    "baseline.py",
    "test_smoke.py",
    "validate_openenv.py",
]

project_dir = os.path.dirname(os.path.abspath(__file__))


def raw_https(method, path, body=None, content_type="application/json"):
    """HTTPS request via raw socket with post-TLS delay to avoid WinError 10054."""
    ctx = ssl.create_default_context()
    sock = socket.create_connection((HOST, 443), timeout=30)
    ss = ctx.wrap_socket(sock, server_hostname=HOST)
    time.sleep(0.5)  # critical delay after TLS handshake

    try:
        if body is None:
            body = b""
        elif isinstance(body, str):
            body = body.encode("utf-8")

        request = (
            f"{method} {path} HTTP/1.0\r\n"
            f"Host: {HOST}\r\n"
            f"Authorization: Bearer {TOKEN}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"\r\n"
        ).encode("utf-8") + body

        ss.sendall(request)

        response = b""
        while True:
            try:
                chunk = ss.recv(8192)
                if not chunk:
                    break
                response += chunk
            except socket.timeout:
                break

        resp_text = response.decode("utf-8", errors="replace")
        if "\r\n\r\n" in resp_text:
            headers_part, body_part = resp_text.split("\r\n\r\n", 1)
        else:
            headers_part, body_part = resp_text, ""

        status_line = headers_part.split("\r\n")[0]
        status_code = int(status_line.split(" ")[1])
        return status_code, body_part
    finally:
        ss.close()


def upload_file(filename):
    """Upload a single file via HF commit API using multipart form data."""
    filepath = os.path.join(project_dir, filename)
    if not os.path.exists(filepath):
        return None

    with open(filepath, "rb") as f:
        file_content = f.read()

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
    ).encode("utf-8") + file_content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    return raw_https(
        "POST",
        f"/api/spaces/{REPO_ID}/commit/main",
        body,
        f"multipart/form-data; boundary={boundary}",
    )


def main():
    print("=" * 60)
    print("  HF Spaces Deployer (Raw Socket - Windows Fix)")
    print("=" * 60)

    # Auth check
    print("\nVerifying authentication...")
    try:
        status, body = raw_https("GET", "/api/whoami-v2")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    if status == 200:
        try:
            info = json.loads(body)
            print(f"  Logged in as: {info.get('name', 'unknown')}")
        except:
            print(f"  Logged in (status {status})")
    else:
        print(f"  Auth failed ({status}): {body[:200]}")
        sys.exit(1)

    # Check space
    print(f"\nChecking Space: {REPO_ID}...")
    status, body = raw_https("GET", f"/api/spaces/{REPO_ID}")
    if status == 200:
        print("  Space exists!")
    else:
        print(f"  Space check returned {status}, attempting uploads anyway...")

    # Upload files
    print(f"\nUploading {len(FILES_TO_UPLOAD)} files...\n")
    ok = 0
    fail = 0

    for fname in FILES_TO_UPLOAD:
        fpath = os.path.join(project_dir, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP  {fname} (not found)")
            fail += 1
            continue

        size = os.path.getsize(fpath)
        print(f"  [{ok+fail+1}/{len(FILES_TO_UPLOAD)}] {fname} ({size:,} bytes)...", end=" ", flush=True)

        try:
            status, resp = upload_file(fname)
            if status in (200, 201):
                print(f"OK ({status})")
                ok += 1
            else:
                print(f"FAIL ({status})")
                if resp:
                    print(f"         {resp[:120]}")
                fail += 1
        except Exception as e:
            print(f"ERROR: {e}")
            fail += 1

        time.sleep(2)  # rate limit between uploads

    print(f"\n{'=' * 60}")
    print(f"  Results: {ok} uploaded, {fail} failed")
    if ok > 0:
        print(f"  Live at: https://huggingface.co/spaces/{REPO_ID}")
        print(f"  (May take 1-2 min to build)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
