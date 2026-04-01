"""
Persistent git push to HF Spaces with aggressive retries.
Keeps trying until it succeeds or hits max attempts.
Usage: python git_push_hf.py <HF_TOKEN>
"""
import subprocess
import sys
import time

if len(sys.argv) < 2:
    print("Usage: python git_push_hf.py <HF_TOKEN>")
    sys.exit(1)

TOKEN = sys.argv[1]
REMOTE_URL = f"https://naveen982:{TOKEN}@huggingface.co/spaces/naveen982/cloud-sre-simulator"
MAX_ATTEMPTS = 20
BASE_WAIT = 5

print("=" * 50)
print("  Persistent HF Spaces Git Push")
print("=" * 50)

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\nAttempt {attempt}/{MAX_ATTEMPTS}...", end=" ", flush=True)
    
    result = subprocess.run(
        ["git", "push", REMOTE_URL, "main", "--force"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode == 0:
        print("SUCCESS!")
        print(result.stdout)
        print(result.stderr)
        print(f"\nDeployed! https://huggingface.co/spaces/naveen982/cloud-sre-simulator")
        sys.exit(0)
    else:
        err = (result.stderr or result.stdout or "unknown error").strip()
        print(f"FAIL: {err[:100]}")
        
        if attempt < MAX_ATTEMPTS:
            wait = BASE_WAIT + (attempt * 2)
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)

print(f"\nFailed after {MAX_ATTEMPTS} attempts.")
print("Your network is blocking connections to huggingface.co.")
print("Try: VPN, mobile hotspot, or different network.")
sys.exit(1)
