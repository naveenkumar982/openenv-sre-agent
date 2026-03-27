"""
Deploy to Hugging Face Spaces.
Usage: python deploy_to_hf.py <YOUR_HF_TOKEN>
"""
import sys
import os
from huggingface_hub import HfApi, login

if len(sys.argv) < 2:
    print("Usage: python deploy_to_hf.py <YOUR_HF_TOKEN>")
    print("Get your token from: https://huggingface.co/settings/tokens")
    sys.exit(1)

token = sys.argv[1]

# Login
print("🔐 Logging in to Hugging Face...")
login(token=token)

api = HfApi()
user = api.whoami()
username = user["name"]
print(f"✅ Logged in as: {username}")

# Create Space
repo_id = f"{username}/cloud-sre-simulator"
print(f"\n📦 Creating Space: {repo_id}")

try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"✅ Space created: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"Space may already exist, continuing... ({e})")

# Upload all project files
print("\n📤 Uploading files...")
project_dir = os.path.dirname(os.path.abspath(__file__))

files_to_upload = [
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

for fname in files_to_upload:
    fpath = os.path.join(project_dir, fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="space",
        )
        print(f"  ✅ {fname}")
    else:
        print(f"  ⚠️ {fname} not found, skipping")

print(f"\n🚀 Deployment complete!")
print(f"🌐 Your Space: https://huggingface.co/spaces/{repo_id}")
print(f"⏳ It may take 1-2 minutes to build. Check the Logs tab if needed.")
