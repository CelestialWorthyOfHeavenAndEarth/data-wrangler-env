"""Sync all files to HF Space."""
from huggingface_hub import HfApi
import os

api = HfApi()
REPO = "Aswini-Kumar/data-wrangler-env"
ROOT = r"C:\Users\aswin\OneDrive\Desktop\meta"

# Files to sync
files = [
    "server/dataset_generator.py",
    "server/data_wrangler_env_environment.py",
    "server/cleaning_engine.py",
    "server/grader.py",
    "server/app.py",
    "server/__init__.py",
    "client.py",
    "models.py",
    "inference.py",
    "__init__.py",
    "Dockerfile",
    "openenv.yaml",
    "pyproject.toml",
    "README.md",
    "requirements.txt",
]

for f in files:
    local = os.path.join(ROOT, f)
    if os.path.exists(local):
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=f,
            repo_id=REPO,
            repo_type="space",
            commit_message=f"Sync {f}",
        )
        print(f"  Synced: {f}")
    else:
        print(f"  SKIP (not found): {f}")

print("\nAll files synced to HF Space.")
