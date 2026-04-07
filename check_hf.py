from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("Aswini-Kumar/data-wrangler-env", repo_type="space")
with open("hf_files_list.txt", "w") as f:
    for name in sorted(files):
        f.write(name + "\n")
    f.write(f"\nTotal: {len(files)} files\n")
print("Done")
