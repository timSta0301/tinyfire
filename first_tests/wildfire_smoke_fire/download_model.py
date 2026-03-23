import requests
import os

url = "https://huggingface.co/odiug77/wildfire-smoke-fire/resolve/main/wildfire-smoke-fire.pt"
dest = os.path.join(os.path.dirname(__file__), "wildfire-smoke-fire.pt")

print("Downloading wildfire-smoke-fire.pt from HuggingFace...")
with requests.get(url, stream=True, timeout=60) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r{downloaded / total * 100:.1f}%", end="", flush=True)
print(f"\nModel saved to: {dest}")
