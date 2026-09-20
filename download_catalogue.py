"""
download_catalogue.py
---------------------
Downloads the LeetCode problems catalogue from Hugging Face dataset
'kaysss/leetcode-problem-set' (MIT License) and saves it to data/leetcode_problems.csv.
"""

from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

def download():
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target_path = data_dir / "leetcode_problems.csv"
    
    print("Downloading data.csv from Hugging Face: kaysss/leetcode-problem-set ...")
    downloaded_path = hf_hub_download(
        repo_id="kaysss/leetcode-problem-set",
        filename="data.csv",
        repo_type="dataset",
    )
    shutil.copyfile(downloaded_path, target_path)
    print(f"Successfully saved to {target_path}")

if __name__ == "__main__":
    download()
