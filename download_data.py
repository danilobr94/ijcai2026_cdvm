"""Download large precomputed data files from OwnCloud.

Upload instructions (one-time, for maintainers):
  1. Zip data/attr_matr/         → attr_matr.zip        → upload to OwnCloud share
  2. Zip data/influence_scores/  → influence_scores.zip  → upload to OwnCloud share
     (i.e. all influence_score_matrix-LogisticRegression*.npy files)

Download (for reproducibility):
  python download_data.py
"""
import os
import zipfile
import urllib.request

SHARE = "https://owncloud.fraunhofer.de/index.php/s/cbK2Hv1B5lR9K2p"

FILES = {
    "attr_matr.zip":        ("data/attr_matr",  f"{SHARE}/download?path=%2F&files=attr_matr.zip"),
    "influence_scores.zip": ("data",             f"{SHARE}/download?path=%2F&files=influence_scores.zip"),
}


def download_and_extract(filename, dest_dir, url):
    os.makedirs(dest_dir, exist_ok=True)
    tmp = f"/tmp/{filename}"
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, tmp)
    print(f"Extracting to {dest_dir}/ ...")
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest_dir)
    os.remove(tmp)
    print(f"Done: {dest_dir}/")


if __name__ == "__main__":
    for filename, (dest_dir, url) in FILES.items():
        download_and_extract(filename, dest_dir, url)
    print("\nAll files ready.")
