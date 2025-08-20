#!/usr/bin/env python3
import os
import sys
import subprocess
from typing import Dict


OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

GDRIVE_FILES: Dict[str, str] = {
    # parquet with vectors
    "1IyPxqaejDYoBXY2WgzZWq-uHO6ak9_lL": "merged_df_with_vectors.parquet",
    # csvs
    "15RPocMQvVLQz9eszcYnvk1noNHMw19GX": "resp_use.csv",
    "1V_xshOgFWW24rEtd4Wlb-4Mz1ce2Jbpw": "pred_use.csv",
    "1nnc8e6oGXxD6XSylAvME0VzEUXSQJzRU": "merged.csv",
    "1284kxH49WfwwNW28KiUl98jnyJe2Tm9M": "gpt2_surprisal_results.csv",
}
# ----------------------------

def ensure_gdown():
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        print("Installing gdown...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def human(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024.0:
            return f"{nbytes:3.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"

def download_all(force: bool = False) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_gdown()
    import gdown  

    print(f"\nTarget folder: {OUT_DIR}\n")

    for fid, fname in GDRIVE_FILES.items():
        out_path = os.path.join(OUT_DIR, fname)
        if os.path.exists(out_path) and not force:
            print(f"✔ Skipping (exists): {fname} — {human(os.path.getsize(out_path))}")
            continue

        url = f"https://drive.google.com/uc?id={fid}"
        print(f"⬇ Downloading {fname} ...")
        # gdown returns the output path (or None on failure)
        result = gdown.download(id=fid, output=out_path, quiet=False)
        if result is None or not os.path.exists(out_path):
            # fallback: try fuzzy mode with URL
            result = gdown.download(url=url, output=out_path, quiet=False, fuzzy=True)
        if result is None or not os.path.exists(out_path):
            print(f" Failed: {fname}")
        else:
            print(f" Saved: {out_path} — {human(os.path.getsize(out_path))}")

    print("\nDone.\n")

def main():
    force = "--force" in sys.argv
    if force:
        print(" Forcing re-download of all files (overwriting if present).")
    download_all(force=force)

if __name__ == "__main__":
    main()
