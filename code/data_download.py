import os, sys, subprocess

FILE_ID = "1IyPxqaejDYoBXY2WgzZWq-uHO6ak9_lL"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(OUT_DIR, "merged_df_with_vectors.parquet")  

os.makedirs(OUT_DIR, exist_ok=True)

try:
    import gdown  # noqa: F401
except ImportError:
    print("Installing gdown...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

link = f"https://drive.google.com/uc?id={FILE_ID}"
subprocess.check_call([sys.executable, "-m", "gdown", link, "-O", OUT_PATH, "--fuzzy"])
print(f"Saved to {os.path.abspath(OUT_PATH)}")