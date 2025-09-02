import json
from pathlib import Path

from datasets import load_dataset

PARQUET_FILES = [
    "./tmp/ScreenQA-short/test-00000-of-00002.parquet",
    "./tmp/ScreenQA-short/test-00001-of-00002.parquet",
]

SCRIPT_DIR = Path(__file__).resolve().parent 
TARGET_DIR = SCRIPT_DIR.parent / "ScreenQA-short"
print(TARGET_DIR)
IMAGES_DIR = TARGET_DIR / "images"
JSONL_PATH = TARGET_DIR / "ScreenQA-short.jsonl"

TARGET_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

ds = load_dataset(
    "parquet",
    data_files=PARQUET_FILES,
    split="train",         
)

with JSONL_PATH.open("w", encoding="utf-8") as f_jsonl:
    for idx, example in enumerate(ds):
        image_filename = Path(example["file_name"]).name
        image_save_path = IMAGES_DIR / image_filename
        example["image"].save(image_save_path)

        record = {
            "images": [image_filename],
            "query": example["question"],
            "response": example["ground_truth"][0] if example["ground_truth"] else "",
            "other_responses": example["ground_truth"],
            "id": idx,    
        }
        f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved at:{TARGET_DIR}")