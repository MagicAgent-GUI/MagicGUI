import pandas as pd
import os
import json
from pathlib import Path
from PIL import Image
import io

parquet_dir = Path("./tmp/ScreenSpot-v2-mobile/data")
files = list(parquet_dir.glob("*parquet"))
target_root = Path(os.path.dirname(os.path.dirname(__file__))) / 'ScreenSpot_v2_mobile'
images_dir = target_root / 'images'
jsonl_path = target_root / 'ScreenSpot_v2_mobile.jsonl'

os.makedirs(images_dir, exist_ok=True)
fd_output = os.open(jsonl_path, os.O_WRONLY | os.O_CREAT, 0o600) 

with os.fdopen(fd_output, 'w', encoding='utf-8') as jsonl_file:
    idx = 0
    for file in files:
        df = pd.read_parquet(file)
        for _, row in df.iterrows():
            image_name = row['file_name']
            if not image_name.startswith("mobile"):
                continue

            image_bytes = row['image']['bytes']
            image_path = images_dir / image_name
            image = Image.open(io.BytesIO(image_bytes))
            image.save(image_path)

            bbox = row['bbox']
            x1, y1, x2, y2 = [int(b * 1000) for b in bbox]
            bbox_str = f"<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"

            type_str = f"t2_{row['data_type']}&t3_{row['data_source']}"

            record = {
                "id": idx,
                "images": [image_name],
                "query": row['instruction'],
                "response": bbox_str,
                "type": type_str
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            idx += 1

print(f"Saved at:{target_root}")