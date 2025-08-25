import os
import json
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence, Array2D
from huggingface_hub import HfApi, HfFolder
from huggingface_hub import login

login("")

images_dir = "results/qwen_direct"
metadata_path = "eval_benchmark/GenAIBenchmark/genai_image_seed.json"

with open(metadata_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

rows = []
for img_id, data in meta.items():
    img_filename = f"{int(img_id):05d}_Qwen-Image.png"
    img_path = os.path.join(images_dir, img_filename)
    if not os.path.exists(img_path):
        continue
    
    row = {
        "id": data["id"],
        "image": img_path,
        "prompt": data["prompt"]
    }
    rows.append(row)

features = Features({
    "id": Value("string"),
    "image": Image(),
    "prompt": Value("string")
})

dataset = Dataset.from_list(rows, features=features)
ds_dict = DatasetDict({"train": dataset})

hf_repo = "Ayush-Singh/qwen-image-base-genaibenchmark"
ds_dict.push_to_hub(hf_repo)
