import os
import argparse
import json
from pathlib import Path
from typing import TextIO

import torch
import yaml
import numpy as np
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder

def jsonl_load(f: TextIO) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    return [json.loads(line) for line in f]

def encode_lang(datasets: list[Path]):
    with open("configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    device = torch.device("cuda:0")
    text_embedder = T5Embedder(
        from_pretrained = "google/t5-v1_1-xxl", 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=None
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    for dataset in datasets:
        task_jsonl = dataset / "meta" / "tasks.jsonl"
        with open(task_jsonl, 'r') as f_instr:
            tasks = jsonl_load(f_instr)
        instructions = []
        for i, task in enumerate(tasks):
            assert i == task["task_index"]
            instructions.append(task["task"])
        
        tokenized_res = tokenizer(
            instructions, return_tensors="pt",
            padding="longest",
            truncation=True
        )
        tokens = tokenized_res["input_ids"].to(device)
        attn_mask = tokenized_res["attention_mask"].to(device)

        with torch.no_grad():
            text_embeds = text_encoder(
                input_ids=tokens,
                attention_mask=attn_mask
            )["last_hidden_state"].detach().cpu()

        text_embeds = text_embeds.cpu().float().numpy()
        attn_mask = attn_mask.cpu().bool().numpy()

        np.savez(dataset / "t5-v1_1-xxl.npz", 
                 text_embeds=text_embeds, attn_mask=attn_mask)

        print(f"Encoded {len(instructions)} instructions for dataset {dataset.name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert language instructions to embeddings.')
    parser.add_argument('dataset', type=Path, nargs='+', 
                        help='The path to the dataset directory (e.g., /path/to/dataset)')
    args = parser.parse_args()

    for dataset in args.dataset:
        if not dataset.exists() or not dataset.is_dir():
            raise ValueError(f"Dataset path {dataset} does not exist or is not a directory.")
        
    encode_lang(args.dataset)
