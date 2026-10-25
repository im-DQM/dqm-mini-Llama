#!/usr/bin/env python3
"""
Upgrade an existing model.json to the Milestone 6 format.
Adds dtype, offset, and byte_size to each tensor entry.
Reads the existing model.json and writes back in-place.
Assumes float32 dtype and sequential layout in model.bin.
"""

import json
import argparse
import os


def upgrade(model_dir: str):
    json_path = os.path.join(model_dir, "model.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    if "tensors" not in data:
        raise ValueError("model.json missing 'tensors' field")

    if "tokenizer" not in data:
        data["tokenizer"] = {
            "type": "json_vocab",
            "path": "vocab.json",
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 0,
        }

    offset = 0
    for t in data["tensors"]:
        shape = t["shape"]
        numel = 1
        for dim in shape:
            numel *= dim
        byte_size = numel * 4  # float32
        t["dtype"] = "float32"
        t["offset"] = offset
        t["byte_size"] = byte_size
        offset += byte_size

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Upgraded {json_path}")
    print(f"Total tensors: {len(data['tensors'])}")
    print(f"Total binary size: {offset} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models/tiny", help="Model directory")
    args = parser.parse_args()
    upgrade(args.model_dir)


if __name__ == "__main__":
    main()
