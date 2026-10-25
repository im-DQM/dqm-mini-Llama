#!/usr/bin/env python3
"""
Generate a vocab.json equivalent to the AsciiTokenizer.
Format: [{"id": N, "content": "...", "special": true/false}, ...]
"""

import json
import argparse
import os


def generate_vocab(output_path: str):
    entries = []

    # Special tokens
    entries.append({"id": 0, "content": "<unk>", "special": True})
    entries.append({"id": 1, "content": "<bos>", "special": True})
    entries.append({"id": 2, "content": "<eos>", "special": True})

    # Control characters (3-31, 127): map to empty string
    for i in range(3, 32):
        entries.append({"id": i, "content": "", "special": False})

    # Printable ASCII (32-126)
    for i in range(32, 127):
        entries.append({"id": i, "content": chr(i), "special": False})

    # DEL (127)
    entries.append({"id": 127, "content": "", "special": False})

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="models/tiny/vocab.json")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_vocab(args.output)


if __name__ == "__main__":
    main()
