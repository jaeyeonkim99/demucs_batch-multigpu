import argparse
import json
import time
from pathlib import Path
from typing import List, Union

import torch
from tqdm import tqdm

import demucs.separate


def demix(
    paths: List[Path],
    demix_dir: Path,
    device: Union[str, torch.device],
    batch_size: int = 32,
    num_workers: int = 32,
    num_jobs: int = 8,
):
    """Demixes audio files using true batch processing (Stacking multiple files)."""
    # 1. Filter files to handle

    model_name = "htdemucs"

    todos = []
    demix_paths = []
    for path in tqdm(paths, desc="Filtering files to demix", dynamic_ncols=True):
        out_dir = demix_dir / model_name / path.stem
        demix_paths.append(out_dir)

        if out_dir.is_dir():
            if (
                (out_dir / "bass.wav").is_file()
                and (out_dir / "drums.wav").is_file()
                and (out_dir / "other.wav").is_file()
                and (out_dir / "vocals.wav").is_file()
            ):
                continue
        todos.append(path)

    existing = len(paths) - len(todos)
    print(f"=> Found {existing} tracks already demixed, {len(todos)} to demix.")

    if not todos:
        return demix_paths

    # 2. Handle file in batches
    print(f"Starting batch demixing (Batch Size: {batch_size}) on {device}...")
    cmd_args = [
        "--name",
        "htdemucs",
        "--out",
        demix_dir.as_posix(),
        "--device",
        str(device),
        "--n_batch",
        str(batch_size),
        "--num_worker",
        str(num_workers),
        "--jobs",
        str(num_jobs),
        *[p.as_posix() for p in todos],
    ]

    demucs.separate.main(cmd_args)

    return demix_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Batch demix audio files using Demucs")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Metadata JSON of input files (e.g. total_metadata_v3_music_filtered.json)",
    )
    parser.add_argument(
        "--audio_root",
        default="",
        type=str,
        help="Root directory of the audio; metadata wav_path is relative to it",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Demix cache dir (pass the same dir as --demix_dir to extract_music_temporal.py)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        default=64,
        type=int,
        help="Batch size for processing files",
    )
    parser.add_argument(
        "--num_workers",
        default=32,
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--num_jobs",
        default=8,
        type=int,
        help="Number of parallel jobs for demixing",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to use for demixing (e.g., 'cuda' or 'cpu')",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.input, "r") as f:
        metadata = json.load(f)

    input_paths = []
    for key, item in metadata.items():
        input_paths.append(Path(args.audio_root) / item["wav_path"])

    start = time.time()
    print(f"Total files to process: {len(input_paths)}")
    print("Start Demixing...")
    demix_paths = demix(
        paths=input_paths,
        demix_dir=Path(args.output),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_jobs=args.num_jobs,
    )
    end = time.time()

    print("Demixing completed.")
    print(
        f"Total files processed: {len(input_paths)}, Total files demixed: {len(demix_paths)}"
    )
    print(
        f"Total time taken: {end - start:.2f} seconds, Average time per file: {(end - start) / len(input_paths):.2f} seconds"
    )
