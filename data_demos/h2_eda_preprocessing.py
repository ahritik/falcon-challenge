"""
H2 EDA & preprocessing utility.

Purpose
- Quick inspection of H2 sessions and trials
- Basic preprocessing: bin/normalize spikes, derive session IDs
- Saves summary CSV and optional preprocessed caches you can reuse for training

Usage (PowerShell):
  python data_demos/h2_eda_preprocessing.py --root data/h2 --out-dir local_data/h2_preproc --save-caches

Outputs
- local_data/h2_preproc/h2_sessions.csv: per-file/session summary
- Optional per-trial cache: npz files with fields: spikes (TxC), session_id (str), text (str)

Notes
- Uses falcon_challenge.dataloaders.load_nwb for consistent parsing
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import json
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb


def infer_session_from_name(stem: str) -> str:
    """Infer H2 session string (YYYY.MM.DD) from various filename styles."""
    # Prefer explicit dotted date
    m = re.search(r"(20\d{2})\.(\d{2})\.(\d{2})", stem)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    # Fallback: DANDI style ses-YYYYMMDD
    m = re.search(r"ses-(20\d{2})(\d{2})(\d{2})", stem)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return stem  # last resort, whole stem


def segment_trials(spikes: np.ndarray, trial_done: np.ndarray) -> List[slice]:
    """Return list of slices for each trial using a boolean 'done' mask per time step."""
    ends = np.where(trial_done)[0].tolist()
    starts = [0] + [e + 1 for e in ends[:-1]]
    return [slice(s, e + 1) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/h2", help="Path to data/h2 root")
    ap.add_argument("--out-dir", type=str, default="local_data/h2_preproc", help="Output directory for summaries and caches")
    ap.add_argument("--save-caches", action="store_true", help="Save per-trial caches (npz)")
    args = ap.parse_args()

    root = Path(args.root)
    held_in = sorted((root / "held_in_calib").glob("*.nwb"))
    minival = sorted((root / "minival").glob("*.nwb"))
    held_out = sorted((root / "held_out_calib").glob("*.nwb"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cache_dir = out_dir / "trial_caches"
    if args.save_caches:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for split_name, files in [("held_in", held_in), ("minival", minival), ("held_out", held_out)]:
        for fn in tqdm(files, desc=f"Scanning {split_name}"):
            spikes, targets, trial_done, eval_mask = load_nwb(fn, dataset=FalconTask.h2)
            session = infer_session_from_name(fn.stem)
            T, C = spikes.shape
            n_trials = int(np.sum(trial_done))
            trial_slices = segment_trials(spikes, trial_done)
            lengths = [slc.stop - slc.start for slc in trial_slices]
            rows.append({
                "file": str(fn),
                "session": session,
                "split": split_name,
                "timesteps": T,
                "channels": C,
                "trials": n_trials,
                "t_mean": float(np.mean(lengths)) if lengths else 0.0,
                "t_std": float(np.std(lengths)) if lengths else 0.0,
            })

            if args.save_caches:
                # Save each trial as individual cache; pair in-order with targets
                # targets is list/array of per-trial character codes -> convert to string
                tgt_strings = ["".join(chr(int(c)) for c in arr if int(c) != 0) for arr in targets]
                # some files may have mismatch; guard by min length
                n = min(len(trial_slices), len(tgt_strings))
                for i in range(n):
                    slc = trial_slices[i]
                    trial_spikes = spikes[slc]
                    text = tgt_strings[i]
                    cache_path = cache_dir / f"{session}__{fn.stem}__trial{i:04d}.npz"
                    np.savez_compressed(cache_path, spikes=trial_spikes.astype(np.float32), text=text, session=session)

    # Write summary CSV
    import csv
    csv_path = out_dir / "h2_sessions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file","session","split","timesteps","channels","trials","t_mean","t_std"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Also write a JSON with discovered sessions
    sessions = sorted({r["session"] for r in rows})
    with open(out_dir / "sessions.json", "w") as f:
        json.dump({"sessions": sessions}, f, indent=2)

    print(f"Summary written to {csv_path}")
    if args.save_caches:
        print(f"Trial caches written to {cache_dir}")


if __name__ == "__main__":
    main()
