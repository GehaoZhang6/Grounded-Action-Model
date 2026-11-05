"""Prepare RoboTwin demonstrations for GAM.precompute: writes the frames_keep.json that drops the
frames in which the robot does not move (RoboTwin pauses between motion segments).

    python benchmarks/robotwin/prepare_data.py --data ./data/robotwin [--tasks shake_bottle ...]

Expected layout (what `collect_data.sh <task> gam_collect` writes, copied or linked under --data):
    <data>/<task>/gam_collect/data/episode*.hdf5
Output: <data>/frames_keep.json  {task: {episode: {"T": frames, "still": dropped, "frames_keep": [indices]}}}
"""
import argparse, glob, json, os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--still-eps", type=float, default=1e-4, help="max joint change (rad) for a frame to count as still")
    a = ap.parse_args()
    out = os.path.join(a.data, "frames_keep.json")
    W = {}
    tasks = a.tasks or sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d, "gam_collect", "data")))
    for task in tasks:
        files = sorted(glob.glob(os.path.join(a.data, task, "gam_collect", "data", "episode*.hdf5")),
                       key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0))
        W[task] = {}; kept = []
        for f in files:
            ep = "".join(ch for ch in os.path.splitext(os.path.basename(f))[0] if ch.isdigit())
            with h5py.File(f, "r") as h:
                q = h["joint_action/vector"][:]
            still = np.concatenate([[False], np.abs(np.diff(q, axis=0)).max(1) < a.still_eps])
            W[task][ep] = {"T": int(len(q)), "still": int(still.sum()), "frames_keep": np.nonzero(~still)[0].tolist()}
            kept.append(len(W[task][ep]["frames_keep"]))
        print(f"{task}: {len(files)} episodes, {np.mean(kept) if kept else 0:.0f} frames kept per episode", flush=True)
    import fcntl                       # several tasks may be prepared in parallel: merge under a lock, never overwrite another task
    with open(out + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        merged = json.load(open(out)) if os.path.exists(out) else {}
        merged.update(W)
        tmp = f"{out}.{os.getpid()}.tmp"
        json.dump(merged, open(tmp, "w")); os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
