"""Episode reader driven by the config's key map, adapt the YAML, not the code.

An episode is one HDF5 file with, per frame, the head camera RGB-D + calibration, the
robot state and the action, plus a pixel click per object slot (either recorded per
frame, or projected from known 3D object centres). See configs/example_bimanual.yaml.
"""
from __future__ import annotations

import glob
import os
import json
from dataclasses import dataclass
from typing import List, Optional

import cv2
import h5py
import numpy as np



@dataclass
class Episode:
    path: str
    T: int
    state: np.ndarray            # (T, state_dim)
    action: np.ndarray           # (T, action_dim)
    K: np.ndarray                # (T, 3, 3)
    E: np.ndarray                # (T, 3, 4) world(base)->camera, OpenCV
    clicks: np.ndarray           # (T, S, 2) pixel click per slot (NaN = missing that frame)
    instruction: str = ""
    segments: Optional[np.ndarray] = None   # (T,) int segment id (tracks source); None = one segment

    _f: Optional[h5py.File] = None
    _rgb_key: str = ""
    _depth_key: str = ""
    _depth_scale: float = 1.0

    _idx: Optional[np.ndarray] = None   # kept frame -> original frame index (frames_keep_json)

    def rgb(self, t: int) -> np.ndarray:
        """(H,W,3) uint8 RGB."""
        if self._idx is not None: t = int(self._idx[t])
        v = self._f[self._rgb_key][t]
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(np.asarray(v).tobytes()
                                                       if not isinstance(v, (bytes, np.bytes_)) else v,
                                                       np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def depth_m(self, t: int) -> np.ndarray:
        """(H,W) float32 metres."""
        if self._idx is not None: t = int(self._idx[t])
        if not self._depth_key:                                  # no depth recorded: the monocular recipe never reads it
            return np.zeros(self.rgb(t).shape[:2], np.float32)
        return np.asarray(self._f[self._depth_key][t], np.float32) * self._depth_scale

    def close(self):
        if self._f is not None:
            self._f.close()


def _per_frame(a, T):
    a = np.asarray(a, np.float64)
    return a if a.ndim >= 3 or (a.ndim == 2 and a.shape[0] == T and a.shape != (3, 3)) \
        else np.repeat(a[None], T, 0)


def list_episodes(cfg: dict) -> List[str]:
    if cfg["data"].get("source") == "lerobot":
        from GAM.data import lerobot_episodes
        return lerobot_episodes.list_episodes(cfg)
    g = cfg["data"]["episodes_glob"]
    pats = g if isinstance(g, (list, tuple)) else [g]   # several globs allowed
    eps = sorted(set(p for pat in pats for p in glob.glob(pat)))
    assert eps, f"no episodes match {g}"
    return eps


def _frames_keep(path: str, d: dict):
    """data.frames_keep_json: {task: {episode: {"frames_keep": [n, [idx...]]}}}, frames to keep per episode;
    task = the <task> part of .../data/<task>/<cfg>/data/episodeN.hdf5, episode = N. Returns the kept frame indices or None."""
    fj = d.get("frames_keep_json")
    if not fj:
        return None
    if not hasattr(_frames_keep, "_cache") or _frames_keep._cache[0] != fj:
        _frames_keep._cache = (fj, json.load(open(fj)))
    W = _frames_keep._cache[1]
    parts = os.path.normpath(path).split(os.sep)
    task = parts[-4] if len(parts) >= 4 else ""
    ep = "".join(ch for ch in os.path.splitext(os.path.basename(path))[0] if ch.isdigit())
    rec = W.get(task, {}).get(ep)
    if rec is None or "frames_keep" not in rec:
        return None
    fk = rec["frames_keep"]
    return np.asarray(fk[1] if isinstance(fk, (list, tuple)) and len(fk) == 2 and isinstance(fk[1], list) else fk, np.int64)


def load_episode(path: str, cfg: dict) -> Episode:
    if path.startswith("lerobot://"):
        from GAM.data import lerobot_episodes
        return lerobot_episodes.load_episode(path, cfg)
    d = cfg["data"]
    f = h5py.File(path, "r")
    state = np.asarray(f[d["state_key"]], np.float32)
    action = np.asarray(f[d["action_key"]], np.float32)
    T = len(state)
    K = _per_frame(f[d["intrinsic_key"]], T).reshape(T, 3, 3)
    E_raw = _per_frame(f[d["extrinsic_key"]], T)
    if d.get("extrinsic_is_cam2base", False):                    # 4x4 camera pose -> world->cam
        E = np.zeros((T, 3, 4))
        for t in range(T):
            M = E_raw[t].reshape(4, 4)
            E[t, :, :3] = M[:3, :3].T
            E[t, :, 3] = -M[:3, :3].T @ M[:3, 3]
    else:
        E = E_raw.reshape(T, -1)[:, :12].reshape(T, 3, 4)
    slots = d["slots"]
    S = len(slots)
    clicks = np.full((T, S, 2), np.nan, np.float32)
    segments = None
    csrc = d["clicks"]["source"]
    if csrc == "dataset":
        c = np.asarray(f[d["clicks"]["dataset_key"]], np.float32)
        if c.ndim == 2:                                          # (S,2) frame-0 -> held constant
            clicks[:] = c[None]
        else:                                                    # (T,S,2) per-frame
            clicks[:] = c
    elif csrc == "tracks":                                       # GAM.tools.track output
        tp = os.path.join(d["clicks"]["tracks_dir"],
                          os.path.splitext(os.path.basename(path))[0] + ".npz")
        if os.path.exists(tp):
            z = np.load(tp)
            clicks[:] = np.asarray(z["clicks"], np.float32)[:T]
            seg = np.asarray(z["seg_id"], np.int32)[:T]
            segments = seg if len(np.unique(seg)) > 1 else None      # one segment = plain per-frame clicks
        else:
            print(f"[episodes] WARNING no tracks for {os.path.basename(path)} (run GAM.tools.track first)")
    elif csrc == "none":
        pass          # text-prompted grounding (perception.grounding: text): no clicks needed
    else:
        raise ValueError(f"unknown clicks.source {csrc!r}")
    instr = ""
    ik = d.get("instruction_key")
    if ik and ik in f:
        v = f[ik][()]
        v = v[0] if getattr(v, "shape", None) else v
        instr = v.decode() if isinstance(v, bytes) else str(v)
    ifile = d.get("instruction_file")                            # e.g. <task>/gam_collect/instructions/{stem}.json (RoboTwin)
    if not instr and ifile:
        ip = ifile.format(stem=os.path.splitext(os.path.basename(path))[0])
        if os.path.exists(ip):
            v = json.load(open(ip))
            v = v.get("seen") or v.get("unseen") or [""] if isinstance(v, dict) else v
            instr = str(v[0] if isinstance(v, list) else v)
    idx = _frames_keep(path, d)
    if idx is not None:
        idx = idx[(idx >= 0) & (idx < T)]
        state, action, K, E, clicks = state[idx], action[idx], K[idx], E[idx], clicks[idx]
        segments = segments[idx] if segments is not None else None
        T = len(idx)
    ep = Episode(path=path, T=T, state=state, action=action, K=K.astype(np.float32),
                 E=E.astype(np.float32), clicks=clicks, instruction=instr,
                 segments=segments)
    ep._idx = idx
    ep._f = f
    ep._rgb_key = d["rgb_key"]
    ep._depth_key = d.get("depth_key") or ""
    ep._depth_scale = 0.001 if d.get("depth_unit", "m") == "mm" else 1.0
    return ep
