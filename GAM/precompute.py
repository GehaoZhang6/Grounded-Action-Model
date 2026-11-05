"""Precompute the training cache: for every episode and frame, run the perception stack
(WildDet3D patch tokens + boxes, URDF robot silhouette, depth geometry) and store
everything the policy trains on. Output: one <episode>.npz per episode in train.cache_dir.

    python -m GAM.precompute --config configs/my_task.yaml [--ep-start 0 --ep-end 999]

Sharding: run several jobs with disjoint --ep-start/--ep-end; finished episodes are skipped.

Segment protocol (clicks source `tracks`, as recorded by GAM.tools.track or an annotator):
episodes are split at annotator breakpoints; per segment the pick slot is stabilized
frame-by-frame (BoxStabilizer EMA, reset at segment change) while the place slot is
FROZEN at its first successful detection (box2d/xyz/boxgeo/point cloud reused, its box
fed back to the detector as an extra patch-retention box). Static-click configs are
unaffected (single segment, no freeze/stabilizer unless perception.segment_protocol).
Cost: one WildDet3D forward per frame per slot (~0.5 s on a modern GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from GAM.config import load_config, perception_settings, build_extractor
from GAM.data.episodes import list_episodes, load_episode


def build_arm_mask_fn(cfg):
    r = cfg["perception"].get("robot") or None
    if not r:
        print("[precompute] WARNING: no robot section, patch masks will not contain the robot "
              "(the policy learns without it; keep eval identical)")
        return None
    from GAM.perception.arm_render import URDFArmMask
    m = URDFArmMask(r["urdf"], r["joint_names"], T_base_root=np.asarray(r["T_base_root"], np.float64),
                    link_filter=r.get("link_filter"), max_faces=int(r.get("max_faces", 1500)),
                    joint_affine=r.get("joint_affine"), joint_mirror=r.get("joint_mirror"))
    print(f"[precompute] URDF silhouette: {len(m._nodes)} geometries, {m.n_faces} faces")
    return m


def _depth_K(K, rgb_shape, depth_shape):
    """RGB intrinsics -> intrinsics at the depth resolution (depth is often half the RGB size; pixel-centre convention).
    Returns (KD, dscale); unchanged when the resolutions match."""
    if tuple(depth_shape[:2]) == tuple(rgb_shape[:2]):
        return K, 1.0
    sy = depth_shape[0] / float(rgb_shape[0]); sx = depth_shape[1] / float(rgb_shape[1])
    KD = K.copy()
    KD[0, 0] *= sx; KD[1, 1] *= sy
    KD[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    KD[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return KD, sx


def _obfb_kwargs(z_band):
    """object_from_box without a z_band parameter (older package versions) -> call without it."""
    import inspect
    from GAM.perception.geometry import object_from_box
    return {"z_band": z_band} if "z_band" in inspect.signature(object_from_box).parameters else {}


def geometry_from_detection(det, depth_m, K, E, source: str, z_band: float = 0.15):
    """WildDet3D detection -> object dict, by config `perception.geometry_source`:
    box2d_depth = depth backprojection inside the predicted 2D box (axis-aligned);
    box3d       = the predicted oriented 3D box. Falls back to the other source."""
    from GAM.perception.geometry import object_from_box
    obj = None
    if source == "box3d" and det.center is not None:
        obj = {"p": det.center, "extents": det.extents, "R": det.R}
    if obj is None and det.box2d[2] > det.box2d[0]:
        obj = object_from_box(det.box2d, depth_m, K, E, **_obfb_kwargs(z_band))
    if obj is None and det.center is not None:
        obj = {"p": det.center, "extents": det.extents, "R": det.R}
    return obj


def slot_prompts(slots):
    """Slot name -> text prompt: bowl1 / bowl2 -> "left bowl" / "right bowl" (as in the detector fine-tuning data);
    target_box -> "target box"; three or more of the same name keep the plain name."""
    import re
    base = [re.sub(r"\d+$", "", s).replace("_", " ").strip() if s != "__none__" else None for s in slots]
    out = list(base)
    for b in set(x for x in base if x):
        ids = [i for i, x in enumerate(base) if x == b]
        if len(ids) == 2:
            out[ids[0]] = "left " + b; out[ids[1]] = "right " + b
    return out


class FrameFeaturizer:
    """Per-frame policy inputs from one episode frame, the single code path shared by the cache
    precompute, end-to-end training and the evaluation servers.

        f = FrameFeaturizer(cfg, ext, arm_fn); f.reset(ep)
        o = f.frame(ep, t)   # patches (G*G,1024) f16, box2d (S,4), xyz (S,3), boxgeo (S,9), pcd_pts (S,N,3) f16, pcd_mask (S,N)

    Depth modes (perception.detector_depth / geometry_depth):
      sensor / sensor : detector and geometry both use the recorded depth
      zero   / sensor : monocular detector; the recorded depth, with the robot silhouette and the table removed,
                        only shapes the geometry (2D boxes stay the tight detector boxes)
      zero   / pred   : monocular detector; its own predicted depth, cleaned the same way, shapes the geometry
    Segment protocol (clicks from GAM/tools/track.py): per segment the pick slot is EMA-smoothed and the place
    slot frozen at its first detection; that box is fed back to the detector so its patch cells stay unmasked.
    """

    def __init__(self, cfg, ext, arm_fn):
        pc = perception_settings(cfg)
        self.cfg, self.ext, self.arm_fn = cfg, ext, arm_fn
        self.slots = list(cfg["data"]["slots"]); self.S = len(self.slots)
        self.n_pts = int(pc.get("n_points", 512)); self.geom_src = pc.get("geometry_source", "box2d_depth")
        self.G2 = int(pc.get("patch_grid", 36)) ** 2
        self.pred_depth = pc["geometry_depth"] == "pred"
        self.sensor_depth = pc["geometry_depth"] == "sensor" and pc["detector_depth"] == "zero"
        assert not (self.pred_depth and self.sensor_depth)
        self.text_mode = pc["grounding"] == "text"
        self.text_min = float(pc.get("text_score_min", 0.3))
        self.prompts = slot_prompts(self.slots) if self.text_mode else None
        from GAM.text.instruction_prompts import build_instruction_prompts
        self.iprompts = build_instruction_prompts(cfg) if self.text_mode else None   # data.prompts_from_instruction: per-episode prompts
        self.z_band = float(pc.get("z_band", 0.15))
        self.table_remove = bool(pc.get("table_remove", True)); self.table_margin = float(pc.get("table_margin", 0.008))   # drop the table plane from the geometry depth
        self.pick_z_band = float(pc.get("pick_z_band", 0.06 if self.sensor_depth else self.z_band))
        self.force_segments = bool(pc.get("segment_protocol", False))
        self.roles = ["pick" if n.startswith("pick") else "place" if n.startswith("place") else "other" for n in self.slots]

    def reset(self, ep):
        """Per-episode state: table height, segment protocol, stabilisers."""
        self.seg_proto = ep.segments is not None or self.force_segments
        self.seg_ids = ep.segments if ep.segments is not None else np.zeros(ep.T, np.int32)
        self.stab = self.frozen = None; self.prev_seg = None; self.table_z = None
        if self.iprompts is not None:
            self.prompts = self.iprompts(ep.instruction, self.slots, slot_prompts(self.slots))
            assert ep.instruction, f"{ep.path}: data.prompts_from_instruction needs the episode instruction (data.instruction_file)"
        if self.seg_proto:
            from GAM.perception.box_stabilizer import BoxStabilizer
            self.stab = BoxStabilizer(); self.frozen = {}
        if self.sensor_depth:
            from GAM.perception.geometry import depth_to_world_grid
            d0 = ep.depth_m(0); KD0, _ = _depth_K(ep.K[0].astype(np.float64), ep.rgb(0).shape, d0.shape)
            P0, ok0 = depth_to_world_grid(d0, KD0, ep.E[0].astype(np.float64))
            self.table_z = self._table_height(P0[ok0][:, 2])

    def _table_height(self, z):
        if not self.table_remove:
            return -1e9
        hz, ez = np.histogram(z, bins=200, range=(-0.3, 1.0))
        return float(ez[hz.argmax()] + (ez[1] - ez[0]) / 2)

    def frame(self, ep, t):
        import dataclasses
        from GAM.perception.geometry import build_slot_obs, depth_to_world, depth_to_world_grid
        S, slots = self.S, self.slots
        rgb = ep.rgb(t); dep = ep.depth_m(t)
        K, E = ep.K[t].astype(np.float64), ep.E[t].astype(np.float64)
        H, W = rgb.shape[:2]
        arm = self.arm_fn(ep.state[t], K, E, H, W) if self.arm_fn is not None else None
        cl = ep.clicks[t]
        valid = [i for i in range(S) if np.isfinite(cl[i]).all() and 0 <= cl[i][0] < W and 0 <= cl[i][1] < H]
        if self.pred_depth:
            KD, dscale, dep_geo = K, 1.0, None                 # the predicted depth exists only after the forward pass
            det_depth = np.zeros((H, W), np.float32)
        elif self.sensor_depth:
            KD, dscale = _depth_K(K, rgb.shape, dep.shape)
            dep_geo = dep.copy()
            if arm is not None:
                am_s = cv2.resize(arm.astype(np.uint8), (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                dep_geo[am_s] = 0.0
            Pg, okg = depth_to_world_grid(dep_geo, KD, E)
            dep_geo[okg & (Pg[..., 2] < self.table_z + self.table_margin)] = 0.0
            det_depth = np.zeros((H, W), np.float32)
        else:
            KD, dscale, dep_geo, det_depth = K, 1.0, dep, dep
        if self.seg_proto and self.seg_ids[t] != self.prev_seg:   # segment boundary: reset the pick stabilisers, unfreeze the place slots
            for i in range(S):
                if self.roles[i] == "pick":
                    self.stab.reset(i)
            self.frozen.clear(); self.prev_seg = self.seg_ids[t]
        frozen = self.frozen or {}
        if self.text_mode:
            send = [i for i in range(S) if self.prompts[i] is not None]
            pr = self.ext(rgb, det_depth, K, E, [], arm_mask=arm, texts=[self.prompts[i] for i in send])
        else:
            send = [i for i in valid if not (self.seg_proto and self.roles[i] == "place" and i in frozen)]
            extra = [frozen[i]["box2d"] for i in valid if self.roles[i] == "place" and i in frozen] or None
            pr = self.ext(rgb, det_depth, K, E, [tuple(cl[i]) for i in send], arm_mask=arm, extra_boxes_px=extra)
        if self.pred_depth:
            dep_geo = np.asarray(self.ext.last_depth, np.float32).copy()
            dep_geo[~np.isfinite(dep_geo) | (dep_geo < 0.05) | (dep_geo > 3.0)] = 0.0
            if arm is not None:
                dep_geo[np.asarray(arm, bool)] = 0.0
            Pg, okg = depth_to_world_grid(dep_geo, KD, E)
            if self.table_z is None:
                self.table_z = self._table_height(Pg[okg][:, 2])
            dep_geo[okg & (Pg[..., 2] < self.table_z + self.table_margin)] = 0.0
        objs, det_boxes = {}, {}
        for j, i in enumerate(send):
            det = pr.detections[j]
            if self.text_mode and (det.score < self.text_min or det.box2d[2] <= det.box2d[0]):
                continue                                      # text not found -> the slot is empty this frame
            if self.seg_proto and self.roles[i] == "pick":
                det = self.stab(i, det)                       # per-frame EMA (the same filter as at deployment)
            if self.sensor_depth and det.box2d[2] > det.box2d[0]:
                det = dataclasses.replace(det, box2d=det.box2d * dscale)
            zb = self.pick_z_band if self.roles[i] == "pick" else self.z_band
            obj = geometry_from_detection(det, dep_geo, KD, E, self.geom_src, z_band=zb)
            if obj is not None:
                objs[slots[i]] = obj
                det_boxes[i] = (det.box2d / dscale if self.sensor_depth else det.box2d).astype(np.float32)
        g = build_slot_obs(depth_to_world(dep_geo, KD, E), objs, slots, self.n_pts, E=E, K=K, seed=t)
        out = {"patches": pr.patches.numpy().astype(np.float16), "box2d": np.asarray(g["box2d"], np.float32).copy(),
               "xyz": np.asarray(g["xyz"], np.float32).copy(), "boxgeo": np.asarray(g["boxgeo"], np.float32).copy(),
               "pcd_pts": np.asarray(g["pcd_pts"]).astype(np.float16), "pcd_mask": np.asarray(g["pcd_mask"], bool).copy(),
               "n_found": len(objs)}
        if self.sensor_depth:
            for i, b in det_boxes.items():                    # 2D box = detector box (tight), not the OBB projection
                out["box2d"][i] = b
        if self.seg_proto:
            for i in range(S):
                if self.roles[i] != "place":
                    continue
                if i in frozen:                               # frozen segment: the whole slot reuses the detection from the segment start
                    for k_ in ("box2d", "xyz", "boxgeo", "pcd_pts", "pcd_mask"):
                        out[k_][i] = frozen[i][k_]
                elif i in det_boxes and slots[i] in objs:
                    frozen[i] = {k_: out[k_][i].copy() for k_ in ("box2d", "xyz", "boxgeo", "pcd_pts", "pcd_mask")}
        return out


def process_episode(ep, ext, arm_fn, cfg, out_path, featurizer=None):
    f = featurizer or FrameFeaturizer(cfg, ext, arm_fn); f.reset(ep)
    if f.text_mode:
        print(f"  text-prompt mode: {f.prompts} (score>={f.text_min})", flush=True)
    T, S, n = ep.T, f.S, f.n_pts
    patches = np.zeros((T, f.G2, 1024), np.float16)
    box2d = np.zeros((T, S, 4), np.float32); xyz = np.zeros((T, S, 3), np.float32); boxgeo = np.zeros((T, S, 9), np.float32)
    pcd_pts = np.zeros((T, S, n, 3), np.float16); pcd_mask = np.zeros((T, S, n), bool)
    t0 = time.time()
    for t in range(T):
        o = f.frame(ep, t)
        patches[t], box2d[t], xyz[t], boxgeo[t], pcd_pts[t], pcd_mask[t] = o["patches"], o["box2d"], o["xyz"], o["boxgeo"], o["pcd_pts"], o["pcd_mask"]
        if t % 25 == 0 or t == T - 1:
            print(f"  frame {t + 1}/{T}  slots-found={o['n_found']}/{S}  ({(time.time() - t0) / (t + 1):.2f} s/frame)", flush=True)
    H0, W0 = ep.rgb(0).shape[:2]
    tmp = out_path + ".tmp"                                  # never a *.npz name while incomplete
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, state=ep.state, action=ep.action, patches=patches, box2d=box2d,
                            xyz=xyz, boxgeo=boxgeo, pcd_pts=pcd_pts, pcd_mask=pcd_mask,
                            clicks=ep.clicks, image_wh=np.array([W0, H0], np.float32),
                            instruction=np.str_(ep.instruction),
                            prompts=np.array([p or "" for p in (f.prompts or [])], dtype=str),
                            **({"lang": ep.lang_emb} if getattr(ep, "lang_emb", None) is not None else {}))
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--ep-end", type=int, default=10**9)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg["train"]["cache_dir"]
    os.makedirs(out_dir, exist_ok=True)
    eps = list_episodes(cfg)[args.ep_start:args.ep_end]
    ext = build_extractor(perception_settings(cfg))
    arm_fn = build_arm_mask_fn(cfg)
    featurizer = None
    for i, path in enumerate(eps):
        name = f"episode_{path.split('#')[-1]}" if path.startswith("lerobot://") else os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, name + ".npz")
        if os.path.exists(out):
            print(f"[{i + 1}/{len(eps)}] {name}: exists, skip")
            continue
        print(f"[{i + 1}/{len(eps)}] {name}")
        ep = load_episode(path, cfg)
        lang_emb = None
        if cfg.get("text"):                        # sentence embedding for train.lang_mode: film
            from GAM.text.language_clicks import sentence_embedding
            lang_emb = sentence_embedding(cfg, ep.instruction).astype(np.float32)
        ep.lang_emb = lang_emb
        featurizer = featurizer or FrameFeaturizer(cfg, ext, arm_fn)
        process_episode(ep, ext, arm_fn, cfg, out, featurizer)
        ep.close()
    json.dump({"slots": cfg["data"]["slots"], "n_points": cfg["perception"].get("n_points", 512),
               "patch_grid": int(cfg["perception"].get("patch_grid", 36)),
               "geometry_source": cfg["perception"].get("geometry_source", "box2d_depth")},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print("[precompute] done")


if __name__ == "__main__":
    main()
