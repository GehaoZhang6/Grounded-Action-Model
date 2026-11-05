"""GAMRunner: one observation in, one action chunk out, the closed-loop policy for any harness.

    runner = GAMRunner(cfg, ckpt)                      # cfg = load_config(...); ckpt local path or hf://
    runner.reset(instruction, seed=episode)
    chunk, boxes = runner.act(rgb, depth, K, E, state[, prefix])
    boxes = runner.boxes(rgb, depth, K, E, state)      # detection only (overlay videos)

Per frame (the same code path as the training-cache precompute): WildDet3D detection of every object slot,
robot silhouette from the URDF + joint state, object geometry and point clouds from the detector's 3D boxes and
(predicted or measured) depth. No simulator ground truth is used. How the slots are grounded follows
perception.grounding of the config the policy was trained with:
    text        the slot names are text prompts on every frame (RoboTwin experts)
    text+track  text detection on the first frame of the episode, then the online CoTracker follows the objects
    click       (training used offline tracks) -> at test time the same as text+track
`rgb` is the live camera image in RGB (if the training recordings were BGR, perception.image_channel_swap: true,
the runner converts so that live frames match the cache); `depth` in metres or millimetres (auto-detected),
K (3,3), E (3,4) base->camera, `state` the joint vector. Stitching successive chunks into per-step actions is
GAM.eval.chunker; GAM.eval.run drives a whole evaluation; benchmarks/robotwin/server.py exposes this class over HTTP.
A checkpoint trained with train.pcd_ref: ee (single-arm Franka) gets its object geometry re-expressed relative to the TCP
here, with the Panda forward kinematics on the first 7 state dims (GAM/data/panda_fk.py; swap in your own FK for another arm).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from GAM.config import perception_settings, build_extractor
from GAM.model.policy import GAMPolicy
from GAM.precompute import build_arm_mask_fn, geometry_from_detection, slot_prompts
from GAM.perception.geometry import build_slot_obs, depth_to_world, depth_to_world_grid

ONLINE_TRACKER = "hf://facebook/cotracker3/scaled_online.pth"


class GAMRunner:
    def __init__(self, cfg: dict, ckpt: str):
        from GAM.hub import fetch
        self.cfg = cfg; pc = perception_settings(cfg); ev = cfg.get("eval", {})
        self.grounding = {"click": "text+track"}.get(pc["grounding"], pc["grounding"])
        assert self.grounding in ("text", "text+track"), f"perception.grounding {pc['grounding']!r} is not supported at test time"
        self.policy = GAMPolicy.from_checkpoint(fetch(ckpt)); self.meta = getattr(self.policy, "meta", {}) or {}
        self.pcd_ref = str(self.meta.get("pcd_ref") or ("ee" if self.meta.get("pcd_ee_relative") else "none"))
        if self.pcd_ref != "none":                                     # single-arm Franka: geometry relative to the TCP (Panda FK on state[:7])
            from GAM.data.cache_dataset import relativize, ee_fk
            self._relativize, self._tcp = relativize, ee_fk()
            print("[runner] pcd_ref: ee uses the Franka Panda forward kinematics on state[:7] (GAM/data/panda_fk.py); other arms need their own FK", flush=True)
        self.slots = list(cfg["data"]["slots"])
        assert len(self.slots) == int(self.policy.encoder.n_slots), \
            f"config has {len(self.slots)} slots, checkpoint expects {self.policy.encoder.n_slots}"
        self.prompts = slot_prompts(self.slots)
        self.slots_auto = bool(cfg["data"].get("slots_auto")); self.head = None   # slots: auto -> the span head names the prompts per episode
        from GAM.text.instruction_prompts import build_instruction_prompts
        self.iprompts = build_instruction_prompts(cfg)               # data.prompts_from_instruction: object names parsed from the instruction
        self.n_pts = int(pc.get("n_points", 512)); self.geom_src = pc.get("geometry_source", "box2d_depth")
        self.pred_depth = pc["geometry_depth"] == "pred"
        self.text_min = float(pc.get("text_score_min", 0.3))
        self.table_remove = bool(pc.get("table_remove", True)); self.table_margin = float(pc.get("table_margin", 0.008))   # drop the table plane from the geometry depth
        self.swap = bool(pc.get("image_channel_swap", False))        # recordings were BGR -> make live RGB frames match
        pc_eval = dict(pc, precision=ev.get("precision", pc["precision"]))      # eval may run the detector in bf16
        self.precision = pc_eval["precision"]
        self.ext = build_extractor(pc_eval, grid=int(self.meta.get("patch_grid", round(self.policy.encoder.n_patch ** 0.5))))
        self.arm_fn = build_arm_mask_fn(cfg)
        self.state_hist = int(self.meta.get("state_hist", 2)); self.gen = torch.Generator(device="cuda").manual_seed(0)
        self.rtc_delay = int((ev.get("rtc") or {}).get("delay", 1))
        self.tracker_ckpt = fetch(cfg.get("checkpoints", {}).get("cotracker", ONLINE_TRACKER)) if self.grounding == "text+track" else None
        self.tracker_stagger = int(pc.get("tracker_stagger", 1))
        self.reset("", 0)
        print(f"[runner] ckpt={ckpt} slots={self.slots} prompts={self.prompts} grounding={self.grounding} grid={self.ext.grid} "
              f"precision={self.precision} tf32={torch.backends.cuda.matmul.allow_tf32} pred_depth={self.pred_depth}", flush=True)

    def reset(self, instruction: str = "", seed: int = 0):
        self.hist = []; self.t = 0; self.table_z = None
        if self.slots_auto:                                            # object names from the instruction (as GAM.tools.track did for training)
            from GAM.hub import fetch
            from GAM.text.phrases import manipulation_objects
            if self.head is None:
                from GAM.text.segmenter import T5ObjectSegmenter
                self.head = T5ObjectSegmenter.from_checkpoint(fetch(self.cfg["checkpoints"]["span_head"]))
            names = manipulation_objects(instruction, self.head.extract(instruction))[:len(self.slots)]
            self.prompts = names + [None] * (len(self.slots) - len(names))
            if instruction:
                print(f"[runner] {instruction!r} -> prompts {names}", flush=True)
        if self.iprompts is not None:
            self.prompts = self.iprompts(instruction, self.slots, slot_prompts(self.slots))
            if instruction:
                print(f"[runner] {instruction!r} -> prompts {self.prompts}", flush=True)
        self.tracker = None; self.tk_slots = []; self.last_pts = {}   # text+track state of the episode
        self.lang = None
        if getattr(self.policy, "lang_film", False):
            from GAM.text.language_clicks import sentence_embedding
            self.lang = torch.from_numpy(np.asarray(sentence_embedding(self.cfg, instruction), np.float32)).reshape(1, 768).cuda()
        self.gen.manual_seed(int(seed))                                # fixed noise stream per episode

    def _inputs(self, rgb, depth, K, E, state):
        rgb = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1] if self.swap else np.asarray(rgb)); dep = np.asarray(depth, np.float32)
        if dep[dep > 0].size and np.median(dep[dep > 0]) > 50:
            dep = dep / 1000.0                                         # mm -> m
        K = np.asarray(K, np.float64).reshape(3, 3); E = np.asarray(E, np.float64).reshape(3, 4)
        return rgb, dep, K, E, np.asarray(state, np.float32).reshape(-1)

    def _ground(self, rgb, dep, K, E, arm, valid, advance):
        """Detector pass with the slot prompts: text every frame, or first-frame text -> tracked points."""
        H, W = rgb.shape[:2]; det_dep = np.zeros((H, W), np.float32) if self.pred_depth else dep
        if self.grounding == "text":
            return self.ext(rgb, det_dep, K, E, [], arm_mask=arm, texts=[self.prompts[i] for i in valid]), valid
        if self.tracker is None:                                       # first frame: text detection seeds the tracker
            from GAM.perception.tracker import PointTracker
            r = self.ext(rgb, det_dep, K, E, [], arm_mask=arm, texts=[self.prompts[i] for i in valid])
            pts = []; self.tk_slots = []
            for j, i in enumerate(valid):
                d = r.detections[j]
                if d.score >= self.text_min and d.box2d[2] > d.box2d[0]:
                    pts.append((float((d.box2d[0] + d.box2d[2]) / 2), float((d.box2d[1] + d.box2d[3]) / 2))); self.tk_slots.append(i)
            self.tracker = PointTracker(self.tracker_ckpt, "cuda", track_wh=(W, H) if max(H, W) <= 640 else (640, int(640 * H / W)),
                                        stagger=self.tracker_stagger)
            self.tracker.start(rgb, pts); self.last_pts = {i: p for i, p in zip(self.tk_slots, pts)}
            return r, valid
        pts = self.tracker.push(rgb) if advance else self.tracker.pts; vis = self.tracker.vis
        for k, i in enumerate(self.tk_slots):                          # a lost point keeps its last visible position
            if k < len(pts) and vis[k] and 0 <= pts[k][0] < W and 0 <= pts[k][1] < H:
                self.last_pts[i] = (float(pts[k][0]), float(pts[k][1]))
        order = [i for i in self.tk_slots if i in self.last_pts]
        return self.ext(rgb, det_dep, K, E, [self.last_pts[i] for i in order], arm_mask=arm), order

    @torch.no_grad()
    def perceive(self, rgb, depth, K, E, state, advance=True):
        """Detector + object geometry of one observation: (patches, objects {slot: p/extents/R}, slot obs).
        advance=False (overlay frames) does not move the tracker on."""
        rgb, dep, K, E, q = self._inputs(rgb, depth, K, E, state); H, W = rgb.shape[:2]
        arm = self.arm_fn(q, K, E, H, W) if self.arm_fn is not None else None
        valid = [i for i in range(len(self.slots)) if self.prompts[i] is not None]
        pr, order = self._ground(rgb, dep, K, E, arm, valid, advance)
        if self.pred_depth:                                            # geometry from the detector's own depth: robot and table removed
            dep = np.asarray(self.ext.last_depth, np.float32).copy()
            dep[~np.isfinite(dep) | (dep < 0.05) | (dep > 3.0)] = 0.0
            if arm is not None:
                dep[np.asarray(arm, bool)] = 0.0
            Pg, okg = depth_to_world_grid(dep, K, E)
            if self.table_z is None:
                hz, ez = np.histogram(Pg[okg][:, 2], bins=200, range=(-0.3, 1.0)); self.table_z = float(ez[hz.argmax()] + (ez[1] - ez[0]) / 2) if self.table_remove else -1e9
            dep[okg & (Pg[..., 2] < self.table_z + self.table_margin)] = 0.0
        objs = {}
        for j, i in enumerate(order):
            d = pr.detections[j]
            if (self.grounding == "text" and d.score < self.text_min) or d.box2d[2] <= d.box2d[0]:
                continue                                               # prompt not found -> empty slot this frame
            obj = geometry_from_detection(d, dep, K, E, self.geom_src)
            if obj is not None:
                objs[self.slots[i]] = obj
        g = build_slot_obs(depth_to_world(dep, K, E), objs, self.slots, self.n_pts, E=E, K=K, seed=self.t)
        return pr, objs, g

    @staticmethod
    def _boxes(objs):
        return [{"slot": n, "p": np.asarray(ob["p"], float).tolist(), "extents": np.asarray(ob["extents"], float).tolist(),
                 "R": np.asarray(ob["R"], float).reshape(-1).tolist()} for n, ob in objs.items()]

    def advance(self, rgb):
        """text+track: feed a frame the policy is not queried on, so the tracker sees every frame (call once per step)."""
        if self.tracker is not None:
            self.tracker.push(np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1] if self.swap else np.asarray(rgb)))

    def boxes(self, rgb, depth, K, E, state):
        """Detected 3D boxes only (overlay videos: run on every step between policy queries)."""
        return self._boxes(self.perceive(rgb, depth, K, E, state, advance=False)[1])

    @torch.no_grad()
    def act(self, rgb, depth, K, E, state, prefix=None):
        """-> (chunk (H, Da) float32, boxes). `prefix` = unexecuted tail of the previous chunk (real-time chunking)."""
        t0 = time.time()
        q = np.asarray(state, np.float32).reshape(-1); self.hist = (self.hist + [q])[-self.state_hist:]
        if len(self.hist) < self.state_hist:
            self.hist = [q] * self.state_hist
        pr, objs, g = self.perceive(rgb, depth, K, E, state)
        xyz, pts = g["xyz"], g["pcd_pts"]
        if self.pcd_ref != "none":
            xyz, pts = self._relativize(xyz, pts, g["pcd_mask"], self._tcp(q[:7].astype(np.float64)), ref=self.pcd_ref)
        dev = "cuda"; t = lambda a, dt=np.float32: torch.from_numpy(np.asarray(a, dt))[None].to(dev)
        o = {"patch_tokens": pr.patches[None].to(dev), "box2d": t(g["box2d"]), "xyz": t(xyz), "boxgeo": t(g["boxgeo"]),
             "pcd_pts": t(pts), "pcd_mask": torch.from_numpy(np.asarray(g["pcd_mask"]))[None].to(dev),
             "state": t(np.concatenate(self.hist)), "lang": (self.lang if self.lang is not None else torch.zeros(1, 768, device=dev))}
        if prefix is not None and len(prefix):
            pf = torch.from_numpy(np.asarray(prefix, np.float32))[None].to(dev); Kp = pf.shape[1]
            w = self.policy.rtc_weights(self.policy.horizon, Kp, self.rtc_delay).to(dev)
            act = self.policy.predict_action(o, generator=self.gen, prefix=pf, prefix_weights=w, n_frozen=min(self.rtc_delay, Kp))
        else:
            act = self.policy.predict_action(o, generator=self.gen)
        self.t += 1; self.last_ms = int(1000 * (time.time() - t0)); self.last_n_found = len(objs)
        return act[0].float().cpu().numpy(), self._boxes(objs)
