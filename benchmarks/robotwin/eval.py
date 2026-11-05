"""Evaluate a GAM policy on a RoboTwin task.

    python benchmarks/robotwin/eval.py --config configs/robotwin/shake_bottle.yaml \
        [--ckpt my_policy.pt] [--settings clean randomized] [--episodes 20] \
        [--sim-python /path/to/robotwin/env/bin/python] [--sim-env robotwin_env.sh] [--sim-gpu 1]

RoboTwin needs its own Python environment, so this script (run from the GAM environment)
  1. links the policy client and the two task configs into your RoboTwin checkout (eval.robotwin_root),
  2. starts the GAM policy server (server.py) on this GPU,
  3. runs script/eval_policy_gam.py (the official eval_policy.py main with --test_num and an output folder, see
     eval_policy_gam.py) for every setting and collects the success rate.
The checkpoint is `checkpoints.policy` of the config (downloaded from the Hub) unless --ckpt is given.
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from GAM.config import load_yaml

CAMERA = """
HD1K_D435:
  fovy: 45
  w: 1024
  h: 768
"""


POLICY_NAME = {"client": "gam", "universal": "gam_universal"}
ROBOTWIN_COMMIT = "6e75b8b"                               # RoboTwin 2.0 commit the release was evaluated with


def _symlink(target, link):
    """link -> target. Several evaluations may set up the same RoboTwin checkout at once, so an existing link to the
    same target is kept; a link to another checkout of this repository (an older clone) is replaced atomically,
    otherwise the simulator would import that clone's GAM."""
    try:
        os.symlink(target, link); return True
    except FileExistsError:
        if os.path.islink(link) and os.path.realpath(link) != os.path.realpath(target):
            tmp = f"{link}.{os.getpid()}.tmp"
            os.symlink(target, tmp); os.replace(tmp, link)
            print(f"[setup] {link} pointed to another checkout, now -> {target}", flush=True); return True
        return False


def setup_robotwin(root, adapter="client"):
    """Idempotent: policy/gam -> client, script/eval_policy_gam.py, task_config/gam_*.yml, the 1024x768 head camera type."""
    assert os.path.isfile(os.path.join(root, "script", "eval_policy.py")), \
        f"{root}/script/eval_policy.py not found: ROBOTWIN_ROOT must be a RoboTwin 2.0 checkout at commit {ROBOTWIN_COMMIT} (newer commits moved / removed it)"
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head and not head.startswith(ROBOTWIN_COMMIT):
        print(f"[setup] WARNING RoboTwin is at {head[:7]}, the released evaluation used {ROBOTWIN_COMMIT}", flush=True)
    drv_src, drv_dst = os.path.join(HERE, "eval_policy_gam.py"), os.path.join(root, "script", "eval_policy_gam.py")
    if not os.path.exists(drv_dst) or open(drv_dst).read() != open(drv_src).read():
        shutil.copy(drv_src, drv_dst); print(f"[setup] wrote {drv_dst}")
    pname = POLICY_NAME[adapter]; link = os.path.join(root, "policy", pname)
    if adapter == "universal":                               # policy/gam_universal: deploy_policy.py = GAM/eval/inprocess_policy.py
        src = os.path.join(HERE, "client_universal"); os.makedirs(src, exist_ok=True)
        for name, target in (("deploy_policy.py", os.path.join(ROOT, "GAM", "eval", "inprocess_policy.py")),
                             ("deploy_policy.yml", os.path.join(HERE, "client", "deploy_policy.yml"))):
            _symlink(target, os.path.join(src, name))
        if not os.path.exists(os.path.join(src, "__init__.py")):
            open(os.path.join(src, "__init__.py"), "w").write(open(os.path.join(HERE, "client", "__init__.py")).read())
    else:
        src = os.path.join(HERE, "client")
    if _symlink(src, link):
        print(f"[setup] linked {link} -> {src}")
    for f in sorted(os.listdir(os.path.join(HERE, "task_config"))):
        dst = os.path.join(root, "task_config", f)
        if not os.path.exists(dst) or open(dst).read() != open(os.path.join(HERE, "task_config", f)).read():
            shutil.copy(os.path.join(HERE, "task_config", f), dst); print(f"[setup] wrote {dst}")
    cam = os.path.join(root, "task_config", "_camera_config.yml")
    if "HD1K_D435:" not in open(cam).read():
        open(cam, "a").write(CAMERA); print(f"[setup] added camera type HD1K_D435 to {cam}")


def wait_ready(url, proc, timeout=2400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=2).read(); return True
        except Exception:
            time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--settings", nargs="+", default=None, choices=["clean", "randomized"])
    ap.add_argument("--episodes", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=18120)
    ap.add_argument("--sim-python", default=None, help="RoboTwin's python if it lives in another environment (starts the policy server)")
    ap.add_argument("--sim-env", default=None, help="shell file sourced before starting RoboTwin (library paths, Vulkan)")
    ap.add_argument("--sim-gpu", default=None, help="CUDA_VISIBLE_DEVICES for the simulator (default: same as the server)")
    ap.add_argument("--adapter", choices=["client", "universal"], default="client",
                    help="client: benchmarks/robotwin/client (HTTP or in-process); universal: GAM/eval/inprocess_policy.py (GAMRunner + ActionChunker, single environment only)")
    ap.add_argument("--overlay", action="store_true", help="also write <out>/<setting>/overlay/episode<i>_<S|F>.mp4: the head camera with the 3D boxes the policy saw")
    a = ap.parse_args()
    cfg = load_yaml(a.config); ev = cfg["eval"]; task = cfg["task"]
    root = os.path.abspath(ev["robotwin_root"]); out = os.path.abspath(a.out or ev["out_dir"]); os.makedirs(out, exist_ok=True)
    two_envs = a.sim_python is not None; sim_python = a.sim_python or sys.executable
    assert not (a.adapter == "universal" and two_envs), "--adapter universal imports GAM inside the simulator process (no --sim-python)"
    setup_robotwin(root, a.adapter)
    srv = None; url = f"http://127.0.0.1:{a.port}"; results = {}
    env = dict(os.environ, PYTHONWARNINGS="ignore::UserWarning", GAM_CONFIG=os.path.abspath(a.config))
    env.pop("GAM_SERVER_URL", None)
    if a.ckpt:
        env["GAM_CKPT"] = a.ckpt
    if two_envs:
        cmd = [sys.executable, "-u", os.path.join(HERE, "server.py"), "--config", os.path.abspath(a.config), "--port", str(a.port)] + (["--ckpt", a.ckpt] if a.ckpt else [])
        srv = subprocess.Popen(cmd, stdout=open(os.path.join(out, "server.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
        env["GAM_SERVER_URL"] = url
    try:
        if two_envs:
            if not wait_ready(url + "/", srv):
                raise SystemExit(f"policy server did not start, see {out}/server.log")
            print(f"[eval] policy server ready ({out}/server.log)", flush=True)
        else:
            print("[eval] single environment: the policy runs inside the simulator process", flush=True)
        if a.sim_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.sim_gpu)
        n = int(a.episodes or ev["episodes"])
        pre = (f"source {a.sim_env} && " if a.sim_env else "")
        if subprocess.call(["bash", "-c", pre + "command -v ffmpeg > /dev/null"], env=env) != 0:
            raise SystemExit("ffmpeg not found in the simulator environment (RoboTwin records the evaluation videos with it)")
        for setting in (a.settings or ev["settings"]):
            sdir = os.path.join(out, setting); os.makedirs(sdir, exist_ok=True)
            sim = (f"source {a.sim_env} && " if a.sim_env else "") + \
                  f"{sim_python} script/eval_policy_gam.py --config policy/{POLICY_NAME[a.adapter]}/deploy_policy.yml --overrides --task_name {task} " \
                  f"--task_config gam_{setting} --ckpt_setting gam --test_num {n} --seed 0 --policy_name {POLICY_NAME[a.adapter]}"
            print(f"[eval] {task}/{setting}: {n} episodes -> {sdir}/progress.log", flush=True)
            with open(os.path.join(sdir, "progress.log"), "w") as lf:
                ov = {"GAM_OVERLAY_DIR": os.path.join(sdir, "overlay")} if a.overlay else {}
                rc = subprocess.call(["bash", "-c", sim], cwd=root, env=dict(env, RT_EVAL_DIR=sdir, **ov), stdout=lf, stderr=subprocess.STDOUT)
            m = re.findall(r"Success rate: \D*?(\d+)/(\d+)", re.sub(r"\x1b\[[0-9;]*m", "", open(os.path.join(sdir, "progress.log"), errors="ignore").read()))
            s, t = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)
            results[setting] = {"success": s, "episodes": t, "exit_code": rc}
            print(f"[eval] {task}/{setting}: {s}/{t}" + ("" if rc == 0 and t else f"  (simulator exit code {rc}, see progress.log)"), flush=True)
    finally:
        if srv is not None:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
            except Exception:
                pass
        json.dump({"task": task, "config": os.path.abspath(a.config), "ckpt": a.ckpt or cfg["checkpoints"]["policy"], "results": results},
                  open(os.path.join(out, "result.json"), "w"), indent=1)
    print(f"[eval] done: {results}")


if __name__ == "__main__":
    main()
