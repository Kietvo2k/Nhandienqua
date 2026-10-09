"""Train trên Google Colab bằng Colab CLI chính thức (token/phiên nằm trong .cache/colab-home, giống ../colab.sh).

Yêu cầu: đã đăng nhập 1 lần:  bash colab.sh sessions   (mở link, đồng ý, dán mã vào terminal — người dùng tự làm)

    python scripts/colab_job.py all --gpu L4      # new → upload → setup → start → watch → fetch → stop
    python scripts/colab_job.py watch             # theo dõi lại nếu bị ngắt kết nối (train vẫn chạy trên Colab)
    python scripts/colab_job.py fetch             # tải trọng số về vision/weights/
    python scripts/colab_job.py stop              # LUÔN dừng phiên khi xong để không tốn compute unit

Upload: CLI gửi cả file một lần (base64) → không chịu được zip 1,1 GB, nên chia phần 64 MB, ghép lại trên VM
và kiểm SHA-256. Train chạy nền trên VM (mất kết nối không sao); `watch` giữ kernel bận + in tiến độ mỗi phút.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from nhandienqua import PROJECT_ROOT, VISION_ROOT, WEIGHTS_DIR

SESSION = "ndq"
# Gọi thẳng colab.exe (không qua `bash`: trên Windows lệnh `bash` có thể là WSL) với môi trường giống colab.sh
_CACHE = PROJECT_ROOT / ".cache"
COLAB = [str(_CACHE / "colab-venv" / "Scripts" / "colab.exe")]
COLAB_ENV = {
    **os.environ,
    "HOME": str(_CACHE / "colab-home"),
    "USERPROFILE": str(_CACHE / "colab-home"),
    "PYTHONPATH": str(_CACHE / "colab-shim"),  # shim termios cho Windows
    "TMP": str(_CACHE / "tmp"), "TEMP": str(_CACHE / "tmp"),
    "PIP_CACHE_DIR": str(_CACHE / "pip"), "UV_CACHE_DIR": str(_CACHE / "uv"),
    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
}
ZIPS = {name: VISION_ROOT / "colab" / f"ndq_{name}.zip" for name in ("quality", "detect")}
PART = 64 * 1024 * 1024
PARTS_DIR = PROJECT_ROOT / ".cache" / "tmp" / "colab_parts"


def colab(*args: str, stdin: str | None = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run([*COLAB, *args], input=stdin, text=True, encoding="utf-8", errors="replace",
                       capture_output=capture, cwd=PROJECT_ROOT, env=COLAB_ENV)
    if check and r.returncode != 0:
        raise SystemExit(f"Lỗi khi chạy: colab {' '.join(args)}\n{(r.stdout or '')[-2000:]}{(r.stderr or '')[-2000:]}")
    return r


def remote(code: str, timeout: float = 600, capture: bool = False) -> str:
    r = colab("exec", "-s", SESSION, "--timeout", str(timeout), stdin=code, capture=capture)
    return r.stdout or ""


def remote_json(code: str) -> dict:
    out = remote(code + "\nimport json as _j; print('@@' + _j.dumps(_res))", timeout=120, capture=True)
    line = next(x for x in out.splitlines() if x.startswith("@@"))
    return json.loads(line[2:])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ các bước
def step_new(gpu: str) -> None:
    out = colab("sessions", capture=True, check=False).stdout or ""
    if SESSION in out:
        print(f"Phiên '{SESSION}' đã tồn tại — dùng lại.")
        return
    colab("new", "-s", SESSION, "--gpu", gpu)


def split(p: Path) -> list[Path]:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    with open(p, "rb") as f:
        for i, chunk in enumerate(iter(lambda: f.read(PART), b"")):
            q = PARTS_DIR / f"{p.name}.part{i:03d}"
            if not q.exists() or q.stat().st_size != len(chunk):
                q.write_bytes(chunk)
            parts.append(q)
    return parts


def step_upload() -> None:
    have = remote_json("import os, glob\n_res = {os.path.basename(p): os.path.getsize(p) for p in glob.glob('/content/parts/*')}\n"
                       "os.makedirs('/content/parts', exist_ok=True)")
    for name, z in ZIPS.items():
        if not z.exists():
            raise SystemExit(f"Thiếu {z} — chạy scripts/pack_for_colab.py trước.")
        parts = split(z)
        for i, q in enumerate(parts, 1):
            if have.get(q.name) == q.stat().st_size:
                continue
            for attempt in range(1, 4):
                t = time.time()
                r = colab("upload", "-s", SESSION, str(q), f"/content/parts/{q.name}", check=False, capture=True)
                if r.returncode == 0:
                    print(f"  {q.name} ({i}/{len(parts)}) {q.stat().st_size / 1e6:.0f} MB · {time.time() - t:.0f}s", flush=True)
                    break
                print(f"  {q.name}: lỗi lần {attempt}, thử lại… {(r.stderr or '')[-300:]}", flush=True)
            else:
                raise SystemExit(f"Upload {q.name} thất bại 3 lần")
    colab("upload", "-s", SESSION, str(VISION_ROOT / "colab" / "train_remote.py"), "/content/train_remote.py")


def step_setup() -> None:
    hashes = {name: sha256(z) for name, z in ZIPS.items()}
    code = f"""
import glob, hashlib, os, subprocess, sys, zipfile, yaml
hashes = {json.dumps(hashes)}
for name, want in hashes.items():
    if os.path.exists(f'/content/data/{{name}}'):
        print(name, 'đã giải nén'); continue
    zpath = f'/content/ndq_{{name}}.zip'
    h = hashlib.sha256()
    with open(zpath, 'wb') as out:
        for p in sorted(glob.glob(f'/content/parts/ndq_{{name}}.zip.part*')):
            b = open(p, 'rb').read(); h.update(b); out.write(b)
    assert h.hexdigest() == want, f'SHA-256 không khớp cho {{name}}'
    zipfile.ZipFile(zpath).extractall('/content/data')
    os.remove(zpath)
    print(name, 'OK, SHA-256 khớp')
for y in ['data.yaml', 'data_synth_test.yaml', 'data_real_test.yaml']:
    p = f'/content/data/detect/{{y}}'
    d = yaml.safe_load(open(p, encoding='utf-8')); d['path'] = '/content/data/detect'
    yaml.safe_dump(d, open(p, 'w', encoding='utf-8'), allow_unicode=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.174'], check=True)
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ GPU')
"""
    remote(code, timeout=1800)


def step_start() -> None:
    remote("import os, subprocess\n"
           "if os.path.exists('/content/DONE'): print('Đã train xong trước đó')\n"
           "elif subprocess.run('pgrep -f [t]rain_remote.py', shell=True, capture_output=True).returncode == 0: print('Đang train rồi')\n"
           "else:\n"
           "    subprocess.Popen('nohup python /content/train_remote.py quality detect > /content/train.log 2>&1 &', shell=True)\n"
           "    print('Đã bắt đầu train nền')", timeout=60)


WATCH = r"""
import os, subprocess, time
def alive():
    return subprocess.run('pgrep -f [t]rain_remote.py', shell=True, capture_output=True).returncode == 0
while not (os.path.exists('/content/DONE') or os.path.exists('/content/FAILED')):
    if not alive():
        print('[ndq] tiến trình train không còn chạy', flush=True); break
    try:
        txt = open('/content/train.log', 'rb').read()[-6000:].decode('utf-8', 'replace').replace('\r', '\n')
        lines = [l for l in txt.split('\n') if l.strip()]
        print(time.strftime('%H:%M:%S'), (lines[-1] if lines else '')[:220], flush=True)
    except FileNotFoundError:
        print('chưa có log', flush=True)
    time.sleep(60)
print('KẾT QUẢ:', 'DONE' if os.path.exists('/content/DONE') else open('/content/FAILED').read()[-3000:] if os.path.exists('/content/FAILED') else 'DỪNG BẤT THƯỜNG — xem /content/train.log')
"""


def step_watch() -> None:
    """Websocket của CLI hay rớt sau ~15 phút → tự nối lại cho tới khi train xong (giữ kernel bận)."""
    for attempt in range(1, 200):
        r = subprocess.run([*COLAB, "exec", "-s", SESSION, "--timeout", "900"], input=WATCH, text=True,
                           encoding="utf-8", errors="replace", capture_output=True, cwd=PROJECT_ROOT, env=COLAB_ENV)
        out = r.stdout or ""
        print(out[-3000:], flush=True)
        if "KẾT QUẢ:" in out:
            return
        print(f"[watch] mất kết nối (lần {attempt}) — nối lại sau 10 s", flush=True)
        time.sleep(10)


def step_fetch() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("fruit_quality.pt", "fruit_det.pt"):
        colab("download", "-s", SESSION, f"/content/weights/{f}", str(WEIGHTS_DIR / f), check=False)
        print(f"{f}: {'OK' if (WEIGHTS_DIR / f).exists() else 'chưa có'}")
    for task in ("quality", "detect"):
        d = VISION_ROOT / "runs" / "colab" / task
        d.mkdir(parents=True, exist_ok=True)
        for f in ("results.csv", "results.png", "confusion_matrix_normalized.png", "args.yaml"):
            colab("download", "-s", SESSION, f"/content/runs/{task}/train/{f}", str(d / f), check=False, capture=True)
    colab("download", "-s", SESSION, "/content/train.log", str(VISION_ROOT / "runs" / "colab" / "train.log"), check=False, capture=True)


def step_stop() -> None:
    colab("stop", "-s", SESSION, check=False)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "new", "upload", "setup", "start", "watch", "fetch", "stop"])
    ap.add_argument("--gpu", default="L4", help="T4 | L4 | G4 | A100 | H100")
    a = ap.parse_args(argv)
    steps = {"new": lambda: step_new(a.gpu), "upload": step_upload, "setup": step_setup, "start": step_start,
             "watch": step_watch, "fetch": step_fetch, "stop": step_stop}
    if a.step != "all":
        steps[a.step]()
        return 0
    for name in ("new", "upload", "setup", "start", "watch", "fetch"):
        print(f"\n===== {name} =====", flush=True)
        steps[name]()
    print("\nĐã tải trọng số. Dừng phiên để không tốn compute unit:", flush=True)
    step_stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
