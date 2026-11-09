"""What the notebooks share (a facade over paths, hardware and subprocess runs), so each notebook cell stays short and the
same notebook runs in a local checkout and on Kaggle.

    from datamate.notebook import setup
    env = setup()                       # paths, GPUs, the settings this hardware can run
    env.run([env.python, "-m", "datamate.predict", ...], log=env.runs / "x.log")

Where things live:
    local   repo checkout with data/, models/, runs/ inside it (DATAMATE_DATA / DATAMATE_MODELS / DATAMATE_RUNS override)
    Kaggle  repo wherever it was cloned or attached. Competition data found under /kaggle/input. Base weights in /tmp/models
            (too big for the saved output). Runs in /kaggle/working/runs (kept after the session)
    Colab   Google Drive mounted, code and data in MyDrive/datamate/{code,data}. Base weights in /content/models. Runs in
            MyDrive/datamate/runs (kept after the session)
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle/input").exists()
ON_COLAB = not ON_KAGGLE and "COLAB_RELEASE_TAG" in os.environ
COLAB_DRIVE = Path("/content/drive/MyDrive/datamate")   # Colab: code/ and data/ uploaded to Google Drive under this folder
BASE_MODEL = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")  # repo id, pinned revision (docker/model_manifest.json)


@dataclass
class GPU:
    name: str
    mem_gb: float
    capability: tuple[int, int]

    @property
    def bf16(self) -> bool:      # Ampere and newer
        return self.capability >= (8, 0)

    @property
    def fp8(self) -> bool:       # Ada / Hopper / Blackwell: vLLM fp8 weights and fp8 KV cache
        return self.capability >= (8, 9)


def gpus() -> list[GPU]:
    """Visible GPUs from nvidia-smi (no torch import, so the notebook kernel stays light). DATAMATE_NO_GPU=1 hides them (to try the
    saved-result path of the GPU steps on a machine that has one)."""
    if os.environ.get("DATAMATE_NO_GPU"):
        return []
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    found = []
    for line in out.strip().splitlines():
        name, mem, cap = [x.strip() for x in line.split(",")]
        major, minor = cap.split(".")
        found.append(GPU(name, round(float(mem) / 1024, 1), (int(major), int(minor))))
    return found


def big_gpu(found: list[GPU] | None = None) -> bool:
    """A bf16 GPU with >= 22 GB visible: what inference, the notes step and training need (T4 / P100 are not)."""
    found = gpus() if found is None else found
    return bool(found) and min(found, key=lambda g: g.mem_gb).bf16 and min(g.mem_gb for g in found) >= 22


def find_data() -> Path | None:
    """The folder holding train/ and/or public_test/ (each with questions.jsonl)."""
    if os.environ.get("DATAMATE_DATA"):
        return Path(os.environ["DATAMATE_DATA"])
    if (REPO / "data").exists():
        return REPO / "data"
    if ON_COLAB and (COLAB_DRIVE / "data").exists():
        return COLAB_DRIVE / "data"
    if ON_KAGGLE:
        for q in sorted(Path("/kaggle/input").glob("**/questions.jsonl")):
            if q.parent.name in ("train", "public_test", "private_test"):
                return q.parent.parent
    return None


@dataclass
class Env:
    root: Path
    data: Path | None
    models: Path
    runs: Path
    gpus: list[GPU] = field(default_factory=list)
    python: str = sys.executable

    # ---- hardware -------------------------------------------------------------------------------------------------
    def inference_flags(self) -> tuple[list[str], str]:
        """The submitted flags (scripts/flags.sh) adapted to these GPUs, and the vLLM engine JSON. Raises when the GPUs cannot
        hold the model: bf16 weights of the 9B model are ~18 GB, so the pipeline needs one bf16 GPU with >= 24 GB."""
        flags, engine = submitted_flags(self.root)
        if not self.gpus:
            raise RuntimeError("no GPU visible: inference needs one (on Kaggle: Settings -> Accelerator)")
        g = min(self.gpus, key=lambda x: x.mem_gb)
        if not g.bf16 or g.mem_gb < 22:
            raise RuntimeError(f"{g.name} ({g.mem_gb} GB, sm{g.capability[0]}{g.capability[1]}) cannot run this pipeline: it needs a bf16 GPU "
                               "(Ampere or newer) with >= 24 GB per process. T4 / P100 do not qualify; use them for the CPU notebooks only.")
        if not g.fp8:  # Ampere: no fp8 weights or KV cache in vLLM. bf16 weights, auto KV cache, a shorter context to fit
            flags = _drop(flags, "--quant")
            flags = _set(flags, "--context", "32768" if g.mem_gb >= 40 else "16384")
            engine = engine.replace('"kv_cache_dtype":"fp8",', "")
        if g.mem_gb < 30:
            flags = _set(flags, "--gpu-mem", "0.92")
        return flags, engine

    @property
    def can_run_llm(self) -> bool:
        return big_gpu(self.gpus)

    def use_saved(self, name: str, dest: Path) -> Path:
        """Without a big enough GPU (Kaggle T4 / P100, a CPU session): copy the result of this step saved from a 24 GB GPU run
        (runs/saved/<name> in the checkout) to where the next cells read it, so the notebook still runs end to end."""
        src = self.root / "runs/saved" / name
        if not src.exists():
            raise RuntimeError(f"this step needs a bf16 GPU with >= 24 GB, and {src} (a saved result) is not in this checkout")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
        print(f"no bf16 GPU with >= 24 GB here: using the result saved from a 24 GB GPU run ({src.relative_to(self.root)})", flush=True)
        return dest

    def training_precision(self) -> str:
        return "bf16" if self.gpus and min(g.capability for g in self.gpus) >= (8, 0) else "fp16"

    # ---- running steps --------------------------------------------------------------------------------------------
    def run(self, cmd, log: Path | None = None, env: dict | None = None, show: str | None = None, every: float = 30.0):
        """Run one pipeline step as a subprocess from the repo root. All output goes to `log`. The notebook shows the lines
        matching `show` (a substring list separated by '|', default: progress and errors) and a heartbeat every `every` s."""
        cmd = [str(c) for c in cmd]
        log = Path(log) if log else None
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
        keys = (show or "Traceback|Error|error:|model ready|loaded|views of|[views]|[debts]|answered|steps in|'loss'|notes on|rows|PASS|FAIL|merged|format OK").split("|")
        print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
        t0 = last = time.time()
        with subprocess.Popen(cmd, cwd=self.root, env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, errors="replace") as p, (log.open("w", encoding="utf-8") if log else open(os.devnull, "w")) as f:
            for line in p.stdout:
                f.write(line)
                now = time.time()
                if any(k in line for k in keys) and "Warning" not in line:
                    print(line.rstrip()[:300], flush=True)
                    last = now
                elif now - last > every:  # heartbeat with the latest line (e.g. a progress bar)
                    print(f"  ... {now - t0:.0f}s | {line.strip()[:160]}", flush=True)
                    last = now
        if p.returncode:
            raise RuntimeError(f"step failed with exit code {p.returncode}" + (f"; full log: {log}" if log else ""))
        print(f"done in {time.time() - t0:.0f}s" + (f"; log: {log}" if log else ""), flush=True)

    def run_shards(self, make_cmd, n: int, out: Path, show: str | None = None, every: float = 60.0):
        """n processes at once, process i on GPU i (CUDA_VISIBLE_DEVICES=i), command make_cmd(i), output in out/shard<i>.log,
        the notebook shows their progress lines as they come."""
        keys = (show or "Traceback|Error|model ready|views of|[views]|[debts]| answered,|merged").split("|")
        out.mkdir(parents=True, exist_ok=True)
        procs = []
        for i in range(n):
            cmd = [str(c) for c in make_cmd(i)]
            print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
            log = (out / f"shard{i}.log").open("w", encoding="utf-8")
            procs.append((subprocess.Popen(cmd, cwd=self.root, env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(i)},
                                           stdout=log, stderr=subprocess.STDOUT), out / f"shard{i}.log"))
        seen, t0 = [0] * n, time.time()
        while True:
            done = all(p.poll() is not None for p, _ in procs)
            for i, (_, log) in enumerate(procs):
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in lines[seen[i]:]:
                    if any(k in line for k in keys) and "Warning" not in line:
                        print(f"[shard {i}] {line[:280]}", flush=True)
                seen[i] = len(lines)
            if done:
                break
            time.sleep(every)
            print(f"  ... {time.time() - t0:.0f}s", flush=True)
        bad = [i for i, (p, _) in enumerate(procs) if p.returncode]
        if bad:
            raise RuntimeError(f"shards {bad} failed; see {', '.join(str(procs[i][1]) for i in bad)}")
        print(f"{n} shard(s) done in {time.time() - t0:.0f}s", flush=True)

    def base_model(self) -> Path:
        """The pinned Qwen3.5-9B checkpoint, downloaded once from the Hugging Face Hub when missing (needs internet)."""
        dest = self.models / "Qwen3.5-9B"
        if not (dest / "config.json").exists():
            from huggingface_hub import snapshot_download
            print(f"downloading {BASE_MODEL[0]}@{BASE_MODEL[1][:8]} to {dest} (~19 GB)", flush=True)
            snapshot_download(BASE_MODEL[0], revision=BASE_MODEL[1], local_dir=dest)
        return dest

    def adapter(self, name="sql_rft_v3") -> Path:
        """A LoRA adapter: a name under adapters/ (Git LFS) or a path (e.g. a run of notebooks/05_train_lora.ipynb).
        A pointer file instead of weights means `git lfs pull` has not run."""
        p = Path(name) if Path(name).exists() else self.root / "adapters" / name
        w = p / "adapter_model.safetensors"
        if not w.exists() or w.stat().st_size < 1_000_000:
            raise RuntimeError(f"{w} is missing or an LFS pointer: run `git lfs install && git lfs pull` in {self.root}")
        return p


def install(kind: str = "base"):
    """Kaggle and Colab only (a local checkout has its own venvs): pip install what a notebook needs.
    base: the package (EDA, evaluation), llm: + vLLM (inference, notes), train: the training stack (Unsloth, fla) without vLLM."""
    if not (ON_KAGGLE or ON_COLAB):
        return
    if kind in ("llm", "train") and not big_gpu():
        print(f'no bf16 GPU with >= 24 GB: "{kind}" packages not installed, the steps that need them use saved results', flush=True)
        return
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    if kind == "train":
        subprocess.run(pip + ["-r", str(REPO / "training/requirements.txt")], check=True)
        subprocess.run(pip + ["--no-deps", "fla-core==0.6.0"], check=True)   # Gated DeltaNet kernels. Without them training falls back to a ~90x slower path
    else:
        subprocess.run(pip + ["-e", f"{REPO}[{'llm,dev' if kind == 'llm' else 'dev'}]"], check=True)


def submitted_flags(root: Path = REPO) -> tuple[list[str], str]:
    """DATAMATE_FLAGS and DATAMATE_ENGINE of scripts/flags.sh, read through bash so the notebooks never drift from it."""
    out = subprocess.run(["bash", "-c", 'source scripts/flags.sh >/dev/null; printf "%s\\n" "${DATAMATE_FLAGS[@]}"; printf "ENGINE=%s\\n" "$DATAMATE_ENGINE"'],
                         cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    return [x for x in out if not x.startswith("ENGINE=")], next(x[7:] for x in out if x.startswith("ENGINE="))


def _drop(flags, name):
    i = flags.index(name) if name in flags else -1
    return flags if i < 0 else flags[:i] + flags[i + 2:]


def _set(flags, name, value):
    return (flags[:flags.index(name) + 1] + [value] + flags[flags.index(name) + 2:]) if name in flags else flags + [name, value]


def setup(verbose: bool = True) -> Env:
    """Paths, GPUs and the repo on sys.path. Prints what was found."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    data = find_data()
    if data:
        os.environ["DATAMATE_DATA"] = str(data)      # scripts run from the notebooks (training/*.py) read the same data
    runs = Path(os.environ.get("DATAMATE_RUNS") or ("/kaggle/working/runs" if ON_KAGGLE else COLAB_DRIVE / "runs" if ON_COLAB else REPO / "runs"))
    models = Path(os.environ.get("DATAMATE_MODELS") or ("/tmp/models" if ON_KAGGLE else "/content/models" if ON_COLAB else REPO / "models"))
    env = Env(REPO, find_data(), models, runs, gpus())
    env.runs.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"{'Kaggle' if ON_KAGGLE else 'Colab' if ON_COLAB else 'local'} | repo {env.root} | data {env.data or 'NOT FOUND (set DATAMATE_DATA)'} | models {env.models} | runs {env.runs}")
        for g in env.gpus or []:
            print(f"GPU {g.name}: {g.mem_gb} GB, sm{g.capability[0]}{g.capability[1]}, bf16 {g.bf16}, fp8 {g.fp8}")
        if not env.gpus:
            print("no GPU: EDA and evaluation run; inference and training need one")
    return env
