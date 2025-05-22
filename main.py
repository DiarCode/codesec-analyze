#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
secscan.py ─ Bandit + CodeBERT-GBM SAST  |  ZAP-baseline DAST
──────────────────────────────────────────────────────────────
• Adds contextual snippets for Bandit findings
• Fills missing CWE IDs with "N/A"
• NEW: optional --host-network flag (Linux-only) to run ZAP with
       `--network host`; otherwise auto-maps localhost→host.docker.internal
"""

from __future__ import annotations
import argparse, json, multiprocessing as mp, os, shutil, subprocess, sys, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import joblib, numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from sklearn.metrics import precision_recall_fscore_support

console = Console()

# ────────────────── CONFIG ──────────────────
ROOT_DIR  = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "codebert_vuln_model.joblib"

DATASET   = "suriya7/Vulnerability-Classify-TP-FP"
TEXT_FLD  = "Vulnerability"
LABEL_FLD = "Return Status"

ZAP_IMAGE = os.getenv("ZAP_DOCKER_IMAGE", "zaproxy/zap-stable")
# ────────────────────────────────────────────


###############################################################################
# 1. Dataclass for uniform findings
###############################################################################
@dataclass
class Finding:
    type: str                 # SAST | DAST
    tool: str                 # bandit | ml | zap
    file: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    endpoint: Optional[str] = None
    cwe_id: str = ""
    severity: str = ""
    confidence: str | float = ""
    description: str = ""
    snippet: Optional[str] = None
    evidence: Optional[str] = None
    solution: Optional[str] = None

    def as_dict(self) -> Dict:
        d = asdict(self)
        if not d.get("cwe_id") or d["cwe_id"] in ("-1", ""):
            d["cwe_id"] = "N/A"
        return {k: v for k, v in d.items() if v not in (None, "", [])}


###############################################################################
# 2. Model training / caching  (unchanged)
###############################################################################
def _ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    from datasets import load_dataset
    import torch
    from transformers import AutoTokenizer, AutoModel
    from lightgbm import LGBMClassifier

    console.print(f"[cyan]⏬  Loading dataset {DATASET} …")
    ds = load_dataset(DATASET, split="train")
    ds = ds.map(lambda x: {"label": 1 if x[LABEL_FLD].strip().upper() == "TP" else 0})
    tr, va = ds.train_test_split(test_size=0.2, seed=42).values()

    console.print("[cyan]⏱️  Embedding with CodeBERT …")
    tk, bert = (AutoTokenizer.from_pretrained("microsoft/codebert-base"),
                AutoModel.from_pretrained("microsoft/codebert-base"))
    bert.eval()

    def embed(batch):
        toks = tk(batch[TEXT_FLD], truncation=True, padding="max_length",
                  max_length=256, return_tensors="pt")
        with torch.no_grad():
            cls = bert(**{k: v for k, v in toks.items()}).last_hidden_state[:, 0, :]
        return {"emb": cls.cpu().numpy()}

    tr = tr.map(embed, batched=True, batch_size=32)
    va = va.map(embed, batched=True, batch_size=32)
    X_tr, y_tr = np.vstack(tr["emb"]), np.array(tr["label"])
    X_va, y_va = np.vstack(va["emb"]), np.array(va["label"])

    clf = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                         n_jobs=-1, random_state=42)
    console.print("[cyan]🚂  Training LightGBM …")
    clf.fit(X_tr, y_tr)
    p, r, f1, _ = precision_recall_fscore_support(y_va, clf.predict(X_va),
                                                 average="binary")
    console.print(f"[green]✅  val  prec:{p:.2f}  rec:{r:.2f}  f1:{f1:.2f}")
    joblib.dump((tk, bert, clf), MODEL_PATH)
    console.print(f"[green]💾  Model saved ⇒ {MODEL_PATH}")


###############################################################################
# 3. ML prediction helpers (unchanged logic)
###############################################################################
_TK = _BERT = _CLF = None
def _load_model_once():
    global _TK, _BERT, _CLF
    if _TK is None:
        import torch  # noqa: F401
        _TK, _BERT, _CLF = joblib.load(MODEL_PATH)
        _BERT.eval()


def _ml_worker(args: Tuple[str, float]) -> Optional[Finding]:
    fp_str, thr = args
    from pathlib import Path
    import torch
    _load_model_once()
    text = Path(fp_str).read_text(encoding="utf-8", errors="ignore")
    toks = _TK(text, truncation=True, padding="max_length",
               max_length=256, return_tensors="pt")
    with torch.no_grad():
        cls = _BERT(**{k: v for k, v in toks.items()}).last_hidden_state[:, 0, :]
    prob = float(_CLF.predict_proba(cls.cpu().numpy())[0, 1])
    if prob < thr:
        return None
    sev = "HIGH" if prob > 0.85 else "MEDIUM" if prob > 0.65 else "LOW"
    return Finding(type="SAST", tool="ml", file=fp_str,
                   start_line=1, end_line=text.count("\n") + 1,
                   confidence=round(prob, 3), severity=sev,
                   description="Potential vulnerability detected by CodeBERT model")


def _predict_probs(files: List[Path], thr: float) -> List[Finding]:
    if not MODEL_PATH.exists():
        console.print("[yellow]⚠️  No ML model – run `secscan.py train` first")
        return []
    work = [(str(f), thr) for f in files]
    outs: List[Finding] = []
    with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                  TimeElapsedColumn()) as bar:
        task = bar.add_task("[cyan]ML scanning …", total=len(files))
        with mp.Pool(initializer=_load_model_once) as pool:
            for res in pool.imap_unordered(_ml_worker, work):
                if res:
                    outs.append(res)
                bar.advance(task)
    return outs


###############################################################################
# 4. Bandit SAST  (adds snippet extraction)
###############################################################################
def _get_snippet(fp: Path, line_no: int, ctx: int = 1) -> str:
    try:
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    start = max(line_no - 1 - ctx, 0)
    end = min(line_no + ctx, len(lines))
    return "\n".join(
        (">" if idx == line_no - 1 else " ") + f"{idx + 1:4d}: {lines[idx]}"
        for idx in range(start, end)
    )


def _bandit_scan(root: Path) -> List[Finding]:
    cmd = ["bandit", "-r", str(root), "-f", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        console.print("[yellow]⚠️  Bandit not installed – skipping")
        return []
    if proc.returncode not in (0, 1):
        console.print(proc.stderr or "Bandit failed")
        return []
    data = json.loads(proc.stdout or "{}")
    return [
        Finding(
            type="SAST",
            tool="bandit",
            file=i["filename"],
            start_line=i["line_number"],
            end_line=i["line_number"],
            cwe_id=str(i.get("issue_cwe", {}).get("id", "")),
            severity=i["issue_severity"],
            confidence=i["issue_confidence"],
            description=i["issue_text"],
            snippet=_get_snippet(Path(i["filename"]), i["line_number"]),
        )
        for i in data.get("results", [])
    ]


def analyze_sast(path: str, ml: bool, thr: float) -> List[Finding]:
    root = Path(os.path.expanduser(path)).resolve()
    if not root.is_dir():
        raise ValueError(f"path not found: {root}")
    fnds = _bandit_scan(root)
    if ml:
        fnds += _predict_probs(list(root.rglob("*.py")), thr)
    return fnds


###############################################################################
# 5. ZAP DAST (patched with host-network / host-gateway logic)
###############################################################################
def _docker_available() -> bool:
    return shutil.which("docker") is not None


def analyze_dast(target: str,
                 max_time: int = 60,
                 host_net: bool = False) -> List[Finding]:
    """Run ZAP baseline scan.

    * If host_net=True   ➜ use --network host  (Linux only)
    * Else               ➜ map localhost → host.docker.internal automatically
    """
    if not _docker_available():
        console.print("[yellow]⚠️  Docker not found – skipping DAST")
        return []

    # ── hostname rewrite for localhost targets ────────────────────────────
    add_host_flag: List[str] = []
    parsed = urlparse(target)
    if parsed.hostname in {"localhost", "127.0.0.1"} and not host_net:
        fixed_netloc = parsed.netloc.replace(parsed.hostname,
                                             "host.docker.internal")
        target = urlunparse(parsed._replace(netloc=fixed_netloc))
        console.print(f"[cyan]🔄  Rewriting target to {target} "
                      "(host.docker.internal mapping)")
        add_host_flag = ["--add-host",
                         "host.docker.internal:host-gateway"]

    # ── output dir & command ──────────────────────────────────────────────
    tmp = Path(tempfile.mkdtemp(prefix="zap_"))
    rpt = tmp / "zap.json"
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{tmp}:/zap/wrk",
        *(["--network", "host"] if host_net else []),
        *add_host_flag,
        "-t", ZAP_IMAGE,
        "zap-baseline.py",
        "-t", target,
        "-m", str(max_time),
        "-J", rpt.name,
        "-I",
    ]

    console.print("[cyan]🌐  Running ZAP baseline scan …")
    if subprocess.run(cmd, text=True).returncode not in (0, 1, 2):
        console.print("[red]ZAP scan failed")
        return []
    if not rpt.exists():
        console.print("[red]ZAP report missing")
        return []

    data = json.load(rpt.open())
    findings: List[Finding] = []
    for site in data.get("site", []):
        for a in site.get("alerts", []):
            uri = a["instances"][0]["uri"] if a.get("instances") else ""
            findings.append(
                Finding(
                    type="DAST", tool="zap",
                    endpoint=uri,
                    cwe_id=str(a.get("cweid", "")) or "N/A",
                    severity=a.get("riskdesc", ""),
                    confidence=a.get("confidence", ""),
                    description=a.get("name", ""),
                    evidence=a.get("evidence", ""),
                    solution=a.get("solution", ""),
                )
            )
    return findings


###############################################################################
# 6. Utilities  (unchanged)
###############################################################################
def _merge_dupes(fnds: List[Finding]) -> List[Finding]:
    uniq: Dict[Tuple, Finding] = {}
    for f in fnds:
        key = (f.type, f.tool, f.file or f.endpoint, f.start_line)
        uniq.setdefault(key, f)
    return list(uniq.values())


def write_report(fnds: List[Finding], out_file: str) -> None:
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump([f.as_dict() for f in fnds], fp, indent=2)
    console.print(f"[green]📝  Report → {out_file}  ({len(fnds)} findings)")


###############################################################################
# 7. CLI
###############################################################################
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("secscan.py – Bandit+ML SAST | ZAP DAST")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Download dataset and build ML model")

    an = sub.add_parser("analyze", help="Run scans")
    an.add_argument("--path", help="directory to scan (SAST)")
    an.add_argument("--target", help="URL to scan (DAST)")
    an.add_argument("--no-ml", action="store_true")
    an.add_argument("--threshold", type=float, default=0.65)
    an.add_argument("--host-network", action="store_true",
                    help="Run ZAP with --network host (Linux only)")
    an.add_argument("--out", default="report.json")
    an.add_argument("--exit-zero", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _cli()

    if args.cmd == "train":
        _ensure_model()
        return

    if args.cmd == "analyze":
        findings: List[Finding] = []
        if args.path:
            findings += analyze_sast(args.path, ml=not args.no_ml,
                                     thr=args.threshold)
        if args.target:
            findings += analyze_dast(args.target,
                                     host_net=args.host_network)
        findings = _merge_dupes(findings)
        write_report(findings, args.out)

        high = sum(1 for f in findings if "HIGH" in f.severity.upper())
        if high and not args.exit_zero:
            sys.exit(min(high, 255))
        return

    console.print("[red]No command specified – use -h for help")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    main()
