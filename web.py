#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SecScan mini-portal: upload JSON reports, view SAST/DAST tables,
and request OpenAI remediation hints.
"""

from __future__ import annotations
import os, json, datetime
from pathlib import Path
from typing import Dict, List, Any

import openai
from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, send_from_directory, flash
)

BASE_DIR       = Path(__file__).resolve().parent
DEFAULT_REPORT = BASE_DIR / "report.json"
UPLOAD_DIR     = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-4o-mini"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY","secscan-demo-secret")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# ── helpers ──────────────────────────────────────────────────────────────
def load_report(path: Path) -> List[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                                 # noqa: BLE001
        app.logger.warning("Cannot read %s: %s", path, exc)
        return []

def split_findings(lst: List[Dict]) -> tuple[list, list]:
    return ([f for f in lst if f.get("type")=="SAST"],
            [f for f in lst if f.get("type")=="DAST"])

def render_front(sast: list, dast: list, filename: str):
    return render_template(
        "index.html",
        filename=filename,
        scan_date=datetime.datetime.now().strftime("%B %d, %Y"),
        sast=sast,
        dast=dast
    )

# ── routes ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST" and "file" in request.files:
        f = request.files["file"]
        if not f or not f.filename.endswith(".json"):
            flash("Please upload a valid .json file.","error")
            return redirect(url_for("index"))
        dst = UPLOAD_DIR / f.filename
        f.save(dst)
        flash(f"Uploaded {f.filename}.","success")
        return redirect(url_for("report", filename=f.filename))

    findings = load_report(DEFAULT_REPORT) if DEFAULT_REPORT.exists() else []
    return render_front(*split_findings(findings), filename="report.json")

@app.route("/report/<path:filename>")
def report(filename):
    path = (UPLOAD_DIR/filename).resolve()
    if not path.is_file():
        flash("Report not found.","error")
        return redirect(url_for("index"))
    return render_front(*split_findings(load_report(path)), filename=filename)

@app.route("/uploads/<path:filename>")
def uploads(filename): return send_from_directory(UPLOAD_DIR, filename)

@app.route("/remediate", methods=["POST"])
def remediate():
    if not openai.api_key:
        return jsonify(error="OpenAI API key not configured"), 400

    finding: Dict[str,Any] = request.get_json(silent=True) or {}
    if not finding: return jsonify(error="Invalid finding data"), 400

    prompt = (
        "You are a senior application security engineer.\n\n"
        "Provide:\n"
        "• A concise explanation of the vulnerability.\n"
        "• Step-by-step remediation (bullet list).\n"
        "• A code example showing the fix, if applicable.\n"
        "• Additional best-practice tips.\n\n"
        f"Finding JSON:\n{json.dumps(finding, indent=2)}"
    )

    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You give precise remediation advice."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
        )
        html = resp.choices[0].message.content.strip()
        return jsonify(html=html)
    except Exception as exc:                                 # noqa: BLE001
        app.logger.warning("OpenAI error: %s", exc)
        return jsonify(error=f"OpenAI error: {exc}"), 500

# ── error handlers ───────────────────────────────────────────────────────
@app.errorhandler(404)
def _404(_): flash("Page not found.","error"); return redirect(url_for("index"))

@app.errorhandler(413)
def _413(_): flash("File too large (5 MB limit).","error"); return redirect(url_for("index"))

@app.errorhandler(500)
def _500(_): flash("Internal server error.","error"); return redirect(url_for("index"))

# ── run ───────────────────────────────────────────────────────────────────
if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
