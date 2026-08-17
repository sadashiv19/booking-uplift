"""
Backend for the HTML demo. Same model artifacts, same prediction logic as
the earlier Streamlit version -- only the presentation layer changed.

Routes:
  GET  /                     -> static/index.html   (customer profile screen)
  GET  /results.html         -> static/results.html (full results screen)
  GET  /figures/<file>       -> serves the notebook-generated PNGs
  POST /api/predict          -> {predictions per model, propensity, quadrant}
  GET  /api/results          -> final_results.json content
  GET  /api/qini             -> qini_curves.json content

Run from the repo root:  python web/server.py
"""

import os
import sys

# Must be set before torch or lightgbm are imported anywhere in this
# process, and only applies on macOS. Both libraries bundle their own
# OpenMP runtime; loading both into one process -- exactly what happens
# the moment a request calls both the LightGBM baselines and the PyTorch
# DragonNet -- causes the two runtimes to collide. On macOS this crashes
# the process with a hard segfault (no Python traceback) rather than a
# warning. This is a known, well-documented interaction between these
# specific libraries on macOS specifically (Linux's manylinux wheels don't
# hit this the same way, since they share a system OpenMP rather than each
# bundling their own) -- the standard fix is telling OpenMP to tolerate
# the duplicate rather than abort.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # KMP_DUPLICATE_LIB_OK alone only silences the abort-on-duplicate-load
    # warning -- it does not prevent two independently-initialized OpenMP
    # thread pools (one from torch, one from lightgbm) from actually
    # colliding during concurrent execution. Capping both to a single
    # thread removes the concurrency that triggers the collision in the
    # first place. VECLIB_MAXIMUM_THREADS covers Apple's Accelerate
    # framework, which numpy commonly links against on macOS and is a
    # third participant in this same class of conflict.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))
from model import AttentionDragonNet, FeatureScaler, predict_cate, predict_propensity  # noqa: E402

if sys.platform == "darwin":
    torch.set_num_threads(1)

MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

FEATURE_COLS = ["loyalty_tier", "price_sensitivity", "lead_time_days",
                 "past_bookings", "device_mobile", "length_of_stay", "base_price"]
MODEL_NAMES = ["S-learner (naive)", "T-learner (naive)", "T-learner + IPW", "Attention-DragonNet"]

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="")

_artifacts = None


def load_artifacts():
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    required = [
        MODELS_DIR / "s_learner.joblib", MODELS_DIR / "t_learner_treated.joblib",
        MODELS_DIR / "t_learner_control.joblib", MODELS_DIR / "ipw_t_learner_treated.joblib",
        MODELS_DIR / "ipw_t_learner_control.joblib", MODELS_DIR / "attention_dragonnet.pt",
        MODELS_DIR / "scaler.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts (run notebooks 01-04 first): " + ", ".join(missing)
        )

    with open(MODELS_DIR / "scaler.json") as f:
        scaler = FeatureScaler.from_dict(json.load(f))
    dragonnet = AttentionDragonNet(d_model=32, n_heads=4, n_layers=2)
    dragonnet.load_state_dict(torch.load(MODELS_DIR / "attention_dragonnet.pt", map_location="cpu"))
    dragonnet.eval()

    _artifacts = {
        "s_model": joblib.load(MODELS_DIR / "s_learner.joblib"),
        "t_model_1": joblib.load(MODELS_DIR / "t_learner_treated.joblib"),
        "t_model_0": joblib.load(MODELS_DIR / "t_learner_control.joblib"),
        "ipw_model_1": joblib.load(MODELS_DIR / "ipw_t_learner_treated.joblib"),
        "ipw_model_0": joblib.load(MODELS_DIR / "ipw_t_learner_control.joblib"),
        "dragonnet": dragonnet,
        "scaler": scaler,
    }
    return _artifacts


def classify_quadrant(ite_pred: float, threshold: float = 3.0):
    if ite_pred > threshold:
        return "persuadable", "Promo likely drives a genuinely incremental booking — good target."
    elif ite_pred < -threshold:
        return "sleeping_dog", "Promo likely cannibalizes a booking that would've happened anyway — avoid."
    elif ite_pred >= 0:
        return "sure_thing", "Small positive effect — likely books regardless, promo is low-value spend."
    else:
        return "lost_cause", "Small negative effect — unlikely to book either way."


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/results.html")
def results_page():
    return send_from_directory(app.static_folder, "results.html")


@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        artifacts = load_artifacts()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    body = request.get_json(force=True)
    profile_df = pd.DataFrame([{
        "loyalty_tier": int(body["loyalty_tier"]),
        "price_sensitivity": float(body["price_sensitivity"]),
        "lead_time_days": float(body["lead_time_days"]),
        "past_bookings": int(body["past_bookings"]),
        "device_mobile": int(body["device_mobile"]),
        "length_of_stay": int(body["length_of_stay"]),
        "base_price": float(body["base_price"]),
    }])

    s_features = FEATURE_COLS + ["treatment"]
    treated = profile_df[FEATURE_COLS].copy(); treated["treatment"] = 1
    control = profile_df[FEATURE_COLS].copy(); control["treatment"] = 0
    s_pred = float(artifacts["s_model"].predict(treated[s_features])[0]
                    - artifacts["s_model"].predict(control[s_features])[0])

    t_pred = float(artifacts["t_model_1"].predict(profile_df[FEATURE_COLS])[0]
                    - artifacts["t_model_0"].predict(profile_df[FEATURE_COLS])[0])

    ipw_pred = float(artifacts["ipw_model_1"].predict(profile_df[FEATURE_COLS])[0]
                      - artifacts["ipw_model_0"].predict(profile_df[FEATURE_COLS])[0])

    dragonnet_pred = float(predict_cate(artifacts["dragonnet"], profile_df, artifacts["scaler"])[0])
    propensity = float(predict_propensity(artifacts["dragonnet"], profile_df, artifacts["scaler"])[0])

    quadrant, explanation = classify_quadrant(dragonnet_pred)

    return jsonify({
        "predictions": {
            "S-learner (naive)": round(s_pred, 2),
            "T-learner (naive)": round(t_pred, 2),
            "T-learner + IPW": round(ipw_pred, 2),
            "Attention-DragonNet": round(dragonnet_pred, 2),
        },
        "propensity": round(propensity, 4),
        "quadrant": quadrant,
        "explanation": explanation,
    })


@app.route("/api/results")
def results():
    path = DATA_DIR / "final_results.json"
    if not path.exists():
        return jsonify({"error": "Run notebook 04 to generate final_results.json"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/qini")
def qini():
    path = DATA_DIR / "qini_curves.json"
    if not path.exists():
        return jsonify({"error": "Run notebook 04 to generate qini_curves.json"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    load_artifacts()  # fail fast at startup if artifacts are missing
    app.run(host="0.0.0.0", port=5050, debug=False)
