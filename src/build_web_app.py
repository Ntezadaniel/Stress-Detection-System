"""
build_web_app.py
-----------------
Packages the two trained models (outputs/demo_teacher.pt, outputs/demo_student.pt)
and the three HELD-OUT wearers' recordings into ONE self-contained web page:

    web_demo/index.html

The page runs both neural networks in the browser (a JavaScript port of the
PyTorch model), so it needs no server, works offline, and can be hosted on any
free static host. A built-in self-test compares the browser's output with
PyTorch's output on a few windows, and the page shows the result.

Run:   python src\\build_web_app.py
Then:  double-click web_demo\\index.html
"""

import argparse
import base64
import copy
import io
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from wesad_core import StressClassifier, load_raw_windows, downsample, LABEL_TO_INDEX

CHANNELS = ["BVP (pulse)", "EDA (skin conductance)", "Skin temperature", "Movement"]
TRANSPARENT_PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def load_model(path):
    ck = torch.load(path, weights_only=False)
    model = StressClassifier(ck["input_channels"], ck["hidden_size"], ck["num_layers"],
                             conv_channels=ck["conv_channels"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def pack_model(model, ck):
    sd = model.state_dict()
    order = ["conv.0.weight", "conv.0.bias", "conv.3.weight", "conv.3.bias"]
    for layer in range(ck["num_layers"]):
        order += [f"lstm.weight_ih_l{layer}", f"lstm.weight_hh_l{layer}",
                  f"lstm.bias_ih_l{layer}", f"lstm.bias_hh_l{layer}"]
    order += ["classifier.weight", "classifier.bias"]
    chunks, manifest, offset = [], [], 0
    for name in order:
        t = sd[name].detach().cpu().numpy().astype("<f4")
        manifest.append({"name": name, "shape": list(t.shape), "offset": offset})
        offset += t.size
        chunks.append(t.reshape(-1))
    return {"b64": b64(np.concatenate(chunks)), "manifest": manifest,
            "hidden": ck["hidden_size"], "layers": ck["num_layers"], "conv": ck["conv_channels"],
            "in": ck["input_channels"], "params": model.count_parameters()}


def student_footprint(student, X_eval):
    """32-bit size, plus an 8-bit size (measured if PyTorch supports it, else estimated)."""
    n = student.count_parameters()
    info = {"student_fp32_kb": n * 4 / 1024, "student_int8_kb": n / 1024,
            "int8_method": "estimated", "int8_agreement": None}
    try:
        warnings.filterwarnings("ignore")
        q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(student).eval(),
                                                   {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
        buf = io.BytesIO()
        torch.save(q.state_dict(), buf)
        x = torch.tensor(X_eval, dtype=torch.float32)
        with torch.no_grad():
            agree = (student(x).argmax(1) == q(x).argmax(1)).float().mean().item()
        info.update(student_int8_kb=buf.getbuffer().nbytes / 1024, int8_method="measured",
                    int8_agreement=round(agree * 100, 1))
    except Exception as exc:  # quantisation support differs between PyTorch versions
        print(f"  (8-bit quantisation not available here: {type(exc).__name__}; using an estimate)")
    return info


def normalise_full(Xs):
    mean = Xs.mean(axis=(0, 1), keepdims=True)
    std = Xs.std(axis=(0, 1), keepdims=True) + 1e-8
    return (Xs - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/raw_windows.npz")
    parser.add_argument("--teacher", default="outputs/demo_teacher.pt")
    parser.add_argument("--student", default="outputs/demo_student.pt")
    parser.add_argument("--template", default=str(Path(__file__).with_name("web_template.html")))
    parser.add_argument("--out", default="web_demo/index.html")
    parser.add_argument("--calib-windows", type=int, default=10)
    args = parser.parse_args()

    teacher, tck = load_model(args.teacher)
    student, sck = load_model(args.student)
    X, y_raw, subject_ids = load_raw_windows(args.data)
    X = downsample(X, sck["downsample"]).astype(np.float64)
    y = np.array([LABEL_TO_INDEX[v] for v in y_raw])
    heldout = list(sck["heldout_subjects"])
    print(f"Held-out wearers in the demo: {heldout}")

    subjects, first_norm = {}, None
    for sid in heldout:
        idx = np.where(subject_ids == sid)[0]
        Xs = X[idx]
        subjects[sid] = {"n": int(len(idx)), "T": int(Xs.shape[1]), "C": int(Xs.shape[2]),
                         "b64": b64(Xs), "labels": [int(v) for v in y[idx]]}
        if first_norm is None:
            first_norm = (sid, normalise_full(Xs))
        print(f"  {sid}: {len(idx)} windows")

    # Reference outputs from PyTorch, so the page can prove the JavaScript port is faithful.
    sid0, Xn0 = first_norm
    picks = sorted(set([0, len(Xn0) // 3, (2 * len(Xn0)) // 3, len(Xn0) - 1]))
    with torch.no_grad():
        xt = torch.tensor(Xn0[picks], dtype=torch.float32)
        ref_s = student(xt).numpy().tolist()
        ref_t = teacher(xt).numpy().tolist()

    foot = student_footprint(student, Xn0)
    meta = {"channels": CHANNELS, "calib_windows": args.calib_windows, "heldout": heldout,
            "teacher_params": teacher.count_parameters(), "student_params": student.count_parameters(),
            "teacher_kb": teacher.count_parameters() * 4 / 1024, **foot,
            "penalty": sck.get("penalty"), "epochs": sck.get("epochs")}
    payload = {"meta": meta, "teacher": pack_model(teacher, tck), "student": pack_model(student, sck),
               "subjects": subjects,
               "selftest": {"subject": sid0, "windows": picks, "student": ref_s, "teacher": ref_t}}

    html = Path(args.template).read_text(encoding="utf-8")
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__APP_DATA__", data_json).replace("__LOGO_B64__", TRANSPARENT_PNG)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nWrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("Open it by double-clicking, or host the web_demo folder on any static host.")


if __name__ == "__main__":
    main()
