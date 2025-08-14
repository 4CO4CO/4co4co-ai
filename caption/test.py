# tools_eval_infer.py
# Python 3.10+
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score
)
from transformers import CLIPProcessor, CLIPModel


# ---------------------------
# MLP (학습 시 사용한 것과 동일 구조)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=(512, 256), out_dim=17, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(True), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


EMOTIONS_17 = [
    "Happiness", "Confidence", "Surprise", "Pain", "Disquietment",
    "Fear", "Yearning", "Excitement", "Embarrassment", "Affection",
    "Aversion", "Engagement", "Anticipation", "Sensitivity",
    "Annoyance", "Sympathy", "Pleasure"
]


# ---------------------------
# 유틸
# ---------------------------
def load_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MLP(
        in_dim=ckpt["in_dim"],
        hidden=ckpt["hidden"],
        out_dim=ckpt["out_dim"],
        dropout=ckpt["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def evaluate_from_features(model, feat_path, lbl_path, thresholds=None):
    X = torch.load(feat_path)           # (N, 1024)
    Y = torch.load(lbl_path).numpy()    # (N, 17) float {0,1}

    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).numpy()

    if thresholds is None:
        thresholds = np.full((probs.shape[1],), 0.5, dtype=np.float32)

    preds = (probs >= thresholds[None, :]).astype(int)

    metrics = {
        "exact_match_accuracy": accuracy_score(Y, preds),
        "hamming_loss": hamming_loss(Y, preds),
        "jaccard_macro": jaccard_score(Y, preds, average="macro"),
        "macro_precision": precision_score(Y, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(Y, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(Y, preds, average="macro", zero_division=0),
        "micro_precision": precision_score(Y, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(Y, preds, average="micro", zero_division=0),
        "micro_f1": f1_score(Y, preds, average="micro", zero_division=0),
    }

    per_class = {}
    for i, name in enumerate(EMOTIONS_17):
        per_class[name] = {
            "precision": precision_score(Y[:, i], preds[:, i], zero_division=0),
            "recall": recall_score(Y[:, i], preds[:, i], zero_division=0),
            "f1": f1_score(Y[:, i], preds[:, i], zero_division=0),
            "support": int(Y[:, i].sum()),
            "pred_pos": int(preds[:, i].sum()),
            "threshold": float(thresholds[i]),
        }

    return metrics, per_class, probs, preds, Y


def tune_thresholds(model, val_feat_path, val_lbl_path, metric="f1"):
    Xv = torch.load(val_feat_path)
    Yv = torch.load(val_lbl_path).numpy()

    with torch.no_grad():
        logits = model(Xv)
        Pv = torch.sigmoid(logits).numpy()

    thresholds = np.zeros((Pv.shape[1],), dtype=np.float32)
    scores = np.zeros_like(thresholds)

    for i in range(Pv.shape[1]):
        best_t, best_s = 0.5, 0.0
        pv = Pv[:, i]
        yv = Yv[:, i]
        for t in np.arange(0.1, 0.95, 0.05):
            pred = (pv >= t).astype(int)
            if metric == "precision":
                s = precision_score(yv, pred, zero_division=0)
            elif metric == "recall":
                s = recall_score(yv, pred, zero_division=0)
            else:
                s = f1_score(yv, pred, zero_division=0)
            if s > best_s:
                best_s, best_t = s, t
        thresholds[i] = best_t
        scores[i] = best_s

    return thresholds, scores


def build_clip(device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_proc


def encode_image_text_1024(clip_model, clip_proc, image_path, text, device):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_proc(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        img_feat = clip_model.get_image_features(inputs["pixel_values"])  # (1,512)
        txt_feat = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )  # (1,512)
    feats = torch.cat([img_feat, txt_feat], dim=1).float()  # (1,1024)
    return feats


# ---------------------------
# Subcommands
# ---------------------------
def cmd_eval(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model, _ = load_checkpoint(args.ckpt, device="cpu")  # features는 CPU 텐서로 저장되어 있음

    thresholds = None
    if args.thresholds and Path(args.thresholds).exists():
        thresholds = np.load(args.thresholds)
        print(f"Loaded thresholds: {args.thresholds}")
    else:
        print("Use fixed threshold 0.5 for all classes.")

    feat = Path(args.dataset) / "features.pt"
    lbl = Path(args.dataset) / "labels.pt"
    metrics, per_class, probs, preds, Y = evaluate_from_features(model, feat, lbl, thresholds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overall_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2))

    print("\n[Overall]")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"\nSaved: {out_dir/'overall_metrics.json'}")
    print(f"Saved: {out_dir/'per_class_metrics.json'}")


def cmd_tune(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model, _ = load_checkpoint(args.ckpt, device="cpu")

    val_feat = Path(args.val_dir) / "features.pt"
    val_lbl = Path(args.val_dir) / "labels.pt"

    thresholds, scores = tune_thresholds(model, val_feat, val_lbl, metric=args.metric)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, thresholds)
    print(f"Saved thresholds → {out} (metric={args.metric})")


def cmd_infer(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model, _ = load_checkpoint(args.ckpt, device=device)
    clip_model, clip_proc = build_clip(device)

    feats = encode_image_text_1024(clip_model, clip_proc, args.image, args.caption, device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        probs = torch.sigmoid(model(feats)).cpu().numpy()[0]

    if args.thresholds and Path(args.thresholds).exists():
        ths = np.load(args.thresholds)
    else:
        ths = np.full((len(EMOTIONS_17),), 0.5, dtype=np.float32)

    preds = (probs >= ths).astype(int)
    pred_labels = [EMOTIONS_17[i] for i, p in enumerate(preds) if p == 1]

    print("\n[Scores]")
    for name, p in sorted(zip(EMOTIONS_17, probs), key=lambda x: x[1], reverse=True)[:args.top_k]:
        print(f"  {name:15s}: {p:.3f}")

    print("\n[Predicted labels]")
    print(pred_labels)


def cmd_batch_infer(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model, _ = load_checkpoint(args.ckpt, device=device)
    clip_model, clip_proc = build_clip(device)

    if args.thresholds and Path(args.thresholds).exists():
        ths = np.load(args.thresholds)
    else:
        ths = np.full((len(EMOTIONS_17),), 0.5, dtype=np.float32)

    folder = Path(args.folder)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # caption: 공통 캡션 하나를 모든 이미지에 쓰거나, 파일명과 동일한 .txt가 있으면 그 내용을 사용
    rows = []
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    images = []
    for e in exts:
        images += list(folder.rglob(f"*{e}"))

    print(f"Found {len(images)} images.")

    for img_path in images:
        # 캡션 결정
        cap = args.caption
        if cap is None:
            cap_file = img_path.with_suffix(".txt")
            if cap_file.exists():
                cap = cap_file.read_text(encoding="utf-8").strip()
            else:
                cap = ""  # 빈 캡션 허용(텍스트 임베딩은 거의 0 정보가 되지만 파이프라인은 유지)

        feats = encode_image_text_1024(clip_model, clip_proc, str(img_path), cap, device)
        with torch.no_grad():
            probs = torch.sigmoid(model(feats)).cpu().numpy()[0]

        pred = (probs >= ths).astype(int)
        top_idx = int(np.argmax(probs))
        rows.append({
            "image_path": str(img_path),
            "caption": cap,
            "top_emotion": EMOTIONS_17[top_idx],
            "top_score": float(probs[top_idx]),
            **{f"score_{em}": float(probs[i]) for i, em in enumerate(EMOTIONS_17)},
            "pred_labels": ",".join([EMOTIONS_17[i] for i, p in enumerate(pred) if p == 1])
        })

    # 저장 (pandas 미의존 – JSONL로 저장)
    out_jsonl = out_csv.with_suffix(".jsonl")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    except Exception:
        print(f"Pandas가 없거나 오류 발생. JSONL 저장만 수행: {out_jsonl}")


# ---------------------------
# Main (subparsers)
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Eval / Tune thresholds / Inference for CLIP+MLP emotion model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # eval
    sp = sub.add_parser("eval", help="features.pt / labels.pt 기반 평가")
    sp.add_argument("--ckpt", required=True, help="runs_mlp/best.pt")
    sp.add_argument("--dataset", required=True, help="clip_preprocessed_data/test or val")
    sp.add_argument("--thresholds", default=None, help="npy thresholds (optional)")
    sp.add_argument("--out_dir", default="eval_out", help="output dir")
    sp.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    sp.set_defaults(func=cmd_eval)

    # tune
    sp = sub.add_parser("tune", help="검증셋으로 per-class threshold 튜닝")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--val_dir", required=True, help="clip_preprocessed_data/val")
    sp.add_argument("--metric", choices=["f1", "precision", "recall"], default="f1")
    sp.add_argument("--out_path", default="runs_mlp/val_opt_thresholds.npy")
    sp.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    sp.set_defaults(func=cmd_tune)

    # infer (single)
    sp = sub.add_parser("infer", help="이미지+캡션 단일 추론")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--image", required=True)
    sp.add_argument("--caption", required=True)
    sp.add_argument("--thresholds", default=None)
    sp.add_argument("--top_k", type=int, default=5)
    sp.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    sp.set_defaults(func=cmd_infer)

    # batch_infer
    sp = sub.add_parser("batch_infer", help="폴더 배치 추론(공통 캡션 또는 파일별 .txt)")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--folder", required=True)
    sp.add_argument("--caption", default=None, help="공통 캡션(없으면 같은 이름의 .txt 시도)")
    sp.add_argument("--thresholds", default=None)
    sp.add_argument("--out_csv", default="predictions.csv")
    sp.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    sp.set_defaults(func=cmd_batch_infer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
