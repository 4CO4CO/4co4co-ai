# Python 3.10+
import os, json, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------
# Dataset
# ---------------------------
class TensorFolder(Dataset):
    def __init__(self, folder):
        self.X = torch.load(Path(folder)/"features.pt")  # (N, 1024)
        self.Y = torch.load(Path(folder)/"labels.pt")    # (N, 17) float {0,1}
        assert self.X.size(0) == self.Y.size(0)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# ---------------------------
# Model: simple MLP 1024 -> 17
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=(512,256), out_dim=17, dropout=0.2):
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

# ---------------------------
# Metrics
# ---------------------------
def f1_from_logits(logits, y_true, thr=0.5, eps=1e-9):
    """multilabel micro/macro F1 with fixed threshold"""
    y_prob = torch.sigmoid(logits)
    y_pred = (y_prob >= thr).float()

    tp = (y_pred * y_true).sum(dim=0)
    fp = (y_pred * (1 - y_true)).sum(dim=0)
    fn = ((1 - y_pred) * y_true).sum(dim=0)

    prec_c = tp / (tp + fp + eps)
    rec_c  = tp / (tp + fn + eps)
    f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    f1_macro = f1_c.mean().item()

    tp_m = tp.sum(); fp_m = fp.sum(); fn_m = fn.sum()
    prec_m = tp_m / (tp_m + fp_m + eps)
    rec_m  = tp_m / (tp_m + fn_m + eps)
    f1_micro = (2 * prec_m * rec_m / (prec_m + rec_m + eps)).item()
    return f1_micro, f1_macro

# ---------------------------
# Train/Eval loop
# ---------------------------
def run_epoch(model, loader, criterion, opt=None, scaler=None, device="cpu"):
    is_train = opt is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_logits, all_labels = [], []

    for xb, yb in tqdm(loader, disable=len(loader)<5):
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()

        if is_train:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(yb.detach().cpu())

    avg_loss = total_loss / max(1, len(loader.dataset))
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    f1_micro, f1_macro = f1_from_logits(logits, labels, thr=0.5)
    return avg_loss, f1_micro, f1_macro

def compute_pos_weight(labels):  # (N, C)
    pos = labels.sum(dim=0)
    neg = labels.size(0) - pos
    pos = torch.clamp(pos, min=1.0)
    return neg / pos

def main():
    ap = argparse.ArgumentParser("Train MLP on precomputed CLIP features (1024 -> 17)")
    ap.add_argument("--root", required=True, help="preprocessed root (contains train/ val/ test/)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, nargs="*", default=[512,256])
    ap.add_argument("--pos_weight", action="store_true", help="use class-wise pos_weight for BCE")
    ap.add_argument("--early_stop", type=int, default=5, help="patience epochs")
    ap.add_argument("--out_dir", default="runs_mlp")
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # datasets
    dtrain = TensorFolder(Path(args.root)/"train")
    dval   = TensorFolder(Path(args.root)/"val") if (Path(args.root)/"val").exists() else None
    dtest  = TensorFolder(Path(args.root)/"test") if (Path(args.root)/"test").exists() else None
    print(f"Train: {len(dtrain)}  Val: {len(dval) if dval else 0}  Test: {len(dtest) if dtest else 0}")

    train_loader = DataLoader(dtrain, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(dval, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True) if dval else None
    test_loader  = DataLoader(dtest, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True) if dtest else None

    # model / loss
    in_dim = dtrain.X.size(1)
    out_dim = dtrain.Y.size(1)
    model = MLP(in_dim=in_dim, hidden=args.hidden, out_dim=out_dim, dropout=args.dropout).to(device)

    if args.pos_weight:
        pw = compute_pos_weight(dtrain.Y)  # (C,)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val = -1.0
    patience = args.early_stop
    best_path = Path(args.out_dir) / "best.pt"

    # train
    for ep in range(1, args.epochs+1):
        tr_loss, tr_f1m, tr_f1M = run_epoch(model, train_loader, criterion, opt, scaler, device)
        if val_loader:
            vl_loss, vl_f1m, vl_f1M = run_epoch(model, val_loader, criterion, None, None, device)
            print(f"[{ep:02d}] train loss {tr_loss:.4f} f1_micro {tr_f1m:.3f} f1_macro {tr_f1M:.3f} | "
                  f"val loss {vl_loss:.4f} f1_micro {vl_f1m:.3f} f1_macro {vl_f1M:.3f}")
            score = vl_f1m
            if score > best_val:
                best_val = score
                torch.save({
                    "state_dict": model.state_dict(),
                    "in_dim": in_dim, "out_dim": out_dim,
                    "hidden": args.hidden, "dropout": args.dropout
                }, best_path)
                print(f"✅ saved best → {best_path} (val micro-F1={best_val:.3f})")
                patience = args.early_stop
            else:
                patience -= 1
                if patience == 0:
                    print("⛔ Early stopping.")
                    break
        else:
            print(f"[{ep:02d}] train loss {tr_loss:.4f} f1_micro {tr_f1m:.3f} f1_macro {tr_f1M:.3f}")

    # final test
    if test_loader:
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
        else:
            print("⚠️ best.pt가 없어 현재 모델(last 상태)로 테스트합니다.")
        te_loss, te_f1m, te_f1M = run_epoch(model, test_loader, criterion, None, None, device)
        print(f"[TEST] loss {te_loss:.4f} f1_micro {te_f1m:.3f} f1_macro {te_f1M:.3f}")

    # save last
    last_path = Path(args.out_dir) / "last.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim, "out_dim": out_dim,
        "hidden": args.hidden, "dropout": args.dropout
    }, last_path)
    print(f"Done. last → {last_path}")

if __name__ == "__main__":
    main()
