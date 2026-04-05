"""
solution_phase2_final.py
========================
Full 20-class disease classification with:
  - ViT-B/16 backbone + EMA
  - Warmup + OneCycleLR fine-tuning
  - Hard-negative triplet loss for confusable old classes (11, 13)
  - 50-view augmented prototype building with outlier filtering
  - CLIP prompt ensembling for new classes (15-20)
  - Prototype repulsion: push 15 & 20 away from class-11 centroid
  - Proto score calibration (z-score) before head/proto concat
  - Two-stage inference: unified embedding decision for {11, 15, 20}
  - Test-Time Augmentation (TTA, 10 views)
  - LOO temperature tuning on support set
  - Confusion diagnostic logger
"""

import random, time, os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ───────────────────────────── CONFIG ─────────────────────────────
CFG = dict(
    seed=42,
    img_size=224,
    batch_size=32,
    epochs=15,
    warmup_epochs=5,

    lr_head=5e-4,
    lr_backbone=3e-5,

    proto_temp=20.0,          # will be tuned via LOO
    clip_weight=0.25,

    ema_decay=0.999,
    triplet_weight=0.3,
    triplet_margin=0.3,

    tta_views=10,
    proto_views=50,

    device="cuda" if torch.cuda.is_available() else "cpu",
)

CLASSES_OLD = [f"disease{i}" for i in range(1, 15)]
CLASSES_NEW = [f"disease{i}" for i in range(15, 21)]
CLASSES_ALL = CLASSES_OLD + CLASSES_NEW
C2I = {c: i for i, c in enumerate(CLASSES_ALL)}
I2C = {i: c for c, i in C2I.items()}

# Classes that are visually confusable
CONFUSABLE_OLD = [C2I["disease11"], C2I["disease13"]]
TRIO_IDS       = {C2I["disease11"], C2I["disease15"], C2I["disease20"]}

# ───────────────────────────── SEED ──────────────────────────────
def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

seed_all(CFG["seed"])

# ───────────────────────────── AUGMENTATIONS ─────────────────────
S = CFG["img_size"]

train_tfm = A.Compose([
    A.RandomResizedCrop((S, S), scale=(0.85, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ColorJitter(0.15, 0.15, 0.1, 0.02, p=0.5),
    A.Normalize([0.485]*3, [0.229]*3),
    ToTensorV2()
])

val_tfm = A.Compose([
    A.Resize(S, S),
    A.Normalize([0.485]*3, [0.229]*3),
    ToTensorV2()
])

# Aggressive augmentation for few-shot prototype building & TTA
few_shot_tfm = A.Compose([
    A.RandomResizedCrop((S, S), scale=(0.6, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.ColorJitter(0.3, 0.3, 0.2, 0.05, p=0.7),
    A.GaussianBlur(p=0.3),
    A.GridDistortion(p=0.3),
    A.Normalize([0.485]*3, [0.229]*3),
    ToTensorV2()
])

# ───────────────────────────── DATASET ───────────────────────────
class DS(Dataset):
    def __init__(self, paths, labels=None, tfm=None):
        self.p, self.l, self.t = paths, labels, tfm

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = np.array(Image.open(self.p[i]).convert("RGB"))
        img = self.t(image=img)["image"]
        if self.l is None:
            return img, Path(self.p[i]).name
        return img, self.l[i]

# ───────────────────────────── MODEL ─────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True,
            num_classes=0
        )
        dim = self.backbone.num_features
        self.head = nn.Linear(dim, 14)   # old 14 classes only

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)

    def feat(self, x):
        return F.normalize(self.backbone(x), dim=-1)

# ───────────────────────────── EMA ───────────────────────────────
class EMA:
    def __init__(self, model, decay):
        self.model  = model
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay  = decay

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self):
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

# ───────────────────────────── TRIPLET MINER ─────────────────────
class HardTripletMiner:
    """
    Online hard triplet mining restricted to confusable old classes.
    Pulls hard positives closer, pushes hard negatives (from same
    confusable set) further apart.
    """
    def __init__(self, confusable_ids, margin=0.3):
        self.conf = set(confusable_ids)
        self.margin = margin

    def mine(self, feats, labels):
        dist  = torch.cdist(feats, feats)   # [B, B]
        loss  = torch.tensor(0.0, device=feats.device)
        count = 0
        idxs  = torch.arange(len(labels), device=labels.device)

        for i in range(len(labels)):
            if labels[i].item() not in self.conf:
                continue
            pos_mask = (labels == labels[i]) & (idxs != i)
            neg_mask = (labels != labels[i]) & \
                       torch.tensor([l.item() in self.conf for l in labels],
                                    device=labels.device)
            if not pos_mask.any() or not neg_mask.any():
                continue

            hard_pos = dist[i][pos_mask].max()
            hard_neg = dist[i][neg_mask].min()
            loss    += F.relu(hard_pos - hard_neg + self.margin)
            count   += 1

        return loss / max(count, 1)

miner = HardTripletMiner(CONFUSABLE_OLD, margin=CFG["triplet_margin"])

# ───────────────────────────── TRAINING ──────────────────────────
def run_epoch(model, loader, opt, scaler, ema=None,
              scheduler=None, train=True):
    model.train() if train else model.eval()
    preds, labs = [], []

    for x, y in loader:
        x, y = x.to(CFG["device"]), y.to(CFG["device"])

        with torch.amp.autocast("cuda"):
            f      = model.backbone(x)
            logits = model.head(f)
            ce     = F.cross_entropy(logits, y, label_smoothing=0.05)

            # triplet loss only for confusable old classes
            f_norm   = F.normalize(f, dim=-1)
            tri_loss = miner.mine(f_norm, y)
            loss     = ce + CFG["triplet_weight"] * tri_loss

        if train:
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if scheduler: scheduler.step()
            if ema:        ema.update()

        preds += logits.argmax(1).cpu().tolist()
        labs  += y.cpu().tolist()

    return accuracy_score(labs, preds)

# ───────────────────────────── PROTOTYPES ────────────────────────
@torch.no_grad()
def build_prototypes(model, paths, labels):
    """
    50-view augmented prototype per image with outlier view filtering.
    """
    model.eval()
    feats = {c: [] for c in set(labels)}

    for p, l in zip(paths, labels):
        img   = np.array(Image.open(p).convert("RGB"))
        views = []
        for _ in range(CFG["proto_views"]):
            aug = few_shot_tfm(image=img)["image"].unsqueeze(0).to(CFG["device"])
            views.append(model.feat(aug)[0].cpu())

        stack  = torch.stack(views)                         # [50, D]
        center = stack.mean(0)
        dists  = ((stack - center) ** 2).sum(1)
        keep   = dists < dists.mean() + dists.std()        # drop outliers
        feats[l].append(stack[keep].mean(0))

    protos = {}
    for k, v in feats.items():
        protos[k] = F.normalize(torch.stack(v).mean(0), dim=0).to(CFG["device"])

    return protos

# ───────────────────────────── CLIP ──────────────────────────────
CLIP_TEMPLATES = [
    "microscopy image of {}",
    "histology slide showing {}",
    "medical image of {}",
    "a photo of {}, a disease",
    "clinical photograph of {}",
    "dermoscopy image of {}",
]

def build_clip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tok   = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(CFG["device"]).eval()

    text_feats = {}
    for cls in CLASSES_NEW:
        all_f = []
        for tmpl in CLIP_TEMPLATES:
            txt = tok([tmpl.format(cls)]).to(CFG["device"])
            with torch.no_grad():
                f = F.normalize(model.encode_text(txt), dim=-1)
            all_f.append(f[0])
        # ensemble across prompt templates
        text_feats[C2I[cls]] = F.normalize(
            torch.stack(all_f).mean(0), dim=0
        )

    def img_fn(x):
        with torch.no_grad():
            return F.normalize(model.encode_image(x), dim=-1)

    return text_feats, img_fn

# ───────────────────────────── TRIO SEPARATOR ────────────────────
@torch.no_grad()
def build_trio_centroids(model, old_paths, old_labels, protos):
    """
    Unified embedding-space centroids for {11, 15, 20}.
    Class 11: averaged over ALL its training images.
    Classes 15, 20: taken directly from prototypes.
    """
    feats_11 = []
    for p, l in zip(old_paths, old_labels):
        if l != C2I["disease11"]: continue
        img = val_tfm(image=np.array(Image.open(p).convert("RGB")))["image"]
        f   = model.feat(img.unsqueeze(0).to(CFG["device"]))
        feats_11.append(f[0].cpu())

    c11 = F.normalize(torch.stack(feats_11).mean(0), dim=0).to(CFG["device"])

    return {
        C2I["disease11"]: c11,
        C2I["disease15"]: protos[C2I["disease15"]],
        C2I["disease20"]: protos[C2I["disease20"]],
    }

def trio_predict(f_norm, centroids):
    """Nearest-centroid within the confusable trio."""
    best_cls, best_sim = None, -999.0
    for cls_id, c in centroids.items():
        sim = (f_norm * c).sum().item()
        if sim > best_sim:
            best_sim = sim
            best_cls = cls_id
    return best_cls

# ───────────────────────────── PROTOTYPE REPULSION ───────────────
def repulse_protos(protos, c11, steps=300, lr=0.005):
    """
    Gradient-based refinement: push protos[15] and protos[20]
    away from c11 and away from each other, with anchor regularisation.
    """
    p15_orig = protos[C2I["disease15"]].detach().clone()
    p20_orig = protos[C2I["disease20"]].detach().clone()

    p15 = nn.Parameter(protos[C2I["disease15"]].clone())
    p20 = nn.Parameter(protos[C2I["disease20"]].clone())
    opt = torch.optim.Adam([p15, p20], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        n15  = F.normalize(p15, dim=0)
        n20  = F.normalize(p20, dim=0)
        c11n = F.normalize(c11, dim=0)

        loss  = F.relu((n15 * c11n).sum() + 0.3)   # push 15 from 11
        loss += F.relu((n20 * c11n).sum() + 0.3)   # push 20 from 11
        loss += F.relu((n15 * n20).sum()  + 0.2)   # push 15 from 20
        # anchor: don't drift too far from original proto
        loss += 0.5 * (1 - (n15 * F.normalize(p15_orig, dim=0)).sum())
        loss += 0.5 * (1 - (n20 * F.normalize(p20_orig, dim=0)).sum())

        loss.backward()
        opt.step()

    protos[C2I["disease15"]] = F.normalize(p15.detach(), dim=0)
    protos[C2I["disease20"]] = F.normalize(p20.detach(), dim=0)
    return protos

# ───────────────────────────── SCORE CALIBRATION ─────────────────
@torch.no_grad()
def build_cal_stats(model, new_paths, new_labels, protos):
    """
    Compute per-new-class mean & std of proto similarity score
    on the support set, for z-score normalisation at inference.
    """
    new_ids = sorted(protos.keys())
    P       = torch.stack([protos[i] for i in new_ids])
    stats   = {}

    for p, l in zip(new_paths, new_labels):
        img = val_tfm(image=np.array(Image.open(p).convert("RGB")))["image"]
        f   = model.feat(img.unsqueeze(0).to(CFG["device"]))
        sim = ((P @ f.T).T * CFG["proto_temp"])[0]
        idx = new_ids.index(l)
        stats.setdefault(l, []).append(sim[idx].item())

    return {k: (np.mean(v), max(np.std(v), 1e-6))
            for k, v in stats.items()}

# ───────────────────────────── LOO TEMP TUNING ───────────────────
@torch.no_grad()
def loo_tune_temp(model, new_paths, new_labels):
    """
    Leave-one-out cross-validation over support set to find best
    proto_temp. Only uses new-class images (5 per class).
    """
    best_acc, best_temp = 0.0, CFG["proto_temp"]

    for temp in [5, 10, 20, 40, 80, 100]:
        correct = 0
        for i in range(len(new_paths)):
            sup_p = new_paths[:i] + new_paths[i+1:]
            sup_l = new_labels[:i] + new_labels[i+1:]
            p_loo = build_prototypes(model, sup_p, sup_l)
            P_loo = torch.stack([p_loo[k] for k in sorted(p_loo.keys())])
            new_ids_loo = sorted(p_loo.keys())

            img = val_tfm(image=np.array(
                Image.open(new_paths[i]).convert("RGB")))["image"]
            f   = model.feat(img.unsqueeze(0).to(CFG["device"]))
            sim = ((P_loo @ f.T).T * temp)[0]
            pred_idx = sim.argmax().item()
            pred_cls = new_ids_loo[pred_idx]
            correct += int(pred_cls == new_labels[i])

        acc = correct / len(new_paths)
        print(f"  LOO temp={temp}: acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_temp = acc, temp

    print(f"  → Best temp: {best_temp}  (acc={best_acc:.3f})")
    return best_temp

# ───────────────────────────── DIAGNOSTIC ────────────────────────
@torch.no_grad()
def confusion_diagnostic(model, test_paths, protos, clip_txt,
                          clip_img, cal_stats, n_samples=30):
    """
    Print score breakdown for predictions in {11, 15, 20}.
    Run this on a labelled val set to understand which branch wins.
    """
    WATCH   = {C2I["disease11"], C2I["disease15"], C2I["disease20"]}
    new_ids = sorted(protos.keys())
    P       = torch.stack([protos[i] for i in new_ids])

    for path in test_paths[:n_samples]:
        img = val_tfm(image=np.array(
            Image.open(path).convert("RGB")))["image"].unsqueeze(0).to(CFG["device"])

        f        = model.feat(img)
        old_prob = F.softmax(model(img), dim=-1)[0]
        sim      = (P @ f.T).T * CFG["proto_temp"]
        cf       = clip_img(img)
        clip_s   = torch.stack([cf @ clip_txt[i] for i in new_ids], dim=1)
        sim      = (1 - CFG["clip_weight"]) * sim + CFG["clip_weight"] * clip_s
        new_prob = F.softmax(sim, dim=-1)[0]

        scores = torch.cat([old_prob, new_prob])
        pred   = scores.argmax().item()

        if pred in WATCH:
            print(f"\n{path.name}  →  predicted: {I2C[pred]}")
            print(f"  disease11 head  score : {old_prob[C2I['disease11']]:.4f}")
            i15 = new_ids.index(C2I["disease15"])
            i20 = new_ids.index(C2I["disease20"])
            print(f"  disease15 proto score : {new_prob[i15]:.4f}")
            print(f"  disease20 proto score : {new_prob[i20]:.4f}")

# ───────────────────────────── INFERENCE HELPER ──────────────────
@torch.no_grad()
def infer_single(img_np, model, P, new_ids, clip_txt, clip_img,
                 cal_stats, trio_centroids, use_tta=True):
    """
    Full inference pipeline for one image (numpy RGB).
    Returns predicted class index.
    """
    tfms = [few_shot_tfm] * CFG["tta_views"] if use_tta else [val_tfm]

    all_old, all_new = [], []

    for tfm in tfms:
        img_t = tfm(image=img_np)["image"].unsqueeze(0).to(CFG["device"])
        f     = model.feat(img_t)

        old_logits = model(img_t)
        old_prob   = F.softmax(old_logits, dim=-1)

        sim    = (P @ f.T).T * CFG["proto_temp"]
        cf     = clip_img(img_t)
        clip_s = torch.stack([cf @ clip_txt[i] for i in new_ids], dim=1)
        sim    = (1 - CFG["clip_weight"]) * sim + CFG["clip_weight"] * clip_s

        # z-score calibrate each new-class score
        sim_cal = sim.clone()
        for j, cls_id in enumerate(new_ids):
            if cls_id in cal_stats:
                mu, sigma = cal_stats[cls_id]
                sim_cal[0, j] = (sim[0, j] - mu) / sigma

        new_prob = F.softmax(sim_cal, dim=-1)

        all_old.append(old_prob)
        all_new.append(new_prob)

    old_avg = torch.stack(all_old).mean(0)
    new_avg = torch.stack(all_new).mean(0)
    scores  = torch.cat([old_avg, new_avg], dim=1)   # [1, 20]
    pred    = scores.argmax().item()

    # Two-stage: re-decide in unified embedding space for confusable trio
    if pred in TRIO_IDS:
        # use clean val_tfm for stable feature
        img_t = val_tfm(image=img_np)["image"].unsqueeze(0).to(CFG["device"])
        f     = model.feat(img_t)
        pred  = trio_predict(f[0], trio_centroids)

    return pred

# ───────────────────────────── MAIN ──────────────────────────────
def main():
    # ── data paths ──
    old_paths  = sum([list(Path("hour0_train/" + c).glob("*"))
                      for c in CLASSES_OLD], [])
    old_labels = sum([[C2I[c]] * len(list(Path("hour0_train/" + c).glob("*")))
                      for c in CLASSES_OLD], [])

    new_paths  = sum([list(Path("phase2_support/" + c).glob("*"))
                      for c in CLASSES_NEW], [])
    new_labels = sum([[C2I[c]] * len(list(Path("phase2_support/" + c).glob("*")))
                      for c in CLASSES_NEW], [])

    tr_p, va_p, tr_l, va_l = train_test_split(
        old_paths, old_labels, stratify=old_labels, test_size=0.1
    )

    tr_loader = DataLoader(DS(tr_p, tr_l, train_tfm),
                           batch_size=CFG["batch_size"], shuffle=True)
    va_loader = DataLoader(DS(va_p, va_l, val_tfm),
                           batch_size=CFG["batch_size"])

    # ── model ──
    model  = Net().to(CFG["device"])
    scaler = torch.amp.GradScaler("cuda")
    ema    = EMA(model, CFG["ema_decay"])

    # ── warmup: head only ──
    for p in model.backbone.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(model.head.parameters(), lr=CFG["lr_head"])
    for e in range(CFG["warmup_epochs"]):
        acc = run_epoch(model, tr_loader, opt, scaler, train=True)
        print(f"warmup {e}: train_acc={acc:.4f}")

    # ── fine-tune: full model + triplet loss ──
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": CFG["lr_backbone"]},
        {"params": model.head.parameters(),     "lr": CFG["lr_head"]}
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[CFG["lr_backbone"], CFG["lr_head"]],
        epochs=CFG["epochs"],
        steps_per_epoch=len(tr_loader)
    )

    for e in range(CFG["epochs"]):
        tr_acc = run_epoch(model, tr_loader, opt, scaler, ema, scheduler, True)
        va_acc = run_epoch(model, va_loader, opt, scaler, train=False)
        print(f"epoch {e:02d}: train={tr_acc:.4f}  val={va_acc:.4f}")

    # ── swap in EMA weights ──
    ema.apply()
    
    # ── create checkpoints directory ──
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # ── save model checkpoint after training ──
    model_checkpoint_path = checkpoint_dir / "best_vit_phase2.pth"
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"✅ Model checkpoint saved → {model_checkpoint_path}")

    # ── build prototypes (50-view, outlier-filtered) ──
    print("\nBuilding prototypes...")
    protos = build_prototypes(model, new_paths, new_labels)

    # ── build CLIP text features (prompt ensemble) ──
    print("Building CLIP features...")
    clip_txt, clip_img_fn = build_clip()

    # ── build trio centroids (unified embedding space) ──
    print("Building trio centroids for {11, 15, 20}...")
    trio_centroids = build_trio_centroids(model, old_paths, old_labels, protos)

    # ── repulse protos[15] and protos[20] from class-11 centroid ──
    print("Repulsing confusable prototypes...")
    protos = repulse_protos(protos, trio_centroids[C2I["disease11"]])

    # ── rebuild trio centroids with refined protos ──
    trio_centroids[C2I["disease15"]] = protos[C2I["disease15"]]
    trio_centroids[C2I["disease20"]] = protos[C2I["disease20"]]

    # ── z-score calibration stats ──
    print("Computing calibration stats...")
    cal_stats = build_cal_stats(model, new_paths, new_labels, protos)

    # ── LOO temperature tuning ──
    print("LOO temperature tuning...")
    best_temp = loo_tune_temp(model, new_paths, new_labels)
    CFG["proto_temp"] = best_temp
    
    # ── stack proto matrix for inference (define new_ids before checkpoint) ──
    new_ids = sorted(protos.keys())
    
    # ── save prototypes and metadata ──
    proto_checkpoint_path = checkpoint_dir / "phase2_prototypes.pth"
    torch.save({
        'protos': protos,
        'clip_txt': clip_txt,
        'trio_centroids': trio_centroids,
        'cal_stats': cal_stats,
        'proto_temp': CFG["proto_temp"],
        'new_ids': new_ids,
    }, proto_checkpoint_path)
    print(f"✅ Prototypes checkpoint saved → {proto_checkpoint_path}")
    
    # ── save complete checkpoint for resuming ──
    complete_checkpoint_path = checkpoint_dir / "phase2_complete_checkpoint.pth"
    torch.save({
        'model': model.state_dict(),
        'protos': protos,
        'clip_txt': clip_txt,
        'trio_centroids': trio_centroids,
        'cal_stats': cal_stats,
        'config': CFG,
        'proto_temp': CFG["proto_temp"],
        'new_ids': new_ids,
    }, complete_checkpoint_path)
    print(f"✅ Complete checkpoint saved → {complete_checkpoint_path}")

    # ── stack proto matrix for inference ──
    P = torch.stack([protos[i] for i in new_ids])

    # ── optional: confusion diagnostic on val set ──
    # confusion_diagnostic(model, va_p, protos, clip_txt,
    #                      clip_img_fn, cal_stats)

    # ── test inference ──
    print("\nRunning inference...")
    test_paths = sorted(Path("phase2_test_20").glob("*.jpg"))
    names, preds = [], []

    for path in tqdm(test_paths):
        img_np = np.array(Image.open(path).convert("RGB"))
        pred   = infer_single(
            img_np, model, P, new_ids,
            clip_txt, clip_img_fn,
            cal_stats, trio_centroids,
            use_tta=True
        )
        names.append(path.name)
        preds.append(pred)

    # ── save submission ──
    df = pd.DataFrame({
        "image_name":      names,
        "predicted_class": [I2C[p] for p in preds]
    })
    df.to_csv("submission.csv", index=False)
    print(f"\nDONE — {len(df)} predictions saved to submission.csv")
    print(df["predicted_class"].value_counts())


if __name__ == "__main__":
    main()