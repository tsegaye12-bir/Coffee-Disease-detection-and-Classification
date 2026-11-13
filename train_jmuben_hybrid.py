#!/usr/bin/env python3
"""JMuBEN five-class coffee leaf disease experiment: CNN baselines + hybrid fusion.

Input layout: DATA_ROOT/<class_name>/*.(jpg|jpeg|png)
Experiments: EfficientNet-B0, MobileNetV3-Large, ResNet-50, hybrid
EfficientNet-B0 + MobileNetV3-Large feature fusion, and a soft-voting
ensemble of the three single-backbone models. All models train >=30 epochs.

Exact SHA-256 duplicates are removed before stratified splitting; cross-class
hash conflicts are excluded and logged. Augmentation is online and train-only.
Captum is optional; when installed, it generates Grad-CAM, Integrated Gradients,
occlusion maps and a simple top-10%-pixel deletion confidence-drop summary.

Install: pip install torch torchvision scikit-learn pandas matplotlib pillow captum
Run: python train_jmuben_hybrid.py --data-root /path/to/JMuBEN --output jmuben_results
"""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_NAMES = ["EfficientNetB0", "MobileNetV3Large", "ResNet50", "HybridEfficientMobile"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def audit_exact_duplicates(base):
    """Keep one image per decoded RGB pixel array; exclude cross-label conflicts."""
    by_hash = defaultdict(list)
    for idx, (path, label) in enumerate(base.samples):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            shape = f"{rgb.width}x{rgb.height}:".encode()
            h = hashlib.sha256(shape + rgb.tobytes()).hexdigest()
        by_hash[h].append((idx, label, path))
    keep, dropped_same, conflict_groups = [], [], []
    for group in by_hash.values():
        labels = {item[1] for item in group}
        if len(labels) > 1:
            conflict_groups.append(group)
            continue
        keep.append(group[0][0])
        dropped_same.extend(item[0] for item in group[1:])
    return sorted(keep), dropped_same, conflict_groups


class ImageSubset(Dataset):
    def __init__(self, base, indices, size, train=False):
        self.base, self.indices, self.size, self.train = base, list(indices), size, train
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.25),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        path, y = self.base.samples[idx]
        with Image.open(path) as im:
            x = self.tf(im.convert("RGB"))
        return x, y


def create_single(name, nclass, pretrained=True):
    if name == "EfficientNetB0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nclass)
        for p in m.parameters(): p.requires_grad = False
        for p in m.features[-1].parameters(): p.requires_grad = True
        for p in m.classifier.parameters(): p.requires_grad = True
        groups = [p for p in m.features[-1].parameters() if p.requires_grad]
        groups += list(m.classifier.parameters())
        return m, groups[:len([p for p in m.features[-1].parameters() if p.requires_grad])], list(m.classifier.parameters())
    if name == "MobileNetV3Large":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nclass)
        for p in m.parameters(): p.requires_grad = False
        for p in m.features[-1].parameters(): p.requires_grad = True
        for p in m.classifier.parameters(): p.requires_grad = True
        last = [p for p in m.features[-1].parameters() if p.requires_grad]
        return m, last, list(m.classifier.parameters())
    if name == "ResNet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, nclass)
        for p in m.parameters(): p.requires_grad = False
        for p in m.layer4.parameters(): p.requires_grad = True
        for p in m.fc.parameters(): p.requires_grad = True
        return m, list(m.layer4.parameters()), list(m.fc.parameters())
    raise ValueError(name)


class HybridEfficientMobile(nn.Module):
    def __init__(self, nclass, pretrained=True):
        super().__init__()
        we = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        wm = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.efficient = models.efficientnet_b0(weights=we)
        self.mobile = models.mobilenet_v3_large(weights=wm)
        d1 = self.efficient.classifier[-1].in_features
        # With the full classifier replaced by Identity, MobileNetV3 emits the
        # pooled feature vector before classifier[0] (960 for V3-Large).
        d2 = self.mobile.classifier[0].in_features
        self.efficient.classifier = nn.Identity()
        self.mobile.classifier = nn.Identity()
        for p in self.efficient.parameters(): p.requires_grad = False
        for p in self.mobile.parameters(): p.requires_grad = False
        for p in self.efficient.features[-1].parameters(): p.requires_grad = True
        for p in self.mobile.features[-1].parameters(): p.requires_grad = True
        self.fusion = nn.Sequential(
            nn.Linear(d1 + d2, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(256, nclass),
        )
        self.feature_dim = d1 + d2

    def forward(self, x):
        a = self.efficient(x)
        b = self.mobile(x)
        return self.fusion(torch.cat((a, b), dim=1))


def make_model(name, nclass, pretrained=True):
    if name == "HybridEfficientMobile":
        m = HybridEfficientMobile(nclass, pretrained)
        enc_a = [p for p in m.efficient.features[-1].parameters() if p.requires_grad]
        enc_b = [p for p in m.mobile.features[-1].parameters() if p.requires_grad]
        return m, enc_a + enc_b, list(m.fusion.parameters())
    return create_single(name, nclass, pretrained)


def make_weights(labels, nclass):
    counts = np.bincount(np.asarray(labels), minlength=nclass)
    # Inverse square-root weighting is less extreme than inverse-frequency weighting.
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE), counts.tolist()


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, ys, preds, logits_all = 0.0, [], [], []
    n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses += loss.item() * len(y); n += len(y)
        ys.extend(y.detach().cpu().tolist())
        preds.extend(logits.argmax(1).detach().cpu().tolist())
        logits_all.extend(logits.detach().cpu().numpy().tolist())
    return {"loss": losses / max(1, n), "accuracy": accuracy_score(ys, preds),
            "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
            "ys": ys, "preds": preds, "logits": np.asarray(logits_all)}


def save_curves(name, history, out):
    e = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(e, history["train_acc"], label="Train")
    ax[0].plot(e, history["val_acc"], label="Validation")
    ax[0].set(xlabel="Epoch", ylabel="Accuracy", title=f"{name}: accuracy")
    ax[1].plot(e, history["train_loss"], label="Train")
    ax[1].plot(e, history["val_loss"], label="Validation")
    ax[1].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title=f"{name}: loss")
    for a in ax: a.grid(alpha=.25); a.legend()
    fig.tight_layout(); fig.savefig(out / f"{name}_curves.png", dpi=180); plt.close(fig)


def evaluate_and_save(name, run, classes, out):
    y, p = run["ys"], run["preds"]
    rep = classification_report(y, p, labels=list(range(len(classes))),
                                target_names=classes, output_dict=True, zero_division=0)
    (out / f"{name}_classification_report.json").write_text(json.dumps(rep, indent=2))
    cm = confusion_matrix(y, p, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7.5, 6.5)); im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted", ylabel="True", title=f"{name}: test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out / f"{name}_confusion_matrix.png", dpi=180); plt.close(fig)
    pr, re, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    wpr, wre, wf1, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {"model": name, "accuracy": accuracy_score(y,p), "macro_precision": pr,
            "macro_recall": re, "macro_f1": f1, "weighted_precision": wpr,
            "weighted_recall": wre, "weighted_f1": wf1}


def target_layer(model, name):
    if name == "HybridEfficientMobile": return model.efficient.features[-1]
    if name == "EfficientNetB0": return model.features[-1]
    if name == "MobileNetV3Large": return model.features[-1]
    if name == "ResNet50": return model.layer4[-1]
    raise ValueError(name)


def normalize_map(m):
    m = m.detach().float().abs().squeeze().cpu()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def run_xai(best_name, checkpoint, base, test_indices, classes, image_size, out, samples_per_class=5):
    try:
        from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam, Occlusion
    except ImportError:
        print("Captum not installed; skipping XAI. Install `captum` and rerun XAI stage.")
        return
    ck = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model, _, _ = make_model(best_name, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(DEVICE).eval()
    test_set = ImageSubset(base, test_indices, image_size, train=False)
    by_class = defaultdict(list)
    for pos, idx in enumerate(test_indices):
        by_class[base.targets[idx]].append(pos)
    method_rows = []
    for class_id, positions in sorted(by_class.items()):
        for pos in positions[:samples_per_class]:
            x, label = test_set[pos]
            x = x.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(x), 1)
                pred = int(probs.argmax(1).item())
                original_prob = float(probs[0, pred].item())
            gradcam = LayerGradCam(model, target_layer(model, best_name))
            raw_gc = gradcam.attribute(x, target=pred)
            gc = LayerAttribution.interpolate(raw_gc, (image_size, image_size), interpolate_mode="bilinear")
            gradcam_map = normalize_map(gc)
            ig = IntegratedGradients(model)
            ig_attr = ig.attribute(x, baselines=torch.zeros_like(x), target=pred,
                                   n_steps=24, internal_batch_size=8)
            ig_map = normalize_map(ig_attr.abs().sum(dim=1, keepdim=True))
            occ = Occlusion(model)
            occ_attr = occ.attribute(x, target=pred,
                                     strides=(3, 24, 24),
                                     sliding_window_shapes=(3, 32, 32),
                                     baselines=0.0, perturbations_per_eval=8)
            occ_map = normalize_map(occ_attr.abs().sum(dim=1, keepdim=True))
            image = x[0].detach().cpu().permute(1,2,0).numpy() * np.array(STD) + np.array(MEAN)
            image = np.clip(image, 0, 1)
            maps = {"GradCAM": gradcam_map, "IntegratedGradients": ig_map, "Occlusion": occ_map}
            for method, amap in maps.items():
                arr = amap.numpy()
                k = max(1, int(.10 * arr.size))
                mask = torch.zeros_like(amap, dtype=torch.bool)
                mask.view(-1)[torch.topk(amap.view(-1), k).indices] = True
                deleted = x.clone()
                deleted[:, :, mask.squeeze(0).to(DEVICE)] = 0.0
                with torch.no_grad():
                    after = float(torch.softmax(model(deleted),1)[0,pred].item())
                method_rows.append({"model": best_name, "true_class": classes[label],
                                    "predicted_class": classes[pred],
                                    "method": method, "original_confidence": original_prob,
                                    "confidence_after_top10pct_deletion": after,
                                    "confidence_drop": original_prob-after})
                fig, ax = plt.subplots(figsize=(5,4))
                ax.imshow(image); ax.imshow(arr, cmap="jet", alpha=.42)
                ax.set_title(f"{method} | true={classes[label]} pred={classes[pred]}")
                ax.axis("off"); fig.tight_layout()
                fig.savefig(out / f"xai_{best_name}_{method}_{classes[label]}_{pos}.png", dpi=150); plt.close(fig)
    pd.DataFrame(method_rows).to_csv(out / f"{best_name}_xai_deletion_summary.csv", index=False)
    if method_rows:
        (pd.DataFrame(method_rows).groupby("method")["confidence_drop"]
         .agg(["mean","std","count"]).reset_index()
         .to_csv(out / f"{best_name}_xai_method_comparison.csv", index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", default="jmuben_hybrid_results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--xai-samples-per-class", type=int, default=5)
    args = ap.parse_args()
    if args.epochs < 30: raise ValueError("At least 30 epochs per model are required.")
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    seed_all(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    base = datasets.ImageFolder(args.data_root)
    keep, dropped, conflicts = audit_exact_duplicates(base)
    usable = [idx for idx in keep]
    labels = np.asarray([base.targets[i] for i in usable])
    if len(set(labels.tolist())) < 2: raise ValueError("Insufficient class labels after audit")
    tr, hold = train_test_split(np.asarray(usable), test_size=.30, random_state=args.seed,
                                stratify=labels)
    hold_labels = np.asarray([base.targets[i] for i in hold])
    va, te = train_test_split(hold, test_size=.50, random_state=args.seed,
                              stratify=hold_labels)
    class_names = base.classes
    duplicate_report = {
        "archive_images": len(base.samples), "unique_decoded_rgb_images_kept": len(keep),
        "same_label_decoded_pixel_duplicate_files_removed": len(dropped),
        "cross_class_decoded_pixel_conflict_groups_excluded": len(conflicts),
        "training_unique_images": len(tr), "validation_unique_images": len(va),
        "test_unique_images": len(te), "class_names": class_names,
        "class_counts_after_exact_dedup": {c:int(sum(base.targets[i] == j for i in keep))
                                          for j,c in enumerate(class_names)},
    }
    (out / "data_audit_and_split.json").write_text(json.dumps(duplicate_report, indent=2))
    (out / "split_manifest.json").write_text(json.dumps({"seed":args.seed,"train":tr.tolist(),
       "validation":va.tolist(),"test":te.tolist(),"classes":base.class_to_idx},indent=2))
    for group in conflicts:
        print("Excluded cross-label exact-duplicate files:", [(g[1],g[2]) for g in group])
    counts = [base.targets[i] for i in tr]
    weights, train_counts = make_weights(counts, len(class_names))
    pd.DataFrame([{"class":c,"train_count_after_dedup":train_counts[i]}
                  for i,c in enumerate(class_names)]).to_csv(out / "train_class_counts.csv", index=False)
    train_set = ImageSubset(base, tr, args.image_size, train=True)
    val_set = ImageSubset(base, va, args.image_size, train=False)
    test_set = ImageSubset(base, te, args.image_size, train=False)
    workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=workers, pin_memory=False, persistent_workers=workers>0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=False, persistent_workers=workers>0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=False, persistent_workers=workers>0)

    results, val_scores, test_logits = [], {}, {}
    for name in MODEL_NAMES:
        print(f"\n=== {name}: {args.epochs} epochs; {DEVICE}; size={args.image_size} ===", flush=True)
        model, enc_params, head_params = make_model(name, len(class_names), not args.no_pretrained)
        model.to(DEVICE)
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head}], weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        history = {k:[] for k in ["train_loss","val_loss","train_acc","val_acc","train_macro_f1","val_macro_f1"]}
        best_f1, best_state, best_epoch = -1.0, None, None
        started = time.time()
        for ep in range(args.epochs):
            train = run_epoch(model, train_loader, loss_fn, optimizer)
            val = run_epoch(model, val_loader, loss_fn)
            scheduler.step()
            for key, value in [("train_loss",train["loss"]),("val_loss",val["loss"]),
                ("train_acc",train["accuracy"]),("val_acc",val["accuracy"]),
                ("train_macro_f1",train["macro_f1"]),("val_macro_f1",val["macro_f1"])]:
                history[key].append(float(value))
            if val["macro_f1"] > best_f1:
                best_f1, best_epoch = val["macro_f1"], ep+1
                best_state = copy.deepcopy(model.state_dict())
            pd.DataFrame(history).to_csv(out / f"{name}_training_log.csv", index_label="epoch")
            print(f"{name} {ep+1}/{args.epochs}: train_loss={train['loss']:.4f} "
                  f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
                  f"val_macroF1={val['macro_f1']:.4f}", flush=True)
        model.load_state_dict(best_state)
        ckpt = out / f"{name}_best.pt"
        torch.save({"model":name,"state_dict":best_state,"class_names":class_names,
                    "best_epoch":best_epoch,"val_macro_f1":best_f1,"image_size":args.image_size}, ckpt)
        save_curves(name, history, out)
        test = run_epoch(model, test_loader, loss_fn)
        row = evaluate_and_save(name, test, class_names, out)
        row.update({"best_epoch":best_epoch,"best_val_macro_f1":best_f1,
                    "training_seconds":time.time()-started})
        results.append(row); val_scores[name]=best_f1; test_logits[name]=test["logits"]
        print("TEST", row, flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Probability-level soft voting uses only the three single-backbone models; no new fitting.
    single_names = ["EfficientNetB0","MobileNetV3Large","ResNet50"]
    member_probs = [torch.softmax(torch.from_numpy(test_logits[n]), dim=1).numpy()
                    for n in single_names]
    ensemble_probs = np.mean(member_probs, axis=0)
    ensemble_pred = ensemble_probs.argmax(1)
    true_labels = [base.targets[i] for i in te]
    ensemble = {"ys":true_labels,"preds":ensemble_pred.tolist(),"logits":ensemble_probs,
                "loss":float("nan"),"accuracy":accuracy_score(true_labels,ensemble_pred),
                "macro_f1":f1_score(true_labels,ensemble_pred,average="macro",zero_division=0)}
    erow = evaluate_and_save("SoftVotingEnsemble", ensemble, class_names, out)
    erow["component_models"] = ",".join(single_names); results.append(erow)
    pd.DataFrame(results).to_csv(out / "test_model_comparison.csv", index=False)

    best = max(val_scores, key=val_scores.get)
    (out / "validation_selected_model.json").write_text(json.dumps(
        {"selected_model":best,"selection_metric":"validation_macro_f1",
         "validation_macro_f1":val_scores[best]},indent=2))
    best_indices = [i for i in usable if i in set(te)]
    run_xai(best, out / f"{best}_best.pt", base, best_indices, class_names,
            args.image_size, out, args.xai_samples_per_class)
    explained = [best]
    if best != "HybridEfficientMobile":
        # The hybrid is explained as a predeclared architecture of interest;
        # model selection remains based on validation macro-F1, not test scores.
        run_xai("HybridEfficientMobile", out / "HybridEfficientMobile_best.pt",
                base, best_indices, class_names, args.image_size, out,
                args.xai_samples_per_class)
        explained.append("HybridEfficientMobile")
    (out / "xai_models.json").write_text(json.dumps(
        {"validation_selected_model":best,"additional_predeclared_model":"HybridEfficientMobile",
         "models_explained":explained,"note":"The hybrid was included a priori; selection was not based on test performance."},
        indent=2))
    print("Completed. Selected by validation only:",best,flush=True)
    print("All artifacts:",out.resolve(),flush=True)

if __name__ == "__main__":
    main()
