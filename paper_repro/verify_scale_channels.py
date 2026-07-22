#!/usr/bin/env python3
"""Level-A reproduction of Longon (CVPRW 2025) on the Jetson: VERIFY that the paper's
published scale-invariant channels really are scale-invariant, using the paper's own
residual-stream logic (In / Pre / Post) on natural images -- no Lucent, no ImageNet, CPU.

Background (paper's mechanism): in a ResNet basic block,
    Post(pre-ReLU) = Pre (main-path bn2 output)  +  In (identity / block input).
The paper claims certain channels are scale-invariant because In carries a SMALLER-scale
copy of the feature and Pre carries a LARGER-scale copy, so their sum (Post) stays stable
as the image is scaled. It identifies 23 such channels in layer2.1 and 46 in layer3.1
(indices hard-coded in the paper's scale_robust.py).

This script:
  1. Hooks torchvision resnet18: In = layerX[0] output, Pre = layerX[1].bn2 output,
     Post = Pre + In (center unit), for X in {2,3}.
  2. Sweeps natural images through the paper's exact scale transform.
  3. Tests whether the published scale-invariant channels are MORE scale-stable (higher
     cross-scale activation correlation across images) than the other channels.
  4. Plots the In-small / Pre-large / Post-flat mechanism for the top scale-invariant
     channels.

Usage:
  python3 paper_repro/verify_scale_channels.py --images /home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123/boat6 --n 120
Output: results/paper_repro/verify_<layer>.csv + mechanism_<layer>.png
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG = 224
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Paper's published scale-invariant channels (from scale_robust.py)
SCALE_INV = {
    "layer2.1": [108, 21, 25, 127, 56, 47, 18, 52, 35, 67, 48, 72, 110, 106,
                 105, 13, 15, 88, 30, 124, 31, 36, 68],
    "layer3.1": [113, 198, 67, 69, 204, 188, 164, 137, 144, 149, 147, 132, 180, 88,
                 126, 141, 48, 135, 187, 250, 158, 29, 178, 161, 163, 13, 153, 14,
                 80, 206, 220, 179, 215, 16, 192, 9, 59, 200, 63, 1, 100, 58,
                 168, 117, 226, 22],
}
NUM_CH = {"layer2.1": 128, "layer3.1": 256}


def scale_transform(scale):
    """Exactly the paper's scale transform (imnet_val.validate / measure_scale_inv)."""
    base = [transforms.Resize(256)]
    if scale == 1:
        t = base + [transforms.CenterCrop(IMG)]
    elif scale > 1:
        t = base + [transforms.CenterCrop(int(IMG * (2 - scale))), transforms.Resize(IMG)]
    else:
        resize = int(IMG * scale) if int(IMG * scale) % 2 == 0 else math.ceil(IMG * scale)
        t = base + [transforms.CenterCrop(IMG), transforms.Resize(resize),
                    transforms.Pad(int((IMG - resize) / 2), padding_mode="constant", fill=127)]
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


class Taps:
    """Forward hooks to grab In (layerX[0] out) and Pre (layerX[1].bn2 out)."""
    def __init__(self, model):
        self.buf = {}
        model.layer2[0].register_forward_hook(self._save("in_layer2.1"))
        model.layer2[1].bn2.register_forward_hook(self._save("pre_layer2.1"))
        model.layer3[0].register_forward_hook(self._save("in_layer3.1"))
        model.layer3[1].bn2.register_forward_hook(self._save("pre_layer3.1"))

    def _save(self, name):
        def hook(_m, _i, out):
            self.buf[name] = out.detach()
        return hook

    def center_vectors(self, layer):
        """Return (in_c, pre_c, post_c) center-unit channel vectors for a layer."""
        inn = self.buf["in_" + layer][0]     # (C,H,W)
        pre = self.buf["pre_" + layer][0]
        hc, wc = inn.shape[1] // 2, inn.shape[2] // 2
        in_c = inn[:, hc, wc]
        pre_c = pre[:, hc, wc]
        post_c = pre_c + in_c                 # pre-ReLU sum
        return (in_c.cpu().numpy(), pre_c.cpu().numpy(), post_c.cpu().numpy())

    def max_vectors(self, layer):
        """Return (in, pre, post) per-channel MAX-over-space vectors (feature presence).

        Post is the pre-ReLU residual sum; we take spatial max of the summed map so we
        detect the channel's feature wherever it appears, not just at the image center.
        """
        inn = self.buf["in_" + layer][0]     # (C,H,W)
        pre = self.buf["pre_" + layer][0]
        post = pre + inn
        return (inn.amax(dim=(1, 2)).cpu().numpy(),
                pre.amax(dim=(1, 2)).cpu().numpy(),
                post.amax(dim=(1, 2)).cpu().numpy())


def pearson(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="single folder of images (less diverse)")
    ap.add_argument("--data-root", help="UAV123 data_seq/UAV123 dir: sample across MANY "
                    "sequences for diversity (recommended)")
    ap.add_argument("--per-seq", type=int, default=6, help="frames per sequence when --data-root")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    files = []
    if args.data_root:
        seqs = sorted(d for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d)))
        for s in seqs:
            fs = sorted(glob.glob(os.path.join(args.data_root, s, "*.jpg"))) or \
                sorted(glob.glob(os.path.join(args.data_root, s, "*.png")))
            if fs:
                idx = np.linspace(0, len(fs) - 1, min(args.per_seq, len(fs))).astype(int)
                files += [fs[i] for i in idx]
        print("Sampled %d frames across %d sequences (diverse)." % (len(files), len(seqs)))
    elif args.images:
        files = sorted(glob.glob(os.path.join(args.images, "*.jpg"))) or \
            sorted(glob.glob(os.path.join(args.images, "*.png")))
    else:
        ap.error("give --data-root (recommended) or --images")
    if not files:
        print("No images found."); return 1
    if len(files) > args.n:
        files = [files[i] for i in np.linspace(0, len(files) - 1, args.n).astype(int)]
    print("Using %d images." % len(files))

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    taps = Taps(model)

    layers = ["layer2.1", "layer3.1"]
    # post[layer][scale] = (N, C) ; also keep in/pre for mechanism
    post = {L: {s: [] for s in SCALES} for L in layers}
    inn = {L: {s: [] for s in SCALES} for L in layers}
    pre = {L: {s: [] for s in SCALES} for L in layers}

    with torch.no_grad():
        for fi, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            for s in SCALES:
                x = scale_transform(s)(img).unsqueeze(0)
                model(x)
                for L in layers:
                    ic, pc, poc = taps.max_vectors(L)   # feature presence anywhere in frame
                    inn[L][s].append(ic); pre[L][s].append(pc); post[L][s].append(poc)
            if (fi + 1) % 25 == 0:
                print("  %d/%d" % (fi + 1, len(files)))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "paper_repro")
    os.makedirs(out_dir, exist_ok=True)

    for L in layers:
        for s in SCALES:
            post[L][s] = np.array(post[L][s]); inn[L][s] = np.array(inn[L][s])
            pre[L][s] = np.array(pre[L][s])
        C = NUM_CH[L]
        # per-channel cross-scale stability = mean over s!=1 of corr(post@1, post@s) across images
        stab = np.full(C, np.nan)
        for c in range(C):
            x = post[L][1.0][:, c]
            cs = [pearson(x, post[L][s][:, c]) for s in SCALES if s != 1.0]
            cs = [v for v in cs if not np.isnan(v)]
            stab[c] = np.mean(cs) if cs else np.nan

        inv = np.array(SCALE_INV[L])
        others = np.array([c for c in range(C) if c not in set(SCALE_INV[L])])
        inv_stab = stab[inv][~np.isnan(stab[inv])]
        oth_stab = stab[others][~np.isnan(stab[others])]

        print("\n=== %s ===" % L)
        print("  paper scale-invariant channels (n=%d): mean cross-scale stability = %.3f"
              % (len(inv_stab), inv_stab.mean()))
        print("  all other channels        (n=%d): mean cross-scale stability = %.3f"
              % (len(oth_stab), oth_stab.mean()))
        print("  --> paper's channels are %s stable (%+.3f)"
              % ("MORE" if inv_stab.mean() > oth_stab.mean() else "LESS",
                 inv_stab.mean() - oth_stab.mean()))
        # rank: how many of the top-k most-stable channels are paper's?
        order = np.argsort(-np.nan_to_num(stab, nan=-1))
        topk = set(order[:len(inv)].tolist())
        hit = len(topk & set(inv.tolist()))
        print("  overlap of paper's %d channels with the top-%d most stable: %d (%.0f%%)"
              % (len(inv), len(inv), hit, 100.0 * hit / len(inv)))

        # --- paper-faithful metric: measure_scale_inv on each channel's OWN top image ---
        # For channel c, pick the image that most activates it at scale 1, then check how
        # well its Post activation is PRESERVED across scales (clip>=0, /max, mean).
        # High = scale-invariant (activation survives scaling), matching the paper.
        p1 = post[L][1.0]
        pres = np.full(C, np.nan)
        for c in range(C):
            bi = int(np.argmax(p1[:, c]))
            vals = np.clip(np.array([post[L][s][bi, c] for s in SCALES]), 0, None)
            mx = vals.max()
            if mx > 0:
                pres[c] = float((vals / mx).mean())
        inv_pres = pres[inv][~np.isnan(pres[inv])]
        oth_pres = pres[others][~np.isnan(pres[others])]
        order2 = np.argsort(-np.nan_to_num(pres, nan=-1))
        hit2 = len(set(order2[:len(inv)].tolist()) & set(inv.tolist()))
        print("  [paper-faithful] scale preservation on each channel's top image:")
        print("     paper channels = %.3f   others = %.3f   (%s, %+.3f)"
              % (inv_pres.mean(), oth_pres.mean(),
                 "MORE" if inv_pres.mean() > oth_pres.mean() else "LESS",
                 inv_pres.mean() - oth_pres.mean()))
        print("     overlap with top-%d most preserved: %d (%.0f%%)"
              % (len(inv), hit2, 100.0 * hit2 / len(inv)))

        with open(os.path.join(out_dir, "verify_%s.csv" % L), "w") as fcsv:
            fcsv.write("channel,cross_scale_stability,top_image_preservation,is_paper_scale_inv\n")
            for c in range(C):
                fcsv.write("%d,%.4f,%.4f,%d\n"
                           % (c, stab[c], pres[c], int(c in set(SCALE_INV[L]))))

        # mechanism plot: In-small / Pre-large / Post-flat for a few top scale-inv channels
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # pick the 4 paper channels with highest stability here
            inv_sorted = inv[np.argsort(-np.nan_to_num(stab[inv], nan=-1))][:4]
            fig, axes = plt.subplots(1, len(inv_sorted), figsize=(4 * len(inv_sorted), 4),
                                     squeeze=False)
            for ax, c in zip(axes[0], inv_sorted):
                def curve(store):
                    return [np.mean(np.clip(store[L][s][:, c], 0, None)) for s in SCALES]
                for label, store in (("In", inn), ("Pre", pre), ("Post", post)):
                    y = curve(store)
                    y = np.array(y) / (max(y) if max(y) > 0 else 1)
                    ax.plot(SCALES, y, marker="o", label=label)
                ax.axvline(1.0, color="gray", ls="--", lw=1)
                ax.set_title("%s ch %d" % (L, c)); ax.set_xlabel("scale")
                ax.grid(True, alpha=0.3)
            axes[0][0].set_ylabel("norm. center activation")
            axes[0][-1].legend()
            plt.suptitle("Residual-stream mechanism (%s): In small-scale, Pre large-scale, Post flat" % L)
            png = os.path.join(out_dir, "mechanism_%s.png" % L)
            plt.savefig(png, dpi=130, bbox_inches="tight")
            print("  wrote", png)
        except Exception as e:
            print("  plot skipped:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
