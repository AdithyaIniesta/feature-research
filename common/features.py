"""ResNet18 block-wise feature extractor.

Exposes features after each of the 4 residual stages (Block 1..4), matching the
"residual stream" analysis in the CVPRW 2025 paper. CPU-only is fine.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

# ImageNet normalization (ResNet18 pretrained expects this).
_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])


class ResNet18Blocks:
    """Runs a crop through ResNet18 and returns per-block feature maps / vectors."""

    def __init__(self, device="cpu", input_size=64):
        self.device = torch.device(device)
        # weights API differs across torchvision versions; fall back gracefully.
        try:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            net = torchvision.models.resnet18(weights=w)
        except Exception:
            net = torchvision.models.resnet18(pretrained=True)
        net.eval().to(self.device)
        self.net = net
        self.input_size = input_size  # crops resized to input_size x input_size

    def _prep(self, crop_bgr):
        """BGR uint8 HxWx3 -> normalized 1x3xSxS tensor."""
        import cv2
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        t = _NORM(t)
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def block_maps(self, crop_bgr):
        """Return dict block1..block4 -> feature map tensor (1,C,H,W)."""
        x = self._prep(crop_bgr)
        n = self.net
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        out = {}
        x = n.layer1(x); out["block1"] = x
        x = n.layer2(x); out["block2"] = x
        x = n.layer3(x); out["block3"] = x
        x = n.layer4(x); out["block4"] = x
        return out

    @torch.no_grad()
    def block_vectors(self, crop_bgr):
        """Global-average-pooled feature vector per block -> dict block -> np.array(C,)."""
        maps = self.block_maps(crop_bgr)
        vecs = {}
        for k, m in maps.items():
            v = F.adaptive_avg_pool2d(m, 1).flatten().cpu().numpy()
            vecs[k] = v
        return vecs


def cosine(a, b):
    """Cosine similarity between two 1-D numpy vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
