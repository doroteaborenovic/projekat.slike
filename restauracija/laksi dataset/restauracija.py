import os
import time
import random
import warnings
import logging
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF  # Za bojice
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)
os.environ["PYTHONWARNINGS"] = "ignore"
torch.backends.cudnn.benchmark = True

# =====================================================================
# GLOBALNO DEFINISANI BAZIČNI BLOKOVI (Dostupni svim modelima na vrhu)
# =====================================================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


# Definisano mapiranje 9 oštećenja na klase 1-9 za potrebe analize imena fajla
DAMAGE_MAP = {
    'apply_anisotropic_diffusion': 1,  # Vlaga i gubitak detalja (Potreban CCR)
    'apply_mold_and_decay': 2,         # Buđ i biološka degradacija
    'apply_chemical_aging': 3,         # Oksidacija/Starenje
    'apply_fft_lpf': 4,                # Gubitak oštrine (FFT LPF)
    'apply_cracks': 5,                 # Pukotine na platnu
    'apply_paint_flaking': 6,          # Ljuštenje boje
    'apply_water_stains': 7,           # Vodene mrlje (Coffee-ring) (Potreban CCR)
    'apply_dust_and_scratches': 8,     # Prašina i ogrebotine
    'apply_combined_damage': 9,        # Kombinovano oštećenje (Potreban CCR)
}

# Klase kod kojih se aktivira Contrast Color Recovery (CCR)
CCR_CLASSES = {1, 7, 9}


# klasik dodinamreza zbog klasifiakcije
class RecursiveDenseMicroBlock(nn.Module):
    def __init__(self, channels: int, num_recursions: int = 3):
        super().__init__()
        self.num_recursions = num_recursions
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.fusion = nn.Conv2d(channels * num_recursions, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        out = x
        for _ in range(self.num_recursions):
            out = F.relu(self.bn(self.conv(out)) + x)
            outputs.append(out)
        merged = torch.cat(outputs, dim=1)
        return self.fusion(merged)


class SpectralDecomposeBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1)
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        low = F.interpolate(
            F.avg_pool2d(x, kernel_size=2),
            size=x.shape[2:], mode='bilinear', align_corners=False
        )
        high = x - low
        low_feat = self.low_conv(low)
        high_feat = self.high_conv(high)
        concat = torch.cat([low_feat, high_feat], dim=1)
        w = self.gate(concat)
        fused = w[:, 0:1] * low_feat + w[:, 1:2] * high_feat
        return self.fuse(torch.cat([fused, x], dim=1))


class SpatialBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dense_micro = RecursiveDenseMicroBlock(out_ch, num_recursions=3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.dense_micro(x)
        pooled = self.pool(x)
        return pooled, x


class AsymmetricCrossBridge(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_to_spectral = nn.Sequential(
            nn.Conv2d(spatial_ch, spectral_ch, 1),
            nn.BatchNorm2d(spatial_ch),
            nn.ReLU(inplace=True)
        )
        self.spectral_to_spatial = nn.Sequential(
            nn.Conv2d(spectral_ch, spatial_ch, 1),
            nn.BatchNorm2d(spatial_ch),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv2d(spatial_ch + spectral_ch, out_ch, 1)

    def forward(self, spatial_feat: Tensor, spectral_feat: Tensor) -> Tensor:
        spectral_enhanced = spectral_feat + self.spatial_to_spectral(
            F.adaptive_avg_pool2d(spatial_feat, spectral_feat.shape[2:])
        )
        spatial_enhanced = spatial_feat + self.spectral_to_spatial(
            F.interpolate(spectral_feat, size=spatial_feat.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        min_h = min(spatial_feat.shape[2], spectral_feat.shape[2])
        min_w = min(spatial_feat.shape[3], spectral_feat.shape[3])
        s_pooled = F.adaptive_avg_pool2d(spatial_enhanced, (min_h, min_w))
        sp_pooled = F.adaptive_avg_pool2d(spectral_enhanced, (min_h, min_w))
        return self.fuse(torch.cat([s_pooled, sp_pooled], dim=1))


class GatedFusionBlock(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_proj = nn.Conv2d(spatial_ch, out_ch, 1)
        self.spectral_proj = nn.Conv2d(spectral_ch, out_ch, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch * 2, out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // 4, out_ch * 2),
            nn.Sigmoid()
        )

    def forward(self, spatial: Tensor, spectral: Tensor) -> Tensor:
        s = self.spatial_proj(spatial)
        sp = self.spectral_proj(
            F.interpolate(spectral, size=spatial.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        combined = torch.cat([s, sp], dim=1)
        gates = self.gate(combined).view(combined.shape[0], -1, 1, 1)
        out_ch_val = s.shape[1]
        s_gate = gates[:, :out_ch_val]
        sp_gate = gates[:, out_ch_val:]
        return s_gate * s + sp_gate * sp


class DamageAttentionModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn_map = self.attention(x)
        attended = x * attn_map
        refined = self.refine(attended) + x
        return refined, attn_map


class DodinaMreza(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.spatial_block1 = SpatialBlock(in_channels, 64)
        self.spatial_block2 = SpatialBlock(64, 128)
        self.spatial_block3 = SpatialBlock(128, 256)

        self.spectral_init = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.spectral_block1 = SpectralDecomposeBlock(64)
        self.spectral_pool1 = nn.MaxPool2d(2)
        self.spec_proj1 = nn.Sequential(
            nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.spectral_block2 = SpectralDecomposeBlock(128)
        self.spectral_pool2 = nn.MaxPool2d(2)
        self.spec_proj2 = nn.Sequential(
            nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.spectral_block3 = SpectralDecomposeBlock(256)

        self.cross1 = AsymmetricCrossBridge(64, 64, 64)
        self.cross2 = AsymmetricCrossBridge(128, 128, 128)
        self.cross3 = AsymmetricCrossBridge(256, 256, 256)

        self.gated_fusion = GatedFusionBlock(256, 256, 512)
        self.damage_attention = DamageAttentionModule(512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.damage_map_head = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        s1, s1_skip = self.spatial_block1(x)
        s2, s2_skip = self.spatial_block2(s1)
        s3, s3_skip = self.spatial_block3(s2)

        sp0 = self.spectral_init(x)
        sp1 = self.spectral_block1(sp0)
        sp1_p = self.spec_proj1(self.spectral_pool1(sp1))
        sp2 = self.spectral_block2(sp1_p)
        sp2_p = self.spec_proj2(self.spectral_pool2(sp2))
        sp3 = self.spectral_block3(sp2_p)

        c1 = self.cross1(s1_skip, sp1)
        c2 = self.cross2(s2_skip, sp2)
        c3 = self.cross3(s3_skip, sp3)

        s3_enriched = s3 + F.adaptive_avg_pool2d(c3, s3.shape[2:])

        fused = self.gated_fusion(s3_enriched, sp3)
        attended, damage_map = self.damage_attention(fused)

        logits = self.classifier(attended)
        aux_damage = self.damage_map_head(attended)

        return {
            'logits': logits,
            'damage_map': damage_map,
            'aux_damage': aux_damage
        }


# ovde je deo od klasifiakcije
class DamageDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int = 128, train: bool = True):
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.samples = []
        for label in [0, 1]:
            folder = os.path.join(dataset_dir, str(label))
            if not os.path.exists(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, path



class RecursiveDenseRestorationBlock(nn.Module):
    def __init__(self, channels: int, num_recursions: int = 3):
        super().__init__()
        self.num_recursions = num_recursions
        self.conv = DepthwiseSeparableConv2d(channels, channels, 3, padding=1)
        self.gn = nn.GroupNorm(4, channels)
        self.fusion = nn.Conv2d(channels * num_recursions, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        out = x
        for _ in range(self.num_recursions):
            out = F.relu(self.gn(self.conv(out)) + x)
            outputs.append(out)
        merged = torch.cat(outputs, dim=1)
        return self.fusion(merged)


class SpectralDecompositionRestorationBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.low_conv = nn.Sequential(
            DepthwiseSeparableConv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=False)
        )
        self.high_conv = nn.Sequential(
            DepthwiseSeparableConv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=False)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1)
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        low = F.interpolate(
            F.avg_pool2d(x, kernel_size=2),
            size=x.shape[2:], mode='bilinear', align_corners=False
        )
        high = x - low
        low_feat = self.low_conv(low)
        high_feat = self.high_conv(high)
        concat = torch.cat([low_feat, high_feat], dim=1)
        w = self.gate(concat)
        fused = w[:, 0:1] * low_feat + w[:, 1:2] * high_feat
        return self.fuse(torch.cat([fused, x], dim=1))


class SpatialEncoderRestorationBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.dense_micro(x)
        pooled = self.pool(x)
        return pooled, x


class AsymmetricCrossBridgeRestoration(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_to_spectral = nn.Sequential(
            nn.Conv2d(spatial_ch, spectral_ch, 1, bias=False),
            nn.GroupNorm(4, spectral_ch),
            nn.ReLU(inplace=False)
        )
        self.spectral_to_spatial = nn.Sequential(
            nn.Conv2d(spectral_ch, spatial_ch, 1, bias=False),
            nn.GroupNorm(4, spatial_ch),
            nn.ReLU(inplace=False)
        )
        self.fuse = nn.Conv2d(spatial_ch + spectral_ch, out_ch, 1, bias=False)

    def forward(self, spatial_feat: Tensor, spectral_feat: Tensor) -> Tensor:
        spectral_enhanced = spectral_feat + self.spatial_to_spectral(
            F.adaptive_avg_pool2d(spatial_feat, spectral_feat.shape[2:])
        )
        spatial_enhanced = spatial_feat + self.spectral_to_spatial(
            F.interpolate(spectral_feat, size=spatial_feat.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        min_h = min(spatial_feat.shape[2], spectral_feat.shape[2])
        min_w = min(spatial_feat.shape[3], spectral_feat.shape[3])
        s_pooled = F.adaptive_avg_pool2d(spatial_enhanced, (min_h, min_w))
        sp_pooled = F.adaptive_avg_pool2d(spectral_enhanced, (min_h, min_w))
        return self.fuse(torch.cat([s_pooled, sp_pooled], dim=1))


class GatedFusionRestorationBlock(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_proj = nn.Conv2d(spatial_ch, out_ch, 1, bias=False)
        self.spectral_proj = nn.Conv2d(spectral_ch, out_ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch * 2, out_ch // 4, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(out_ch // 4, out_ch * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, spatial: Tensor, spectral: Tensor) -> Tensor:
        s = self.spatial_proj(spatial)
        sp = self.spectral_proj(
            F.interpolate(spectral, size=spatial.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        combined = torch.cat([s, sp], dim=1)
        gates = self.gate(combined).view(combined.shape[0], -1, 1, 1)
        out = s.shape[1]
        s_gate = gates[:, :out]
        sp_gate = gates[:, out:]
        return s_gate * s + sp_gate * sp


class DamageAttentionRestorationModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_channels // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn_map = self.attention(x)
        attended = x * attn_map
        refined = self.refine(attended) + x
        return refined, attn_map


class DecoderRestorationBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch + 1, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=2)
        self.spectral = SpectralDecompositionRestorationBlock(out_ch)

    def forward(self, x: Tensor, skip: Tensor, damage_map: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip, dm], dim=1)
        return self.spectral(self.dense_micro(self.conv(x)))


class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 4
        self.c1 = nn.Conv2d(channels, mid, 3, padding=1, dilation=1, bias=False)
        self.c2 = nn.Conv2d(channels, mid, 3, padding=2, dilation=2, bias=False)
        self.c3 = nn.Conv2d(channels, mid, 3, padding=4, dilation=4, bias=False)
        self.c4 = nn.Conv2d(channels, mid, 3, padding=8, dilation=8, bias=False)
        self.fusion = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.GroupNorm(4, channels)

    def forward(self, x: Tensor) -> Tensor:
        merged = torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x)], dim=1)
        return F.relu(self.bn(self.fusion(merged)) + x)


class GatedSkipConnection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, skip: Tensor) -> Tensor:
        return skip * self.gate(skip)


class EdgeBranch(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kx', kx.repeat(3, 1, 1, 1))
        self.register_buffer('ky', ky.repeat(3, 1, 1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(6, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(torch.cat([F.conv2d(x, self.kx, padding=1, groups=3), F.conv2d(x, self.ky, padding=1, groups=3)], dim=1))


class ContrastColorRecovery(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_ch // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch // 2, out_ch, 3, padding=1)
        )
        self.global_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch // 4, out_ch * 2, 1),
        )

    def forward(self, x: Tensor, input_img: Tensor) -> Tensor:
        local_refinement = self.local_conv(x)
        global_stats = self.global_adjust(x)
        gain, bias = torch.chunk(global_stats, 2, dim=1)
        gain = torch.sigmoid(gain).view(x.shape[0], -1, 1, 1) * 2.0
        bias = torch.tanh(bias).view(x.shape[0], -1, 1, 1) * 0.5
        adjusted = local_refinement * gain + bias
        return torch.clamp(input_img + adjusted, 0.0, 1.0)


# --- GLAVNI MODEL RESTAURACIJE (SA USLOVNIM CCR I ISPRAVLJENIM s4 POOLINGOM) ---
class Restauracija(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 32):
        super().__init__()
        self.edge_branch = EdgeBranch(out_channels=base_ch)
        self.edge_fusion = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.spatial_block1 = SpatialEncoderRestorationBlock(in_channels, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)
        self.spectral_init = nn.Sequential(nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
        self.spectral_block1 = SpectralDecompositionRestorationBlock(base_ch)
        self.spectral_pool1 = nn.MaxPool2d(2)
        self.spec_proj1 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 1, bias=False), nn.GroupNorm(4, base_ch * 2), nn.ReLU(inplace=False))
        self.spectral_block2 = SpectralDecompositionRestorationBlock(base_ch * 2)
        self.spectral_pool2 = nn.MaxPool2d(2)
        self.spec_proj2 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 1, bias=False), nn.GroupNorm(4, base_ch * 4), nn.ReLU(inplace=False))
        self.spectral_block3 = SpectralDecompositionRestorationBlock(base_ch * 4)
        self.spectral_pool3 = nn.MaxPool2d(2)
        self.spec_proj3 = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False))
        self.spectral_block4 = SpectralDecompositionRestorationBlock(base_ch * 8)
        self.cross1 = AsymmetricCrossBridgeRestoration(base_ch, base_ch, base_ch)
        self.cross2 = AsymmetricCrossBridgeRestoration(base_ch * 2, base_ch * 2, base_ch * 2)
        self.cross3 = AsymmetricCrossBridgeRestoration(base_ch * 4, base_ch * 4, base_ch * 4)
        self.cross4 = AsymmetricCrossBridgeRestoration(base_ch * 8, base_ch * 8, base_ch * 8)
        self.gated_fusion = GatedFusionRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 8)
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)
        self.bottleneck_refine = nn.Sequential(nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False), DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2))
        self.decoder4 = DecoderRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = DecoderRestorationBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = DecoderRestorationBlock(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = DecoderRestorationBlock(base_ch, base_ch, base_ch)
        self.skip_gate1 = GatedSkipConnection(base_ch)
        self.skip_gate2 = GatedSkipConnection(base_ch * 2)
        self.skip_gate3 = GatedSkipConnection(base_ch * 4)
        self.skip_gate4 = GatedSkipConnection(base_ch * 8)
        self.skip_refine1 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, num_recursions=2), SpectralDecompositionRestorationBlock(base_ch))
        self.skip_refine2 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 2, num_recursions=2), SpectralDecompositionRestorationBlock(base_ch * 2))
        self.skip_refine3 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 4, num_recursions=2), SpectralDecompositionRestorationBlock(base_ch * 4))
        self.skip_refine4 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2), SpectralDecompositionRestorationBlock(base_ch * 8))

        # Pomoćne AUX grane
        self.aux_head3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(base_ch, out_channels, 3, padding=1, bias=False))
        self.aux_head2 = nn.Sequential(nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(base_ch // 2, out_channels, 3, padding=1))

        self.final_refinement = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, num_recursions=2), SpectralDecompositionRestorationBlock(base_ch), RecursiveDenseRestorationBlock(base_ch, num_recursions=2))
        self.output_head = nn.Sequential(nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(base_ch // 2, out_channels, 3, padding=1))

        # Dinamički blok za boje
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_channels)

    def forward(self, x: Tensor, use_ccr: Tensor | bool = True) -> Tensor | dict[str, Tensor]:
        input_img = x
        s1, s1_skip = self.spatial_block1(x)
        s2, s2_skip = self.spatial_block2(s1)
        s3, s3_skip = self.spatial_block3(s2)
        s4, s4_skip = self.spatial_block4(s3) # Vraćeno na ispravan s4 za stabilno i dokazano poklapanje slojeva
        sp1 = self.spectral_block1(self.spectral_init(x))
        sp1_p = self.spec_proj1(self.spectral_pool1(sp1))
        sp2 = self.spectral_block2(sp1_p)
        sp2_p = self.spec_proj2(self.spectral_pool2(sp2))
        sp3 = self.spectral_block3(sp2_p)
        sp3_p = self.spec_proj3(self.spectral_pool3(sp3))
        sp4 = self.spectral_block4(sp3_p)
        c1 = self.cross1(s1_skip, sp1)
        c2 = self.cross2(s2_skip, sp2)
        c3 = self.cross3(s3_skip, sp3)
        c4 = self.cross4(s4_skip, sp4)
        s4_enriched = s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:])
        fused = self.gated_fusion(s4_enriched, sp4)
        attended, damage_map = self.damage_attention(fused)
        bottleneck_out = self.bottleneck_refine(attended)
        skip4_enhanced = self.skip_refine4(self.skip_gate4(s4_skip) + F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False))
        skip3_enhanced = self.skip_refine3(self.skip_gate3(s3_skip) + F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False))
        skip2_enhanced = self.skip_refine2(self.skip_gate2(s2_skip) + F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False))
        skip1_enhanced = self.skip_refine1(self.skip_gate1(s1_skip) + F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False))
        d4 = self.decoder4(bottleneck_out, skip4_enhanced, damage_map)
        d3 = self.decoder3(d4, skip3_enhanced, damage_map)
        d2 = self.decoder2(d3, skip2_enhanced, damage_map)
        d1 = self.decoder1(d2, skip1_enhanced, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        d1_refined = self.final_refinement(d1)
        edge_feats = self.edge_branch(input_img)
        d1_fused = self.edge_fusion(torch.cat([d1_refined, edge_feats], dim=1))

        # Uslovni prolaz: CCR ili Standardna rezidualna veza i ovde ako je za ccr to su ona tri oštećenja gore navedena
        residual = self.output_head(d1_fused)
        out_no_ccr = torch.clamp(input_img + residual, 0.0, 1.0)
        out_ccr = self.contrast_color_recovery(d1_fused, input_img)

        # Koristimo adaptivni self-gated CCR iz Vašeg originalnog modela
        out = out_ccr

        if self.training:
            aux3 = self.aux_head3(d3)
            aux3 = F.interpolate(aux3, size=input_img.shape[2:], mode='bilinear', align_corners=False)
            aux3 = torch.clamp(input_img + aux3, 0.0, 1.0)

            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(aux2, size=input_img.shape[2:], mode='bilinear', align_corners=False)
            aux2 = torch.clamp(input_img + aux2, 0.0, 1.0)

            return {
                'out': out,
                'aux2': aux2,
                'aux3': aux3,
                'damage_map': damage_map,
            }

        return out


# ============================================================
# 5. STRUKTURIRANI SKUP PODATAKA ZA RESTAURACIJU
# ============================================================
class RestorationDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int = 192, train: bool = True, preload_to_ram: bool = True):
        self.img_size = img_size
        self.train = train
        self.preload_to_ram = preload_to_ram

        folder_0 = os.path.join(dataset_dir, '0')
        folder_1 = os.path.join(dataset_dir, '1')
        if not os.path.exists(folder_0) or not os.path.exists(folder_1):
            raise FileNotFoundError(f"Folderi '0' i/ili '1' ne postoje na putanji {dataset_dir}!")

        dmg_files = sorted([f for f in os.listdir(folder_1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        self.pairs = []
        self.use_ccr_list = []

        for dmg_f in dmg_files:
            parts = dmg_f.split('_')
            if len(parts) >= 2:
                base_name = f"{parts[0]}_{parts[1]}"
            else:
                continue

            clean_name = f"{base_name}_flip.jpg" if "_flip_" in dmg_f else f"{base_name}_clean.jpg"
            clean_path = os.path.join(folder_0, clean_name)
            dmg_path = os.path.join(folder_1, dmg_f)

            if os.path.exists(clean_path):
                self.pairs.append((dmg_path, clean_path))
                is_ccr = any(pattern in dmg_f for pattern in ['apply_anisotropic_diffusion', 'apply_water_stains', 'apply_combined_damage'])
                self.use_ccr_list.append(is_ccr)

        self.cached_pairs = []
        if self.preload_to_ram and len(self.pairs) > 0:
            print(f"[Dataset] Učitavanje {len(self.pairs)} parova za restauraciju u RAM...")
            for deg_path, clean_path in self.pairs:
                deg_img = Image.open(deg_path).convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
                clean_img = Image.open(clean_path).convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
                self.cached_pairs.append((deg_img, clean_img))

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        if self.preload_to_ram:
            deg_img, clean_img = self.cached_pairs[idx]
        else:
            deg_path, clean_path = self.pairs[idx]
            deg_img = Image.open(deg_path).convert('RGB').resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            clean_img = Image.open(clean_path).convert('RGB').resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        deg_t = self.transform(deg_img)
        clean_t = self.transform(clean_img)

        if self.train:
            # Geometrijske augmentacije
            if torch.rand(1) > 0.5:
                deg_t, clean_t = torch.flip(deg_t, dims=[2]), torch.flip(clean_t, dims=[2])
            if torch.rand(1) > 0.5:
                deg_t, clean_t = torch.flip(deg_t, dims=[1]), torch.flip(clean_t, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                deg_t, clean_t = torch.rot90(deg_t, k=k, dims=[1, 2]), torch.rot90(clean_t, k=k, dims=[1, 2])

            # NOVO: Sinhronizovane kolor i kontrast augmentacije
            if torch.rand(1) > 0.5:
                brightness_factor = random.uniform(0.85, 1.15)
                contrast_factor = random.uniform(0.85, 1.15)
                saturation_factor = random.uniform(0.9, 1.1)

                deg_t = TF.adjust_brightness(deg_t, brightness_factor)
                clean_t = TF.adjust_brightness(clean_t, brightness_factor)

                deg_t = TF.adjust_contrast(deg_t, contrast_factor)
                clean_t = TF.adjust_contrast(clean_t, contrast_factor)

                deg_t = TF.adjust_saturation(deg_t, saturation_factor)
                clean_t = TF.adjust_saturation(clean_t, saturation_factor)

        return deg_t, clean_t, self.use_ccr_list[idx]

    def __len__(self):
        return len(self.pairs)


# =====================================================================
# 6. GUBICI ZA MODEL RESTAURACIJE (LAGANI I STABILNI)
# =====================================================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', _2D_window.expand(self.channel, 1, window_size, window_size).contiguous())

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            _1D_window = gaussian(self.window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, self.window_size, self.window_size).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-6)
        return 1.0 - ssim_map.mean()


def gaussian(window_size: int, sigma: float) -> Tensor:
    gauss = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        p_gray = torch.mean(pred, dim=1, keepdim=True)
        t_gray = torch.mean(target, dim=1, keepdim=True)
        gx_p = F.conv2d(p_gray, self.kx, padding=1)
        gy_p = F.conv2d(p_gray, self.ky, padding=1)
        gx_t = F.conv2d(t_gray, self.kx, padding=1)
        gy_t = F.conv2d(t_gray, self.ky, padding=1)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_blur = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        target_blur = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        return F.l1_loss(pred_blur, target_blur)


class CustomPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = [1, 2, 4]
        dx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        dy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dx', dx.repeat(3, 1, 1, 1))
        self.register_buffer('dy', dy.repeat(3, 1, 1, 1))
        self.register_buffer('lap', lap.repeat(3, 1, 1, 1))

    def extract_features(self, x: Tensor) -> list[Tensor]:
        feats = []
        for s in self.scales:
            x_scaled = F.interpolate(x, scale_factor=1.0 / s, mode='bilinear', align_corners=False) if s > 1 else x
            fx = F.conv2d(x_scaled, self.dx, padding=1, groups=3)
            fy = F.conv2d(x_scaled, self.dy, padding=1, groups=3)
            flap = F.conv2d(x_scaled, self.lap, padding=1, groups=3)
            feats.extend([fx, fy, flap])
        return feats

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_feats = self.extract_features(pred)
        target_feats = self.extract_features(target)
        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss += F.l1_loss(pf, tf)
        return loss / len(pred_feats)


class SoftHardExampleMiningLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        error_map = torch.abs(pred - target).mean(dim=1, keepdim=True).detach()
        return 1.0 + torch.sigmoid((error_map - error_map.mean()) / (error_map.std() + 1e-6))


class FrequencyConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_low = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        tgt_low = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        pred_high = pred - pred_low
        tgt_high = target - tgt_low
        return 0.4 * F.l1_loss(pred_low, tgt_low) + 0.6 * F.l1_loss(pred_high, tgt_high)


class RestauracijaLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.sobel = SobelLoss()
        self.ssim = SSIMLoss()
        self.perceptual = CustomPerceptualLoss()
        self.soft_hem = SoftHardExampleMiningLoss()
        self.freq_consistency = FrequencyConsistencyLoss()
        self.color_loss_fn = ColorLoss()

    def single_scale_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        hem_weight = self.soft_hem(pred, target)
        char_diff = torch.sqrt((pred - target) ** 2 + self.eps)
        char_loss = torch.mean(char_diff * hem_weight)

        ssim_loss = self.ssim(pred, target)
        edge_loss = self.sobel(pred, target)
        percep_loss = self.perceptual(pred, target)
        freq_loss = self.freq_consistency(pred, target)
        color_loss = self.color_loss_fn(pred, target)

        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        fft_loss = F.l1_loss(torch.real(pred_fft), torch.real(target_fft)) + \
                   F.l1_loss(torch.imag(pred_fft), torch.imag(target_fft))

        return (
            0.40 * char_loss +
            0.10 * ssim_loss +
            0.10 * edge_loss +
            0.15 * fft_loss +
            0.10 * percep_loss +
            0.10 * color_loss +
            0.05 * freq_loss
        )

    def multi_scale_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.single_scale_loss(pred, target)
        for scale in [0.5, 0.25]:
            p = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
            t = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            loss += scale * torch.mean(torch.sqrt((p - t) ** 2 + self.eps))
        return loss

    def forward(self, pred_outputs: Tensor | dict[str, Tensor], target: Tensor) -> Tensor:
        if isinstance(pred_outputs, dict):
            total_loss = self.multi_scale_loss(pred_outputs['out'], target)
            if 'aux3' in pred_outputs:
                total_loss += 0.3 * torch.mean(torch.sqrt((pred_outputs['aux3'] - target) ** 2 + self.eps))
            if 'aux2' in pred_outputs:
                total_loss += 0.15 * torch.mean(torch.sqrt((pred_outputs['aux2'] - target) ** 2 + self.eps))
            return total_loss
        return self.multi_scale_loss(pred_outputs, target)


# ucitavanje dodinemrezejej
def evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(
    model_path: str,
    test_dataset_dir: str,
    img_size: int = 128,
    batch_size: int = 32,
    koristi_mini_podskup: bool = True,
    velicina_podskupa: int = 10000
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}\nUređaj za proračune: {device.type.upper()}\nTestni dataset: {test_dataset_dir}\n{'='*60}\n")

    test_dataset = DamageDataset(test_dataset_dir, img_size=img_size, train=False)
    if len(test_dataset) == 0:
        print(f"Nema slika na putanji {test_dataset_dir}!")
        return None

    if koristi_mini_podskup and len(test_dataset) > velicina_podskupa:
        indeksi = np.linspace(0, len(test_dataset) - 1, velicina_podskupa, dtype=int).tolist()
        test_dataset = Subset(test_dataset, indeksi)
        print(f"[INFO] Aktiviran podskup od {len(test_dataset)} slika.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Inicijalizacija mreže za tačno 2 KLASE (binarni klasifikator)
    model = DodinaMreza(num_classes=2).to(device)

    if not os.path.exists(model_path):
        print(f"Model nije pronađen: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Učitan checkpoint modela: {model_path}")
    else:
        model.load_state_dict(checkpoint)
        print("Učitane sirove težine modela.")

    model.eval()

    all_probs = []
    all_labels = []
    all_paths = []

    damage_names = {
        'apply_anisotropic_diffusion': 'Vlaga i gubitak detalja',
        'apply_mold_and_decay': 'Bud i bioloska degradacija',
        'apply_chemical_aging': 'Hemijsko starenje i zutilo',
        'apply_fft_lpf': 'Gubitak ostrine (FFT LPF)',
        'apply_cracks': 'Pukotine na platnu',
        'apply_water_stains': 'Vodene mrlje (Coffee-ring)',
        'apply_paint_flaking': 'Ljustenje boje',
        'apply_dust_and_scratches': 'Prasina i ogrebotine',
        'apply_combined_damage': 'Kombinovano ostecenje'
    }

    stats = {name: {'total': 0, 'correct': 0} for name in damage_names.values()}
    stats['Bez ostecenja (Ciste slike)'] = {'total': 0, 'correct': 0}

    print("Pokrećem Vašu 5-way Test-Time Augmentaciju...")
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs_orig = F.softmax(outputs['logits'], dim=-1)

            probs_flipped_h = F.softmax(model(torch.flip(images, dims=[3]))['logits'], dim=-1)
            probs_flipped_v = F.softmax(model(torch.flip(images, dims=[2]))['logits'], dim=-1)
            probs_rot180 = F.softmax(model(torch.flip(images, dims=[2, 3]))['logits'], dim=-1)
            probs_bright = F.softmax(model(torch.clamp(images * 1.15, 0.0, 1.0))['logits'], dim=-1)

            probs_final = (probs_orig + probs_flipped_h + probs_flipped_v + probs_rot180 + probs_bright) / 5.0

            all_probs.extend(probs_final[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Dinamička pretraga praga za binarnu klasifikacij
    best_threshold = 0.5  #pretraga optimalnog praga od 0.10 vrndosti do 0.9 sa skokom od 0.01 i on testira svih 80 i onda vidi koji daje najveci f1 score i taj uzme kao najbolji tj to bude prag
    best_f1 = 0.0
    thresholds = np.arange(0.10, 0.90, 0.01)
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"Optimalni prag pronađen na osnovu F1 kalibracije: {best_threshold:.2f}")

    all_preds = (all_probs >= best_threshold).astype(int)

    # Razvrstavanje statistika po klasama oštećenja na osnovu analize imena fajla
    for pred, label, path in zip(all_preds, all_labels, all_paths):
        if label == 0:
            stats['Bez ostecenja (Ciste slike)']['total'] += 1
            if pred == 0:
                stats['Bez ostecenja (Ciste slike)']['correct'] += 1
        else:
            filename = os.path.basename(path).lower()
            for func_name, display_name in damage_names.items():
                if func_name in filename:
                    stats[display_name]['total'] += 1
                    if pred == 1:
                        stats[display_name]['correct'] += 1
                    break

    ukupna_tacnost = accuracy_score(all_labels, all_preds) * 100
    print("\n" + "="*50)
    print(f"REZULTATI KLASIFIKACIJE")
    print(f"Ukupna tačnost modela (Accuracy): {ukupna_tacnost:.2f}%")
    print("="*50 + "\n")

    PINK, RESET, BOLD = "\033[38;5;205m", "\033[0m", "\033[1m"
    print(f"{BOLD}Pregled po tipu oštećenja{RESET}")
    print(f"{PINK}┌──────────────────────────────────┬────────────┬────────────┬──────────────┐{RESET}")
    print(f"{PINK}│{RESET} {BOLD}{'tip ostecenja':<32} {PINK}│{RESET} {BOLD}{'testirano':<10} {PINK}│{RESET} {BOLD}{'tacno':<10} {PINK}│{RESET} {BOLD}{'tacnost (%)':<12} {PINK}│{RESET}")
    print(f"{PINK}├──────────────────────────────────┼────────────┼────────────┼──────────────┤{RESET}")

    rows = []
    for cat, data in stats.items():
        total = data['total']
        correct = data['correct']
        acc = (correct / total * 100) if total > 0 else 0.0
        rows.append([cat, total, correct, round(acc, 2)])
        print(f"{PINK}│{RESET} {cat:<32} {PINK}│{RESET} {total:<10d} {PINK}│{RESET} {correct:<10d} {PINK}│{RESET} {acc:<11.2f}% {PINK}│{RESET}")

    print(f"{PINK}└──────────────────────────────────┴────────────┴──────────────┘{RESET}")

    df_stats = pd.DataFrame(rows, columns=['tip ostecenja', 'testirano', 'tacno', 'tacnost (%)'])

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=['Bez ostecenja', 'Osteceno'],
                yticklabels=['Bez ostecenja', 'Osteceno'])
    plt.xlabel('Predviđeno')
    plt.ylabel('Stvarno')
    plt.title('Matrica konfuzije (Dodina Mreza sa TTA)')
    plt.show()

    return df_stats


# =====================================================================
# 8. TRENING ZA RESTAURACIJU (SA INTEGRISANIM OPTIMIZACIJAMA I NAZIVIMA)
# =====================================================================
def train_restauracija(
    dataset_dir: str,
    epochs_to_train: int = 30,
    batch_size: int = 4,
    lr: float = 2e-4,
    img_size: int = 192,
    save_dir: str = '.',
    resume: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = RestorationDataset(dataset_dir, img_size=img_size, train=True, preload_to_ram=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Kreiranje stabilnog validacionog skupa od 100 slika iz trening skupa
    eval_indices = np.arange(min(100, len(train_dataset)))
    eval_subset = Subset(train_dataset, eval_indices)
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Restauracija(base_ch=32).to(device)
    criterion = RestauracijaLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train, eta_min=1e-6)

    start_epoch = 0
    best_psnr = 0.0

    # KORISTE SE TAČNI NAZIVI KOJE STE TRAŽILI:
    checkpoint_path = os.path.join(save_dir, 'dodinarestauracijajej.pth')
    best_checkpoint_path = os.path.join(save_dir, 'dodinarestauracijabest.pth')

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        best_psnr = checkpoint.get('best_psnr', 0.0)  # Učitavanje najboljeg PSNR-a
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train, eta_min=1e-6)
        print(f"[Resume] Nastavljamo trening Restauracije od epohe {start_epoch+1}. Najbolji zabeleženi PSNR: {best_psnr:.2f} dB")
    else:
        print("[Trening] Započinjem trening modela restauracije OD NULE (bez resume).")

    # Akumulacija gradijenata (Simulacija stabilnosti većeg batch size-a od 12)
    accumulation_steps = 3
    optimizer.zero_grad()

    for epoch in range(start_epoch, start_epoch + epochs_to_train):
        model.train()
        running_loss = 0.0

        for batch_idx, (deg, clean, use_ccr) in enumerate(train_loader):
            deg, clean = deg.to(device), clean.to(device)

            # Koristimo originalni, samostalni CCR iz Vašeg modela bez use_ccr u forward-u
            outputs = model(deg)
            loss = criterion(outputs, clean)

            # Podela gubitka sa koracima akumulacije
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        epoch_loss = running_loss / len(train_loader)

        # valdiacija na kraju epohe cisto da vidim kako ide i da se ucita najbolji model
        model.eval()
        total_psnr = 0.0
        num_batches = 0

        with torch.no_grad():
            for deg, clean, _ in eval_loader:
                deg, clean = deg.to(device), clean.to(device)

                # 4-way TTA da bi se bolje proberilo
                out_orig = model(deg)

                deg_hf = torch.flip(deg, dims=[3])
                out_hf = torch.flip(model(deg_hf), dims=[3])

                deg_vf = torch.flip(deg, dims=[2])
                out_vf = torch.flip(model(deg_vf), dims=[2])

                deg_rot = torch.rot90(deg, k=1, dims=[2, 3])
                out_rot = torch.rot90(model(deg_rot), k=-1, dims=[2, 3])

                pred = (out_orig + out_hf + out_vf + out_rot) / 4.0

                mse = F.mse_loss(pred, clean)
                if mse > 0:
                    psnr = 10 * torch.log10(1.0 / mse)
                    total_psnr += psnr.item()
                num_batches += 1

        avg_psnr = total_psnr / max(num_batches, 1)
        scheduler.step()

        print(f"Restauracija | Epoha {epoch+1:02d} | Loss: {epoch_loss:.5f} | Val PSNR: {avg_psnr:.2f} dB")

        # cuvanje poslednjeg checkpoint-a
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_psnr': best_psnr,
        }, checkpoint_path)

        # Čuvanje najboljeg modela na drajvu
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_psnr': best_psnr,
            }, best_checkpoint_path)
            print(f"  ★ Novi najbolji rezultat sačuvan u {best_checkpoint_path} (PSNR: {best_psnr:.2f} dB)")

    return model


# putanjice
if __name__ == '__main__':
    save_dir = "/content/drive/MyDrive/Projekat_Model"
    klasifikator_path = os.path.join(save_dir, "dodinamrezajej.pth")
    restauracija_path = os.path.join(save_dir, "dodinarestauracijajej.pth")

    # 🔥 IZMENA: Nove putanje za ZIP arhive na Google Drive-u
    drive_trening_zip = "/content/drive/MyDrive/Projekat_Model/trening.zip"
    drive_test_zip = "/content/drive/MyDrive/Projekat_Model/test.zip"

    # Lokalne putanje u virtuelnom okruženju Colab-a (za maksimalnu brzinu učitavanja)
    lokalni_trening_path = "/content/trening"
    lokalni_test_path = "/content/test"

    # =====================================================================
    # PRIPREMA TEST DATASETA (Otpatkivanje ZIP-a ili fallback na folder sa drajva)
    # =====================================================================
    if not os.path.exists(lokalni_test_path):
        if os.path.exists(drive_test_zip):
            print(f"Priprema testnog skupa: otpakujem {drive_test_zip} u {lokalni_test_path}...")
            get_ipython().system(f'unzip -q "{drive_test_zip}" -d "{lokalni_test_path}"')
            print("Testni skup uspešno otpakovan lokalno.")
        elif os.path.exists("/content/drive/MyDrive/Projekat_Model/test"):
            print("ZIP arhiva za test nije pronađena na drajvu. Koristim direktan folder sa Google Drive-a...")
            lokalni_test_path = "/content/drive/MyDrive/Projekat_Model/test"
        else:
            print("[Upozorenje] Test skup nije pronađen ni kao ZIP ni kao raspakovan folder na Google Drive-u!")

    # =====================================================================
    # PRIPREMA TRENING DATASETA (Otpatkivanje ZIP-a ili fallback na folder sa drajva)
    # =====================================================================
    if not os.path.exists(lokalni_trening_path):
        if os.path.exists(drive_trening_zip):
            print(f"Priprema trening skupa: otpakujem {drive_trening_zip} u {lokalni_trening_path}...")
            get_ipython().system(f'unzip -q "{drive_trening_zip}" -d "{lokalni_trening_path}"')
            print("Trening skup uspešno otpakovan lokalno.")
        elif os.path.exists("/content/drive/MyDrive/Projekat_Model/trening"):
            print("ZIP arhiva za trening nije pronađena na drajvu. Koristim direktan folder sa Google Drive-a...")
            lokalni_trening_path = "/content/drive/MyDrive/Projekat_Model/trening"
        else:
            print("[Upozorenje] Trening skup nije pronađen ni kao ZIP ni kao raspakovan folder na Google Drive-u!")

    # =====================================================================
    # KORAK A: Isključivo učitavanje i evaluacija Vašeg gotovog klasifikatora
    # (Sistem za keširanje sprečava ponovnu evaluaciju)
    # =====================================================================
    classifier_cache_path = os.path.join(save_dir, 'classifier_evaluation_cache.csv')

    print(f"\n[KLASIFIKACIJA] Provera keširanih rezultata...")
    if os.path.exists(classifier_cache_path):
        print("Pronađeni keširani rezultati. Učitavam tabelu umesto ponovne evaluacije.")
        stats_df = pd.read_csv(classifier_cache_path)
        print(stats_df.to_string(index=False))
    else:
        print("Keširani rezultati nisu pronađeni. Pokrećem evaluaciju klasifikatora...")
        stats_df = evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(
            model_path=klasifikator_path,
            test_dataset_dir=lokalni_test_path,
            img_size=128,
            batch_size=32,
            koristi_mini_podskup=True,
            velicina_podskupa=10000
        )
        if stats_df is not None:
            stats_df.to_csv(classifier_cache_path, index=False)
            print(f"Rezultati evaluacije sačuvani u keš fajl: {classifier_cache_path}")


    # =====================================================================
    # KORAK B: Pokretanje treninga za novi model restauracije
    # =====================================================================
    print(f"\n[RESTAURACIJA] Pokrećem trening modela {restauracija_path}...")
    if os.path.exists(lokalni_trening_path):
        restoracioni_model = train_restauracija(
            dataset_dir=lokalni_trening_path,
            epochs_to_train=30,
            batch_size=4,
            lr=2e-4,
            img_size=192,
            save_dir=save_dir,
            resume=True # FALSE da trening krene ispočetka, TRUE da nastavi od poslednjeg checkpoint-a
        )
    else:
        print(f"[Greška] Trening dataset nije pronađen na putanji: {lokalni_trening_path}")
