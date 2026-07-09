import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
import shutil
import warnings
import logging

torch.backends.cudnn.benchmark = True

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))

#rekutzivni mikro blok koji istu konv gleda više puta
class RecursiveDenseMicroBlock(nn.Module):
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

#spektral deo isto kao i kod klas 
class SpectralDecomposeBlock(nn.Module):
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


#prostorni deo
class SpatialEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )
        self.dense_micro = RecursiveDenseMicroBlock(out_ch, num_recursions=3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.dense_micro(x)
        pooled = self.pool(x)
        return pooled, x
#za razmenu info i komunikaciju ihzhmedju dva razlciita domena
class AsymmetricCrossBridge(nn.Module):
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


class GatedFusionBlock(nn.Module):
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

        out_ch = s.shape[1]
        s_gate = gates[:, :out_ch]
        sp_gate = gates[:, out_ch:]
        return s_gate * s + sp_gate * sp


class DamageAttentionModule(nn.Module):
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

#integracija damage mapa
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # in_ch // 2 (upsampled) + skip_ch + 1 (za interpoliranu damage mapu)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch + 1, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )
        self.dense_micro = RecursiveDenseMicroBlock(out_ch, num_recursions=2)
        self.spectral = SpectralDecomposeBlock(out_ch)

    def forward(self, x: Tensor, skip: Tensor, damage_map: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip, dm], dim=1)
        x = self.conv(x)
        x = self.dense_micro(x)
        x = self.spectral(x)
        return x


# ============================================================
# DILATED CONTEXT BLOCK (ZA ŠIROKO VIDNO POLJE)
# ============================================================
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
        o1 = self.c1(x)
        o2 = self.c2(x)
        o3 = self.c3(x)
        o4 = self.c4(x)
        merged = torch.cat([o1, o2, o3, o4], dim=1)
        return F.relu(self.bn(self.fusion(merged)) + x)


# ============================================================
# ADAPTIVE RESIDUAL SCALING MODULE
# ============================================================
class AdaptiveResidualScaling(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 16, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(16, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        pooled = self.pool(x).view(b, c)
        alpha = self.mlp(pooled).view(b, 1, 1, 1) * 2.0
        return alpha


# ============================================================
# GATED SKIP CONNECTION (PAMETNE SKIP KONEKCIJE)
# ============================================================
class GatedSkipConnection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, skip: Tensor) -> Tensor:
        g = self.gate(skip)
        return skip * g


# ============================================================
# PARALELNA GRANA ZA IVICE (EDGE BRANCH)
# ============================================================
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
        gx = F.conv2d(x, self.kx, padding=1, groups=3)
        gy = F.conv2d(x, self.ky, padding=1, groups=3)
        edge_feats = torch.cat([gx, gy], dim=1)
        return self.conv(edge_feats)


# ============================================================
# DODINA RESTAURACIJA V2 — RE-OPTIMIZOVANA ARHITEKTURA (3.7M PARAMETARA)
# ============================================================
class DodinaRestauracijaV2(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 32):
        super().__init__()

        self.adaptive_alpha = AdaptiveResidualScaling(in_channels=3)
        self.edge_branch = EdgeBranch(out_channels=base_ch)
        self.edge_fusion = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)

        # ===================== ENCODER - SPATIAL GRANA =====================
        self.spatial_block1 = SpatialEncoderBlock(in_channels, base_ch)       # 32
        self.spatial_block2 = SpatialEncoderBlock(base_ch, base_ch * 2)       # 64
        self.spatial_block3 = SpatialEncoderBlock(base_ch * 2, base_ch * 4)   # 128
        self.spatial_block4 = SpatialEncoderBlock(base_ch * 4, base_ch * 8)   # 256

        # ===================== ENCODER - SPECTRAL GRANA =====================
        self.spectral_init = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, base_ch),
            nn.ReLU(inplace=False)
        )
        self.spectral_block1 = SpectralDecomposeBlock(base_ch)                # 32
        self.spectral_pool1 = nn.MaxPool2d(2)
        self.spec_proj1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 1, bias=False),
            nn.GroupNorm(4, base_ch * 2),
            nn.ReLU(inplace=False)
        )
        self.spectral_block2 = SpectralDecomposeBlock(base_ch * 2)            # 64
        self.spectral_pool2 = nn.MaxPool2d(2)
        self.spec_proj2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 1, bias=False),
            nn.GroupNorm(4, base_ch * 4),
            nn.ReLU(inplace=False)
        )
        self.spectral_block3 = SpectralDecomposeBlock(base_ch * 4)            # 128
        self.spectral_pool3 = nn.MaxPool2d(2)
        self.spec_proj3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 1, bias=False),
            nn.GroupNorm(4, base_ch * 8),
            nn.ReLU(inplace=False)
        )
        self.spectral_block4 = SpectralDecomposeBlock(base_ch * 8)            # 256

        self.cross1 = AsymmetricCrossBridge(base_ch, base_ch, base_ch)
        self.cross2 = AsymmetricCrossBridge(base_ch * 2, base_ch * 2, base_ch * 2)
        self.cross3 = AsymmetricCrossBridge(base_ch * 4, base_ch * 4, base_ch * 4)
        self.cross4 = AsymmetricCrossBridge(base_ch * 8, base_ch * 8, base_ch * 8)

        self.gated_fusion = GatedFusionBlock(base_ch * 8, base_ch * 8, base_ch * 8)  # 256
        self.damage_attention = DamageAttentionModule(base_ch * 8)  # 256

        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False),
            nn.GroupNorm(4, base_ch * 8),
            nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8),
            RecursiveDenseMicroBlock(base_ch * 8, num_recursions=2),
        )
    #postepena rekonstrukcija detalja
        self.decoder4 = DecoderBlock(base_ch * 8, base_ch * 8, base_ch * 4) # 256 + 256 -> 128
        self.decoder3 = DecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2) # 128 + 128 -> 64
        self.decoder2 = DecoderBlock(base_ch * 2, base_ch * 2, base_ch)     # 64 + 64 -> 32
        self.decoder1 = DecoderBlock(base_ch, base_ch, base_ch)             # 32 + 32 -> 32

        # ===================== GATED SKIP MODULES =====================
        self.skip_gate1 = GatedSkipConnection(base_ch)
        self.skip_gate2 = GatedSkipConnection(base_ch * 2)
        self.skip_gate3 = GatedSkipConnection(base_ch * 4)
        self.skip_gate4 = GatedSkipConnection(base_ch * 8)

        # ===================== SKIP ENHANCEMENT =====================
        self.skip_refine1 = nn.Sequential(
            RecursiveDenseMicroBlock(base_ch, num_recursions=2),
            SpectralDecomposeBlock(base_ch)
        )
        self.skip_refine2 = nn.Sequential(
            RecursiveDenseMicroBlock(base_ch * 2, num_recursions=2),
            SpectralDecomposeBlock(base_ch * 2)
        )
        self.skip_refine3 = nn.Sequential(
            RecursiveDenseMicroBlock(base_ch * 4, num_recursions=2),
            SpectralDecomposeBlock(base_ch * 4)
        )
        self.skip_refine4 = nn.Sequential(
            RecursiveDenseMicroBlock(base_ch * 8, num_recursions=2),
            SpectralDecomposeBlock(base_ch * 8)
        )

        # ===================== DEEP SUPERVISION =====================
        self.aux_head3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_ch, out_channels, 3, padding=1, bias=False),
        )
        self.aux_head2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_ch // 2, out_channels, 3, padding=1, bias=False),
        )

        # ===================== REFINEMENT & OUTPUT HEAD =====================
        self.final_refinement = nn.Sequential(
            RecursiveDenseMicroBlock(base_ch, num_recursions=2),
            SpectralDecomposeBlock(base_ch),
            RecursiveDenseMicroBlock(base_ch, num_recursions=2)
        )
        
        self.output_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_ch // 2, 4, 3, padding=1), 
        )

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        input_img = x

        s1, s1_skip = self.spatial_block1(x)
        s2, s2_skip = self.spatial_block2(s1)
        s3, s3_skip = self.spatial_block3(s2)
        s4, s4_skip = self.spatial_block4(s3)

        sp0 = self.spectral_init(x)
        sp1 = self.spectral_block1(sp0)
        sp1_p = self.spec_proj1(self.spectral_pool1(sp1))
        sp2 = self.spectral_block2(sp1_p)
        sp2_p = self.spec_proj2(self.spectral_pool2(sp2))
        sp3 = self.spectral_block3(sp2_p)
        sp3_p = self.spec_proj3(self.spectral_pool3(sp3))
        sp4 = self.spectral_block4(sp3_p)

        # ============ CROSS BRIDGES ============
        c1 = self.cross1(s1_skip, sp1)
        c2 = self.cross2(s2_skip, sp2)
        c3 = self.cross3(s3_skip, sp3)
        c4 = self.cross4(s4_skip, sp4)

        s4_enriched = s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:])

        # ============ BOTTLENECK ============
        fused = self.gated_fusion(s4_enriched, sp4)
        attended, damage_map = self.damage_attention(fused)
        bottleneck_out = self.bottleneck_refine(attended)

        # ============ GATED SKIP CONNECTIONS ============
        s4_gated = self.skip_gate4(s4_skip)
        s3_gated = self.skip_gate3(s3_skip)
        s2_gated = self.skip_gate2(s2_skip)
        s1_gated = self.skip_gate1(s1_skip)

        # ============ SKIP ENHANCEMENT ============
        skip4_enhanced = s4_gated + F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        skip4_enhanced = self.skip_refine4(skip4_enhanced)

        skip3_enhanced = s3_gated + F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        skip3_enhanced = self.skip_refine3(skip3_enhanced)

        skip2_enhanced = s2_gated + F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        skip2_enhanced = self.skip_refine2(skip2_enhanced)

        skip1_enhanced = s1_gated + F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)
        skip1_enhanced = self.skip_refine1(skip1_enhanced)

        # ============ DECODER ============
        d4 = self.decoder4(bottleneck_out, skip4_enhanced, damage_map)
        d3 = self.decoder3(d4, skip3_enhanced, damage_map)
        d2 = self.decoder2(d3, skip2_enhanced, damage_map)
        d1 = self.decoder1(d2, skip1_enhanced, damage_map)

        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)

        d1_refined = self.final_refinement(d1)
        
        edge_feats = self.edge_branch(input_img)
        d1_fused = self.edge_fusion(torch.cat([d1_refined, edge_feats], dim=1))

        out_feat = self.output_head(d1_fused)
        residual = out_feat[:, :3]
        confidence = torch.sigmoid(out_feat[:, 3:4])
        
        alpha_val = self.adaptive_alpha(input_img)
        
        out = torch.clamp(input_img + alpha_val * confidence * residual, 0.0, 1.0)

        if self.training:
            aux3 = self.aux_head3(d3)
            aux3 = F.interpolate(aux3, size=input_img.shape[2:], mode='bilinear', align_corners=False)
            aux3 = torch.clamp(input_img + alpha_val * aux3, 0.0, 1.0)

            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(aux2, size=input_img.shape[2:], mode='bilinear', align_corners=False)
            aux2 = torch.clamp(input_img + alpha_val * aux2, 0.0, 1.0)

            return {
                'out': out,
                'aux2': aux2,
                'aux3': aux3,
                'damage_map': damage_map,
            }

        return out


# ============================================================
# SSIM LOSS (BEZ EKSTERNIH PAKETA - BEZBEDNO OD NaNs)
# ============================================================
def gaussian(window_size: int, sigma: float) -> Tensor:
    gauss = torch.tensor([
        np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.register_buffer('window', create_window(window_size, self.channel))

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-6)

        return 1.0 - ssim_map.mean()


# ============================================================
# SOBEL EDGE LOSS
# ============================================================
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


# ============================================================
# CUSTOM PERCEPTUAL LOSS (BEZ EKSTERNIH MODELA)
# ============================================================
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
            if s > 1:
                x_scaled = F.interpolate(x, scale_factor=1.0 / s, mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            
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


# ============================================================
# MEKI HARD EXAMPLE MINING (BEZBEDAN I IMUN NA NaNs)
# ============================================================
class SoftHardExampleMiningLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        error_map = torch.abs(pred - target).mean(dim=1, keepdim=True).detach()
        weight = 1.0 + torch.sigmoid((error_map - error_map.mean()) / (error_map.std() + 1e-6))
        return weight


# ============================================================
# FREQUENCY CONSISTENCY LOSS
# ============================================================
class FrequencyConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_low = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        tgt_low = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        
        pred_high = pred - pred_low
        tgt_high = target - tgt_low
        
        loss_low = F.l1_loss(pred_low, tgt_low)
        loss_high = F.l1_loss(pred_high, tgt_high)
        
        return 0.4 * loss_low + 0.6 * loss_high


# ============================================================
# DODINA RESTAURACIJA LOSS (SVEUBUHVATAN I BEZBEDAN OD NaNs)
# ============================================================
class DodinaRestauracijaLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.sobel = SobelLoss()
        self.ssim = SSIMLoss()
        self.perceptual = CustomPerceptualLoss()
        self.soft_hem = SoftHardExampleMiningLoss()
        self.freq_consistency = FrequencyConsistencyLoss()

    def single_scale_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        hem_weight = self.soft_hem(pred, target)
        
        char_diff = torch.sqrt((pred - target) ** 2 + self.eps)
        char_loss = torch.mean(char_diff * hem_weight)
        
        ssim_loss = self.ssim(pred, target)
        edge_loss = self.sobel(pred, target)
        percep_loss = self.perceptual(pred, target)
        freq_loss = self.freq_consistency(pred, target)

        # NaNs stabilizovan FFT gubitak: poredimo realni i imaginarni deo bez korenovanja/uglova
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        fft_loss = F.l1_loss(torch.real(pred_fft), torch.real(target_fft)) + \
                   F.l1_loss(torch.imag(pred_fft), torch.imag(target_fft))

        return (
            0.20 * char_loss +
            0.25 * ssim_loss +
            0.15 * edge_loss +
            0.15 * fft_loss +
            0.15 * percep_loss +
            0.10 * freq_loss
        )

    def multi_scale_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.single_scale_loss(pred, target)

        pred2 = F.interpolate(pred, scale_factor=0.5, mode='bilinear', align_corners=False)
        tgt2 = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False)
        char2 = torch.mean(torch.sqrt((pred2 - tgt2) ** 2 + self.eps))
        loss += 0.5 * char2

        pred4 = F.interpolate(pred, scale_factor=0.25, mode='bilinear', align_corners=False)
        tgt4 = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=False)
        char4 = torch.mean(torch.sqrt((pred4 - tgt4) ** 2 + self.eps))
        loss += 0.25 * char4

        return loss

    def forward(self, pred_outputs: Tensor | dict[str, Tensor], target: Tensor) -> Tensor:
        if isinstance(pred_outputs, dict):
            total_loss = self.multi_scale_loss(pred_outputs['out'], target)
            
            if 'aux3' in pred_outputs:
                total_loss += 0.3 * torch.mean(torch.sqrt((pred_outputs['aux3'] - target) ** 2 + self.eps))
            if 'aux2' in pred_outputs:
                total_loss += 0.15 * torch.mean(torch.sqrt((pred_outputs['aux2'] - target) ** 2 + self.eps))
            return total_loss
        else:
            return self.multi_scale_loss(pred_outputs, target)


# ============================================================
# DETALJNO MAPIRANJE DATASETA (UPARIVANJE FOLDERA 0 I 1)
# ============================================================
class RestorationDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int = 192, train: bool = True, preload_to_ram: bool = True, shared_cached_pairs=None):
        self.img_size = img_size
        self.train = train
        
        folder_0 = os.path.join(dataset_dir, '0') # Čiste slike (Ground Truth)
        folder_1 = os.path.join(dataset_dir, '1') # Degradirane slike (Input)
        
        if not os.path.exists(folder_0) or not os.path.exists(folder_1):
            raise FileNotFoundError(f"Greška: Direktorijumi '0' i/ili '1' ne postoje na putanji {dataset_dir}!")
            
        # Sva oštećenja se nalaze u folderu 1
        dmg_files = sorted([f for f in os.listdir(folder_1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        
        self.pairs = []
        for dmg_f in dmg_files:
            parts = dmg_f.split('_')
            if len(parts) >= 2:
                base_name = f"{parts[0]}_{parts[1]}"
            else:
                continue
                
            if "_orig_" in dmg_f:
                clean_name = f"{base_name}_clean.jpg"
            elif "_flip_" in dmg_f:
                clean_name = f"{base_name}_flip.jpg"
            else:
                clean_name = f"{base_name}_clean.jpg"
                
            clean_path = os.path.join(folder_0, clean_name)
            dmg_path = os.path.join(folder_1, dmg_f)
            
            if os.path.exists(clean_path):
                self.pairs.append((dmg_path, clean_path))
                
        self.transform_deg = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.transform_clean = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        self.preload_to_ram = preload_to_ram
        self.cached_pairs = []
        
        # Ako imamo podeljene keširane parove, koristimo ih odmah
        if shared_cached_pairs is not None:
            self.cached_pairs = shared_cached_pairs
        elif self.preload_to_ram and len(self.pairs) > 0:
            print(f"[Dataset] Učitavanje {len(self.pairs)} parova slika direktno u RAM...")
            start_time = time.time()
            for deg_path, clean_path in self.pairs:
                deg_img = Image.open(deg_path).convert('RGB')
                clean_img = Image.open(clean_path).convert('RGB')
                
                deg_img = deg_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
                clean_img = clean_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
                
                self.cached_pairs.append((deg_img, clean_img))
            print(f"[Dataset] Pre-loading uspešno završen za {time.time() - start_time:.1f}s.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.preload_to_ram:
            deg_img, clean_img = self.cached_pairs[idx]
        else:
            deg_path, clean_path = self.pairs[idx]
            deg_img = Image.open(deg_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')
            
        deg_t = self.transform_deg(deg_img)
        clean_t = self.transform_clean(clean_img)
        
        if self.train:
            if torch.rand(1) > 0.5:
                deg_t = torch.flip(deg_t, dims=[2])
                clean_t = torch.flip(clean_t, dims=[2])
            if torch.rand(1) > 0.5:
                deg_t = torch.flip(deg_t, dims=[1])
                clean_t = torch.flip(clean_t, dims=[1])
                
        return deg_t, clean_t


# ============================================================
# TRENING FUNKCIJA — ČIST ISPIS, OPTIMIZOVANO I BEZ DUPLIRANJA
# ============================================================
def train_dodina_restauracija(
    dataset_dir: str,
    epochs_to_train: int = 30,
    batch_size: int = 16,
    lr: float = 2e-4,
    img_size: int = 192,
    val_split: float = 0.1,
    save_dir: str = '.',
    resume: bool = True,
):
    # ═══ UGASI SVA UPOZORENJA ═══
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ["PYTHONWARNINGS"] = "ignore"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Uređaj: {device}")

    # Dataset — Učitavamo slike samo JEDNOM u RAM i delimo ih kako bismo uštedeli vreme i memoriju
    train_dataset = RestorationDataset(dataset_dir, img_size=img_size, train=True, preload_to_ram=True)
    val_dataset = RestorationDataset(dataset_dir, img_size=img_size, train=False, preload_to_ram=True, shared_cached_pairs=train_dataset.cached_pairs)

    if len(train_dataset) == 0:
        print("GREŠKA: Prazan dataset!")
        return None

    # Split indices
    np.random.seed(42)
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    val_count = int(len(indices) * val_split)

    train_subset = Subset(train_dataset, indices[val_count:])
    val_subset = Subset(val_dataset, indices[:val_count])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train uzoraka: {len(train_subset)} | Val uzoraka: {len(val_subset)}")

    model = DodinaRestauracijaV2(base_ch=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametara: {total_params:,}")

    # .to(device) za gubike popravlja RuntimeError sa Sobel/CustomPerceptual matricama
    criterion = DodinaRestauracijaLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train, eta_min=1e-6)

    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(save_dir, 'dodina_restauracija_v2_latest.pth')

    # Resume checkpoint
    if resume and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Nastavak od epohe {start_epoch+1}. Best Loss: {best_val_loss:.5f}")
        except Exception:
            print("Checkpoint nekompatibilan. Krećem ispočetka.")
            start_epoch = 0
            best_val_loss = float('inf')

    total_epochs = start_epoch + epochs_to_train

    print(f"\n{'='*70}")
    print(f"{'Epoha':<10}{'Train Loss':<14}{'Val Loss':<14}{'PSNR (dB)':<12}{'LR':<12}{'Status'}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()

        # ═══ TRENING ═══
        model.train()
        running_loss = 0.0
        for deg, clean in train_loader:
            deg, clean = deg.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(deg)
            loss = criterion(outputs, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ═══ VALIDACIJA + PSNR ═══
        model.eval()
        val_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
        with torch.no_grad():
            for deg, clean in val_loader:
                deg, clean = deg.to(device), clean.to(device)
                outputs = model(deg)
                loss = criterion(outputs, clean)
                val_loss += loss.item()

                if isinstance(outputs, dict):
                    pred = outputs['out']
                else:
                    pred = outputs
                mse = F.mse_loss(pred, clean)
                if mse > 0:
                    psnr = 10 * torch.log10(1.0 / mse)
                    total_psnr += psnr.item()
                num_batches += 1

        val_loss /= len(val_loader)
        avg_psnr = total_psnr / max(num_batches, 1)
        scheduler.step()

        elapsed = time.time() - start_time

        # ═══ ČUVANJE NAJBOLJEG ═══
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = "★ BEST"
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)

        print(f"{epoch+1:>3}/{total_epochs:<6}{train_loss:<14.5f}{val_loss:<14.5f}{avg_psnr:<12.2f}{scheduler.get_last_lr()[0]:<12.6f}{marker}")

    print(f"{'='*70}")
    print(f"gotojo -  najbolji Val Loss: {best_val_loss:.5f}")
    print(f"Model sacuvan {checkpoint_path}")

    return model


if __name__ == '__main__':
    import argparse
    
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description="Trening Dodina Restauracija V2")
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Mod rada")
    parser.add_argument("--epochs", type=int, default=30, help="Broj epoha za trening")
    parser.add_argument("--batch_size", type=int, default=16, help="Veličina batch-a")
    parser.add_argument("--lr", type=float, default=2e-4, help="Stopa učenja (learning rate)")
    
    args, _ = parser.parse_known_args()

    if not os.path.exists("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            pass

    drive_zip_path = "/content/drive/MyDrive/Projekat_Model/DATASET_TRENING.zip"
    local_dataset_path = "/content/DATASET_TRENING"

    if not os.path.exists(local_dataset_path):
        if os.path.exists(drive_zip_path):
            print("Otpakujem dataset...")
            os.system(f'unzip -q "{drive_zip_path}" -d "/content/"')
            if os.path.exists("/content/DATASET_TRENING/DATASET_TRENING"):
                shutil.move("/content/DATASET_TRENING/DATASET_TRENING", "/content/DATASET_TRENING_TEMP")
                shutil.rmtree("/content/DATASET_TRENING")
                shutil.move("/content/DATASET_TRENING_TEMP", "/content/DATASET_TRENING")
            print("dataset gotoj")
        else:
            print("nema dataseta")
            exit()

    train_dodina_restauracija(
        dataset_dir=local_dataset_path,
        epochs_to_train=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=192,
        val_split=0.1,
        save_dir="/content/drive/MyDrive/Projekat_Model",
        resume=True,
    )
