
from google.colab import drive
import os
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_metric


drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/Projekat_Model/test.zip'
if not os.path.exists(zip_path):
    zip_path = '/content/drive/MyDrive/Projekat_Model/DATASET_TEST.zip'

model_path = '/content/drive/MyDrive/Projekat_Model/dodinarestauracijabest.pth'
if not os.path.exists(model_path):
    model_path = '/content/drive/MyDrive/Projekat_Model/doroteinarestauracijabest.pth'

output_dir = '/content/drive/MyDrive/Projekat_Model/EVALUACIJA_REZULTATI'
local_extract_path = '/content/test'

# Otpakivanje dataset-a
if not os.path.exists(local_extract_path):
    print("Otpakujem dataset lokalno...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_extract_path)
    print("Dataset uspešno otpakovan.")


# arhitektura restauracije
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


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
            F.interpolate(spectral_feat, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False)
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
        sp = self.spectral_proj(F.interpolate(spectral, size=spatial.shape[2:], mode='bilinear', align_corners=False))
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

        self.aux_head3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(base_ch, out_channels, 3, padding=1, bias=False))
        self.aux_head2 = nn.Sequential(nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(base_ch // 2, out_channels, 3, padding=1))

        self.final_refinement = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, num_recursions=2), SpectralDecompositionRestorationBlock(base_ch), RecursiveDenseRestorationBlock(base_ch, num_recursions=2))
        self.output_head = nn.Sequential(nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(base_ch // 2, out_channels, 3, padding=1))
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        input_img = x
        s1, s1_skip = self.spatial_block1(x)
        s2, s2_skip = self.spatial_block2(s1)
        s3, s3_skip = self.spatial_block3(s2)
        s4, s4_skip = self.spatial_block4(s3)
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

        return self.contrast_color_recovery(d1_fused, input_img)


# ostecenja
DAMAGE_MAP = {
    'apply_anisotropic_diffusion': 'Anisotropic Diffusion (Vlaga/Zamućenje)',
    'apply_mold_and_decay': 'Mold and Decay (Buđ/Organski raspad)',
    'apply_chemical_aging': 'Chemical Aging (Oksidacija/Starenje)',
    'apply_fft_lpf': 'FFT LPF (Gubitak visokih frekvencija)',
    'apply_cracks': 'Cracks (Pukotine na laku/papiru)',
    'apply_paint_flaking': 'Paint Flaking (Ljuštenje boje)',
    'apply_water_stains': 'Water Stains (Mrlje od vode)',
    'apply_dust_and_scratches': 'Dust and Scratches (Prašina i ogrebotine)',
    'apply_combined_damage': 'Combined Damage (Teško kombinovano oštećenje)'
}

def detect_damage_type(filename):
    for key, name in DAMAGE_MAP.items():
        if key in filename:
            return name
    return "Other (Nepoznato oštećenje)"

def find_dataset_folders(base_path):
    for root, dirs, files in os.walk(base_path):
        if '0' in dirs and '1' in dirs:
            return os.path.join(root, '0'), os.path.join(root, '1')
    return None, None


# glavni deo petlje
def pokreni_evaluaciju():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Uređaj za evaluaciju: {device}")

    pojedinacne_dir = os.path.join(output_dir, 'pojedinacne')
    poredjenja_dir = os.path.join(output_dir, 'poredjenja')
    os.makedirs(pojedinacne_dir, exist_ok=True)
    os.makedirs(poredjenja_dir, exist_ok=True)

    print(f" model  {model_path}")
    model = Restauracija(base_ch=32).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("ucitan model")

    f0, f1 = find_dataset_folders(local_extract_path)
    if f0 is None or f1 is None:
        raise FileNotFoundError("Nisu pronađeni '0' i '1' folderi u test dataset-u")

    dmg_files = sorted([f for f in os.listdir(f1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    print(f" {len(dmg_files)} oštećenih slika za testiranje.")

    results_list = []
    transform_to_tensor = transforms.ToTensor()

    print("\nrestauracija i analiza")

    for dmg_f in tqdm(dmg_files, desc="Evaluacija"):
        parts = dmg_f.split('_')
        if len(parts) < 2:
            continue
        base_name = f"{parts[0]}_{parts[1]}"

        # spajanje
        clean_name = f"{base_name}_flip.jpg" if "_flip_" in dmg_f else f"{base_name}_clean.jpg"
        clean_path = os.path.join(f0, clean_name)

        if not os.path.exists(clean_path):
            alt_name = f"{base_name}_clean.jpg" if "_flip_" in dmg_f else f"{base_name}_flip.jpg"
            clean_path = os.path.join(f0, alt_name)

        dmg_path = os.path.join(f1, dmg_f)

        if not os.path.exists(clean_path):
            continue

        dmg_pil = Image.open(dmg_path).convert('RGB')
        clean_pil = Image.open(clean_path).convert('RGB')

        dmg_pil_resized = dmg_pil.resize((192, 192), Image.Resampling.BILINEAR)
        clean_pil_resized = clean_pil.resize((192, 192), Image.Resampling.BILINEAR)

        input_tensor = transform_to_tensor(dmg_pil_resized).unsqueeze(0).to(device)
        clean_tensor = transform_to_tensor(clean_pil_resized).unsqueeze(0).to(device)

        # DIREKTAN PROLAZ PREKO MREŽE (Čisto obnavljanje detalja)
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # PARALELAN PRORAČUN PSNR-a NA FLOAT TENZORIMA (Tačno kao u treningu!)
        mse_float = torch.mean((output_tensor - clean_tensor) ** 2).item()
        if mse_float > 0:
            psnr_val = 10.0 * np.log10(1.0 / mse_float)
        else:
            psnr_val = 100.0

        output_tensor_cpu = output_tensor.squeeze(0).cpu()
        restored_pil = transforms.ToPILImage()(output_tensor_cpu)

        clean_np = np.array(clean_pil_resized)
        dmg_np = np.array(dmg_pil_resized)
        restored_np = np.array(restored_pil)

        ssim_val = ssim_metric(clean_np, restored_np, channel_axis=2, data_range=255)
        mse_val = np.mean((clean_np.astype(np.float32) - restored_np.astype(np.float32)) ** 2)
        mae_val = np.mean(np.abs(clean_np.astype(np.float32) - restored_np.astype(np.float32)))

        dmg_type = detect_damage_type(dmg_f)

        results_list.append({
            'Filename': dmg_f,
            'Oštećenje': dmg_type,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'MSE': mse_val,
            'MAE': mae_val
        })

        # cuvanje samostalne slike u originalnoj velicini
        restored_original_size = restored_pil.resize(dmg_pil.size, Image.Resampling.BILINEAR)
        restored_original_size.save(os.path.join(pojedinacne_dir, f"restored_{dmg_f}"))

        # Uporedni prikaz (Osteceno | Restaurisano | Original)
        num_saved_for_type = sum(1 for r in results_list if r['Oštećenje'] == dmg_type)
        if num_saved_for_type <= 10:
            bar_height = 40
            h, w, _ = restored_np.shape
            combined_img = np.zeros((h + bar_height, w * 3, 3), dtype=np.uint8)

            combined_img[bar_height:, :w] = cv2.cvtColor(dmg_np, cv2.COLOR_RGB2BGR)
            combined_img[bar_height:, w:w*2] = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
            combined_img[bar_height:, w*2:] = cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_img, "OSTECENO", (10, 25), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(combined_img, "RESTAURISANO", (w + 10, 25), font, 0.45, (120, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(combined_img, "ORIGINALNA SLIKA", (2*w + 10, 25), font, 0.45, (255, 230, 150), 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(poredjenja_dir, f"compare_{dmg_f}"), combined_img)

    df = pd.DataFrame(results_list)

    statistika = df.groupby('Oštećenje').agg(
        Broj_Slike=('Filename', 'count'),
        Prosečan_PSNR=('PSNR', 'mean'),
        Prosečan_SSIM=('SSIM', 'mean'),
        Prosečan_MSE=('MSE', 'mean'),
        Prosečan_MAE=('MAE', 'mean')
    ).reset_index()

    statistika = statistika.sort_values(by='Prosečan_PSNR', ascending=False)

    csv_report_path = os.path.join(output_dir, 'izvestaj_metrika_test.csv')
    statistika.to_csv(csv_report_path, index=False)

    PINK = '\033[38;5;205m'
    RESET = '\033[0m'

    print(f"\n\n{PINK}{'='*100}")
    print(" EVALUACIJA USPEŠNO ZAVRŠENA")
    print(f"{'='*100}")
    print(f"Pojedinačne restaurisane slike sačuvane u:  {pojedinacne_dir}")
    print(f"Uporedni primeri slika sačuvani u:          {poredjenja_dir}")
    print(f"Tabela sačuvana na:                       {csv_report_path}\n")
    print("="*100)
    print("REZULTATI:")
    print("="*100)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(statistika.to_string(index=False))
    print("="*100 + RESET)

if __name__ == '__main__':
    pokreni_evaluaciju()
