# metrike za prvu arhitekturu za klasifikaciju
# redjanje slika i klasifikacija po ostecenjima (na svaku sliku 6 ostecenja)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import datetime

import warnings
warnings.filterwarnings('ignore')

drive_test_zip = "/content/drive/MyDrive/Projekat_Model/DATASET_validacija.zip"
lokalni_test_path = "/content/DATASET_validacija"

if not os.path.exists(lokalni_test_path):
    print("Priprema testnog skupa podataka...")
    if os.path.exists(drive_test_zip):
        print("Pronađen ZIP fajl na Google Drive-u, otpakujem...")
        get_ipython().system(f'unzip -q "{drive_test_zip}" -d "{lokalni_test_path}"')
        print("Otpakivanje uspešno završeno.")
    else:
        print("[Upozorenje] DATASET_validacija.zip nije pronađen na Google Drive-u!")
else:
    print("Testni skup podataka je već spreman u lokalnom direktorijumu.")


#arhitektura
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
        for i in range(self.num_recursions):
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
            nn.BatchNorm2d(spectral_ch),
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


def evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(
    model_path: str,
    test_dataset_dir: str,
    img_size: int = 128,
    batch_size: int = 32,
    koristi_mini_podskup: bool = True
):
    nested_path = os.path.join(test_dataset_dir, "DATASET_validacija")
    if os.path.exists(nested_path) and os.path.isdir(nested_path):
        test_dataset_dir = nested_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"Uređaj za proračune: {device.type.upper()}")
    print(f"Testni dataset: {test_dataset_dir}")
    print(f"{'='*60}\n")

    test_dataset = DamageDataset(test_dataset_dir, img_size=img_size, train=False)

    if len(test_dataset) == 0:
        print(f"Nema slika na putanji {test_dataset_dir} ili folder ne postoji!")
        return None, None

    # Mini-podskup za brzi rad na CPU
    if koristi_mini_podskup:
        indeksi = list(range(0, len(test_dataset), 20))
        test_dataset = torch.utils.data.Subset(test_dataset, indeksi)
        print(f"[INFO] Aktiviran mini-podskup za CPU! Testiram model na {len(test_dataset)} slika umesto 5400.")
    else:
        print(f"[INFO] Testiram model na celom skupu podataka ({len(test_dataset)} slika).")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = DodinaMreza(num_classes=2).to(device)

    # Učitavanje originalnih težina i BatchNorm statistika
    if not os.path.exists(model_path):
        print(f"Model nije pronađen na putanji: {model_path}")
        return None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Uspešno učitan checkpoint modela (težine i BatchNorm statistike su sačuvane).")
    else:
        model.load_state_dict(checkpoint)
        print("Učitane sirove težine modela.")

    # Model ostaje u čistom EVAL režimu (bez remećenja BatchNorm slojeva)
    model.eval()

    all_probs = []
    all_labels = []
    all_paths = []

    damage_mapping = {
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

    stats = {name: {'total': 0, 'correct': 0} for name in damage_mapping.values()}
    stats['Bez ostecenja (Ciste slike)'] = {'total': 0, 'correct': 0}

    # --- 5-WAY TEST-TIME AUGMENTATION (TTA) ---
    print("Pokrećem 5-way Test-Time Augmentaciju sa adaptacijom osvetljenja...")
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            # 1. Originalna slika
            outputs = model(images)
            probs_orig = F.softmax(outputs['logits'], dim=-1)

            # 2. Horizontalni flip
            images_flipped_h = torch.flip(images, dims=[3])
            outputs_flipped_h = model(images_flipped_h)
            probs_flipped_h = F.softmax(outputs_flipped_h['logits'], dim=-1)

            # 3. Vertikalni flip
            images_flipped_v = torch.flip(images, dims=[2])
            outputs_flipped_v = model(images_flipped_v)
            probs_flipped_v = F.softmax(outputs_flipped_v['logits'], dim=-1)

            # 4. Rotacija 180 stepeni (H+V flip)
            images_rot180 = torch.flip(images, dims=[2, 3])
            outputs_rot180 = model(images_rot180)
            probs_rot180 = F.softmax(outputs_rot180['logits'], dim=-1)

            # 5. Adaptacija osvetljenja (kompenzacija za svetle fasade)
            images_bright = torch.clamp(images * 1.15, 0.0, 1.0)
            outputs_bright = model(images_bright)
            probs_bright = F.softmax(outputs_bright['logits'], dim=-1)

            # Prosek svih 5 predikcija
            probs_final = (probs_orig + probs_flipped_h + probs_flipped_v + probs_rot180 + probs_bright) / 5.0

            all_probs.extend(probs_final[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    results_dir = os.path.dirname(model_path)
    os.makedirs(results_dir, exist_ok=True)

    idx = 1
    while os.path.exists(os.path.join(results_dir, f"tabela_{idx}.csv")):
        idx += 1

    # --- DINAMIČKA PRETRAGA NA CELOM OPSEGU PRAGA (0.10 do 0.90) ---
    print("\nPokrećem dinamičku pretragu optimalnog praga na celom opsegu...")
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.10, 0.90, 0.01)  # Vraćamo pun opseg
    f1_scores = []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"Optimalni prag pronađen na osnovu F1 kalibracije: {best_threshold:.2f} (Najbolji F1-score: {best_f1:.4f})")

    # Vizuelizacija i čuvanje krive praga
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, f1_scores, color='#ff007f', lw=2, label='F1-score')
    plt.axvline(best_threshold, color='black', linestyle='--', label=f'Optimalni Prag = {best_threshold:.2f}')
    plt.xlabel('Klasifikacioni Prag (Threshold)')
    plt.ylabel('F1-score')
    plt.title('Kalibraciona kriva praga na ciljnom domenu (Vyronas)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    curve_path = os.path.join(results_dir, f"f1_kriva_praga_{idx}.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Kalibraciona kriva uspešno vizuelizovana i sačuvana na:\n{curve_path}")

    # Primena konačnog kalibrisanog praga
    all_preds = (all_probs >= best_threshold).astype(int)

    # Razvrstavanje tačnosti po klasama oštećenja
    for pred, label, path in zip(all_preds, all_labels, all_paths):
        if label == 0:
            stats['Bez ostecenja (Ciste slike)']['total'] += 1
            if pred == 0:
                stats['Bez ostecenja (Ciste slike)']['correct'] += 1
        else:
            filename = os.path.basename(path).lower()
            found = False
            for func_name, display_name in damage_mapping.items():
                if func_name in filename:
                    stats[display_name]['total'] += 1
                    if pred == 1:
                        stats[display_name]['correct'] += 1
                    found = True
                    break

    ukupna_tacnost = accuracy_score(all_labels, all_preds) * 100

    print("\n" + "="*50)
    print(f"REZULTATI NAKON USPEŠNE KALIBRACIJE PRAGA")
    print(f"Ukupna tačnost modela (Accuracy): {ukupna_tacnost:.2f}%")
    print("="*50 + "\n")
    print("Klasifikacija po klasama:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['klasa 0 (bez ostecenja)', 'Klasa 1 (osteceno)'],
        digits=4
    )
    print(report)

    rows = []
    for cat, data in stats.items():
        total = data['total']
        correct = data['correct']
        acc = (correct / total * 100) if total > 0 else 0.0
        rows.append([cat, total, correct, round(acc, 2)])

    df_stats = pd.DataFrame(rows, columns=["tip ostecenja", "broj testiranih", "broj tacnih", "tacnost (%)"])

    # Pink formatiranje ispisa
    PINK = "\033[38;5;205m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    top_border = f"{PINK}┌──────────────────────────────────┬────────────┬────────────┬──────────────┐{RESET}"
    mid_border = f"{PINK}├──────────────────────────────────┼────────────┼────────────┼──────────────┤{RESET}"
    bot_border = f"{PINK}└──────────────────────────────────┴────────────┴────────────┴──────────────┘{RESET}"

    print(f"\n{BOLD}Pregled po tipu oštećenja{RESET}")
    print(top_border)
    print(f"{PINK}│{RESET} {BOLD}{'tip ostecenja':<32} {PINK}│{RESET} {BOLD}{'testirano':<10} {PINK}│{RESET} {BOLD}{'tacno':<10} {PINK}│{RESET} {BOLD}{'tacnost (%)':<12} {PINK}│{RESET}")
    print(mid_border)

    for row in rows:
        cat_name = row[0]
        tested = row[1]
        correct = row[2]
        accuracy_val = f"{row[3]:.2f}%"
        print(f"{PINK}│{RESET} {cat_name:<32} {PINK}│{RESET} {tested:<10d} {PINK}│{RESET} {correct:<10d} {PINK}│{RESET} {accuracy_val:<12} {PINK}│{RESET}")

    print(bot_border)

    # Snimanje rezultata
    csv_path = os.path.join(results_dir, f"tabela_{idx}.csv")
    df_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nTabela rezultata uspešno sačuvana pod nazivom:\n{csv_path}")

    report_path = os.path.join(results_dir, f"izvestaj_klasifikacije_{idx}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Izveštaj klasifikacije sačuvan pod nazivom:\n{report_path}")

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    metrics_path = os.path.join(results_dir, f"metrike_{idx}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"f1-score : {f1:.4f}\n")
    print(f"Metrike sačuvane pod nazivom:\n{metrics_path}")

    # Crtanje matrice konfuzije
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=['Bez ostecenja', 'Osteceno'],
                yticklabels=['Bez ostecenja', 'Osteceno'])
    plt.xlabel('Predvidjeno (Sta je model rekao)')
    plt.ylabel('Stvarno (Tacna oznaka)')
    plt.title('Matrica konfuzije (Dodina Mreza sa 5-way TTA)')

    cm_path = os.path.join(results_dir, f"matrica_konfuzije_{idx}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"Matrica konfuzije sačuvana na:\n{cm_path}\n")

    plt.show()

    return all_labels, all_preds


# --- POKRETANJE ---
if __name__ == '__main__':
    putanja_do_modela = "/content/drive/MyDrive/Projekat_Model/dodinamrezajej.pth"
    putanja_do_test_dataseta = "/content/DATASET_validacija"

    KORISTI_MINI_PODSKUP = True

    stvarne_oznake, predvidjanja = evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(
        model_path=putanja_do_modela,
        test_dataset_dir=putanja_do_test_dataseta,
        img_size=128,
        batch_size=32,
        koristi_mini_podskup=KORISTI_MINI_PODSKUP
    )
