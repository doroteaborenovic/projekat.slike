#metrike za prvu arhitekturu za kalsifikaciju
#ređanje slika i klsifikacija po oštećenjima (na svaku sliku 6 oštećenja)
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

import warnings
warnings.filterwarnings('ignore')

drive_test_zip = "/content/drive/MyDrive/Projekat_Model/DATASET_TEST.zip"
lokalni_test_path = "/content/DATASET_TEST"

if not os.path.exists(lokalni_test_path):
    print("Priprema")
    if os.path.exists(drive_test_zip):
        print("Pronađen arhivirani testni skup podataka na Google Drive-u")
        get_ipython().system(f'unzip -q "{drive_test_zip}" -d "/content/"')
        print("Raspakivanje završeno")
    else:
        print(" DATASET_TEST.zip nije pronađen")
else:
    print("Testni skup podataka je već spreman u /content/")


#ovde je arhitektrua modela
class RecursiveDenseMicroBlock(nn.Module):
    #blok koji više puta obrađuje istu sliku da bi izvukao sitne detalje.
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
    #za moj model ovo je jedna od najbitnijih delova
    #ovaj deo razdvaja sliku na dve različite vrste infromacija
        #prvo je na nsike frekv(glatke oblasti tipa svetlo) i visoke frekv (ivice šum tekstura)
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
#SpatialBlock obrađuje prostorne karakteristike slike
#kombinujući osnovne konvolucione filtere, rekurzivnu obradu detalja i redukciju rezolucije radi efikasnijeg izdvajanja vizuelnih informacija.
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dense_micro = RecursiveDenseMicroBlock(out_ch, num_recursions=3) #pojačava detalje tj prvo vidi osnovno, pa onda ide dublje u sliku
        #tako se poboljšava preciznost detalja
        self.pool = nn.MaxPool2d(2) #smanjuje rezolucije

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.dense_micro(x)
        pooled = self.pool(x)
        return pooled, x


class AsymmetricCrossBridge(nn.Module):
#ovde je kao komunikacija spektralnog i tog detaljnog dela
#
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_to_spectral = nn.Sequential(
            nn.Conv2d(spatial_ch, spectral_ch, 1), #pretvara prostorne informacije u format koji spectral može da razume
            nn.BatchNorm2d(spectral_ch),
            nn.ReLU(inplace=True)
        )
        self.spectral_to_spatial = nn.Sequential(
            nn.Conv2d(spectral_ch, spatial_ch, 1), 
            nn.BatchNorm2d(spatial_ch),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv2d(spatial_ch + spectral_ch, out_ch, 1) #uzima informacije iz oba dela i kombinuje ih u finalnu odluku

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
    #ovaj blok spaja prostorne ifnormacije i  spektralne informacije
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_proj = nn.Conv2d(spatial_ch, out_ch, 1) #pretvara spatial feature-e u isti prostor dimenzija
        self.spectral_proj = nn.Conv2d(spectral_ch, out_ch, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch * 2, out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // 4, out_ch * 2),
            nn.Sigmoid()
        ) #ovde se gleda cela slika, globalno izračunava koliko je šta važno i daje težine (weights) 
        #težine su brojevi koje model uči tokom treninga da bi odlučio koliko da veruje nekoj određenoj info

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
    #ovaj deo pornalazi delove slike koji liče na oštećenja i model uči gde treba d agleda kad locira oštećenja
    #i generiše se mapa pažnje
    #naglašavaju se rgije slike gde su potencijalna oštećenja
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
    # mreža kombinuje prostornu svesnost sa frekvencijskom analizom
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        # prostor
        self.spatial_block1 = SpatialBlock(in_channels, 64)
        self.spatial_block2 = SpatialBlock(64, 128)
        self.spatial_block3 = SpatialBlock(128, 256)

        # spektar
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

#cross conection  su zapravo mostovi izmdju moja dva toka mreže tj prostor i spektar
        self.cross1 = AsymmetricCrossBridge(64, 64, 64) #ovo je  za osnvoe ivice
        self.cross2 = AsymmetricCrossBridge(128, 128, 128) #ovo je za teksture i oblike
        self.cross3 = AsymmetricCrossBridge(256, 256, 256)#ovo je za kao neke teže delove

        # fuzija i klasifikacija
        self.gated_fusion = GatedFusionBlock(256, 256, 512) #ovde je glavna odluka i bira model šta je važnija tj tačnije
        self.damage_attention = DamageAttentionModule(512) #posledni  deo koji gleda šta je gde i šta je bitno i gde ide pažnja

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
#Cross connection blokovi omogućavaju razmenu informacija između prostornog i spektralnog toka na različitim nivoima dubine mreže
#kroz gated fusion i attention modul se formira konačna reprezentacija i fokus na oštećene regione.


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
    #priprema slika
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


def evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(model_path: str, test_dataset_dir: str, img_size: int = 128, batch_size: int = 32):
    #učitavaju se težine, primenjuje se optimalni prag; težine odredjuju koliko da se veruje nekoj info tj naučene olduke
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"Evaluacija modela na uređaju: {device}")
    print(f"Testni dataset: {test_dataset_dir}")
    print(f"{'='*60}\n")

    # priprema test podataka
    test_dataset = DamageDataset(test_dataset_dir, img_size=img_size, train=False)

    if len(test_dataset) == 0:
        print(f"nemma slika na putanji {test_dataset_dir} ili folder ne postoji")
        return None, None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Pronađeno ukupno {len(test_dataset)} slika za testiranje.")
    model = DodinaMreza(num_classes=2).to(device)

    # učitavanje sačuvanih težina
    if not os.path.exists(model_path):
        print(f"Model nije pronađen na putanji: {model_path}")
        return None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # preuzimanje sačuvanih težina i optimalnog praga
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = checkpoint.get('best_threshold', 0.5)
        print(f"Uspešno učitan checkpoint (Najbolja tačnost tokom treninga: {checkpoint.get('best_val_acc', 0.0):.2f}%)")
        print(f"Korišćeni optimalni prag (Threshold): {best_threshold:.2f}") #predstavlja matematičku granicu (broj između 0 i 1) koja odlučuje da li će model neku sliku proglasiti oštećenom (Klasa 1) ili neoštećenom (Klasa 0).
    else:
        model.load_state_dict(checkpoint)
        best_threshold = 0.5
        print("učitane težine modela. Koristi se podrazumevani prag: 0.5")

    model.eval()

    all_preds = []
    all_labels = []
    all_paths = []

#ošrećenja
  damage_mapping = {
        'apply_anisotropic_diffusion': 'Vlaga i gubitak detalja',
        'apply_mold_and_decay': 'Buđ i biološka degradacija',
        'apply_chemical_aging': 'Hemijsko starenje i žutilo',
        'apply_fft_lpf': 'Gubitak oštrine (FFT LPF)',
        'apply_cracks': 'Pukotine na platnu',
        'apply_water_stains': 'Vodene mrlje (Coffee-ring)',
        'apply_paint_flaking': 'Ljuštenje boje',
        'apply_combined_damage': 'Kombinovano oštećenje'
    }

    stats = {name: {'total': 0, 'correct': 0} for name in damage_mapping.values()}
    stats['Bez oštećenja (Čiste slike)'] = {'total': 0, 'correct': 0}

    # Prolazak kroz dataset (ovde ide 3way tta gde se modelu 3 puta prikaze slika)  tj model vidi priginalnu sliku i 2 puta okrenututu)
  #ukupna verovatnoca je srednja vrednost za ta tri da bi rezultat bio bolji jej
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            # 1. Originalna predikcija
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

            # ovde je 3way tta dakle prethodna dva deljena sa 3
            probs_final = (probs_orig + probs_flipped_h + probs_flipped_v) / 3.0

            # Predikcija primenom optimalnog praga
            preds = (probs_final[:, 1] >= best_threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # razvrstavanje tačnosti po klasama oštećenja
    for pred, label, path in zip(all_preds, all_labels, all_paths):
        if label == 0:
            stats['Bez oštećenja (Čiste slike)']['total'] += 1
            if pred == 0:
                stats['Bez oštećenja (Čiste slike)']['correct'] += 1
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
    print(f"REZULTATI EVALUACIJE MODELA")
    print(f"Ukupna tačnost modela (Accuracy): {ukupna_tacnost:.2f}%")
    print("="*50 + "\n")
    print("Klasifikacija po klasama:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['Klasa 0 (Bez oštećenja)', 'Klasa 1 (Oštećeno)'],
        digits=4
    )
    print(report)

    rows = []
    for cat, data in stats.items():
        total = data['total']
        correct = data['correct']
        acc = (correct / total * 100) if total > 0 else 0.0
        rows.append([cat, total, correct, round(acc, 2)])

    df_stats = pd.DataFrame(rows, columns=["Tip oštećenja", "Broj testiranih", "Broj tačnih", "Tačnost (%)"])

    PINK = "\033[38;5;205m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    top_border = f"{PINK}┌──────────────────────────────────┬────────────┬────────────┬──────────────┐{RESET}"
    mid_border = f"{PINK}├──────────────────────────────────┼────────────┼────────────┼──────────────┤{RESET}"
    bot_border = f"{PINK}└──────────────────────────────────┴────────────┴────────────┴──────────────┘{RESET}"

    print(f"\n{BOLD}DETALJAN PREGLED PO TIPOVIMA OŠTEĆENJA{RESET}")
    print(top_border)
    print(f"{PINK}│{RESET} {BOLD}{'TIP OŠTEĆENJA / KATEGORIJA':<32} {PINK}│{RESET} {BOLD}{'TESTIRANO':<10} {PINK}│{RESET} {BOLD}{'TAČNO':<10} {PINK}│{RESET} {BOLD}{'TAČNOST (%)':<12} {PINK}│{RESET}")
    print(mid_border)
    
    for row in rows:
        cat_name = row[0]
        tested = row[1]
        correct = row[2]
        accuracy_val = f"{row[3]:.2f}%"
        print(f"{PINK}│{RESET} {cat_name:<32} {PINK}│{RESET} {tested:<10d} {PINK}│{RESET} {correct:<10d} {PINK}│{RESET} {accuracy_val:<12} {PINK}│{RESET}")
            
    print(bot_border)

    results_dir = os.path.dirname(model_path)
    
    # čuvanje tabele skoja prikazuje tačnost modela na različitim oštećenjima kojih ima 8
    csv_path = os.path.join(results_dir, "rezultati_po_tipovima_ostecenja.csv")
    df_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nTabela rezultata uspešno:\n{csv_path}")

    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"klasifikacije :\n{report_path}")

    # izračunavanje i čuvanje osnovnih metrika
    accuracy = accuracy_score(all_labels, all_preds) #tačnost tj koliko je ukupno pogodio
    precision = precision_score(all_labels, all_preds) #koliko su poz. predikcije tačen
    recall = recall_score(all_labels, all_preds) #koliko je stvarnih oštećenja našao
    f1 = f1_score(all_labels, all_preds) #balans između precision i recall

    metrics_path = os.path.join(results_dir, "osnovne_metrike.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"f1-score : {f1:.4f}\n")
    print(f" metrike sačuvane u:\n{metrics_path}")

    # crtanje matrice konfuzije i onda njeno cuvanje da mogu da je gledam kasnije
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',  
                xticklabels=['Bez oštećenja', 'Oštećeno'],
                yticklabels=['Bez oštećenja', 'Oštećeno'])
    plt.xlabel('Predviđeno (Šta je model rekao)')
    plt.ylabel('Stvarno (Tačna oznaka)')
    plt.title('Matrica konfuzije (Dodina Mreža)')
        cm_path = os.path.join(results_dir, "matrica_konfuzije.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"matrica konfuzije sačuvana na:\n{cm_path}\n")
    
    plt.show()

    return all_labels, all_preds

# start
if __name__ == '__main__':
    putanja_do_modela = "/content/drive/MyDrive/Projekat_Model/dodinamreza_best.pth"
    putanja_do_test_dataseta = "/content/DATASET_TEST"

    stvarne_oznake, predvidjanja = evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(
        model_path=putanja_do_modela,
        test_dataset_dir=putanja_do_test_dataseta,
        img_size=128,
        batch_size=32
    )
