#ovo je prva arhitektura za klasifikaciju da li je oštećenja slika ili ne
#kod ove bi trebalo da je unapredjen deo sa detekciijom ostecenja preko citave slike
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
import warnings
from sklearn.metrics import f1_score  

warnings.filterwarnings('ignore')

class RecursiveDenseMicroBlock(nn.Module):
    """
    Mikro-blok koji rekurzivno primenjuje istu konvoluciju više puta.
    Rezultati svih rekurzija se spajaju (konkateniraju) po kanalima 
    i na kraju redukuju 1x1 konvolucijom na početni broj kanala.
    """
    def __init__(self, channels: int, num_recursions: int = 3):
        super().__init__()
        self.num_recursions = num_recursions
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.fusion = nn.Conv2d(channels * num_recursions, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        out = x
        # Prolazak kroz definisan broj rekurzija uz dodavanje rezidulanog (skip) spoja
        for i in range(self.num_recursions):
            out = F.relu(self.bn(self.conv(out)) + x)
            outputs.append(out)
        # Spajanje svih međufaza i njihovo fuzisanje 1x1 konvolucijom
        merged = torch.cat(outputs, dim=1)
        return self.fusion(merged)

class SpectralDecomposeBlock(nn.Module):
    """
    UNAPREĐENI Blok za spektralnu dekompoziciju.
    Koristi višeskalni piramidalni pooling za hvatanje blagih gradijenata (žutilo, mrlje)
    i finu prostorno-kanalnu kapiju (Sigmoid Gate) za lokalizovano mešanje frekvencija.
    """
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
        
        # Fina prostorno-kanalna kapija (pixel-by-pixel, channel-by-channel)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        # Višeskalna dekompozicija (Piramidalno niskofrekventno filtriranje)
        # Ovo omogućava prepoznavanje mrlja i zamućenja različitih promera
        low2 = F.interpolate(
            F.avg_pool2d(x, kernel_size=2, stride=2),
            size=x.shape[2:], mode='bilinear', align_corners=False
        )
        low4 = F.interpolate(
            F.avg_pool2d(x, kernel_size=4, stride=4),
            size=x.shape[2:], mode='bilinear', align_corners=False
        )
        low8 = F.interpolate(
            F.avg_pool2d(x, kernel_size=8, stride=8),
            size=x.shape[2:], mode='bilinear', align_corners=False
        )
        
        # Srednja vrednost različitih frekvencijskih opsega blagih prelaza
        low = (low2 + low4 + low8) / 3.0
        high = x - low
        
        low_feat = self.low_conv(low)
        high_feat = self.high_conv(high)
        
        # Izračunavanje 2D kapije za lokalizovanu fuziju frekvencija
        concat = torch.cat([low_feat, high_feat], dim=1)
        g = self.gate(concat)
        
        # Dinamičko mešanje frekvencija
        fused = g * low_feat + (1 - g) * high_feat
        return self.fuse(torch.cat([fused, x], dim=1))

class SpatialBlock(nn.Module):
    """
    Spatijalni blok zadužen za ekstrakciju prostornih karakteristika.
    Sastoji se od standardne konvolucije, rekurzivnog mikro-bloka i Max Pooling-a za smanjenje rezolucije.
    Vraća procesirane podatke nakon pooling-a, ali i skip konekciju pre pooling-a.
    """
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
        return pooled, x # Vraća rezoluciju smanjenu za 2x i mapu originalne rezolucije (za skip konekciju)

class AsymmetricCrossBridge(nn.Module):
    """
    Asimetrični most za unakrsnu razmenu informacija između prostornog (Spatial) i spektralnog dela mreže.
    Usklađuje dimenzije (širinu, visinu i broj kanala) i vrši fuziju informacija iz obe grane.
    """
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
        # Prenos informacija iz prostorne u spektralnu granu uz prilagođavanje veličine (pooling)
        spectral_enhanced = spectral_feat + self.spatial_to_spectral(
            F.adaptive_avg_pool2d(spatial_feat, spectral_feat.shape[2:])
        )
        # Prenos informacija iz spektralne u prostornu granu uz prilagođavanje veličine (interpolacija)
        spatial_enhanced = spatial_feat + self.spectral_to_spatial(
            F.interpolate(spectral_feat, size=spatial_feat.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        # POPRAVLJENO: Pravilno pronalaženje minimalnih dimenzija pre spajanja
        min_h = min(spatial_feat.shape[2], spectral_feat.shape[2])
        min_w = min(spatial_feat.shape[3], spectral_feat.shape[3])
        s_pooled = F.adaptive_avg_pool2d(spatial_enhanced, (min_h, min_w))
        sp_pooled = F.adaptive_avg_pool2d(spectral_enhanced, (min_h, min_w))
        return self.fuse(torch.cat([s_pooled, sp_pooled], dim=1))

class GatedFusionBlock(nn.Module):
    """
    Blok za fuziju sa kapijom (Gate) koji spaja finalne prostorne i spektralne karakteristike.
    Koristi potpuno povezani (Linear) sloj i Sigmoid aktivaciju da generiše težinske mape
    kojima se selektivno propuštaju najbitniji signali iz obe grane.
    """
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
        # Spajanje i računanje pažnje (atention/gate koeficijenata) za oba ulaza
        combined = torch.cat([s, sp], dim=1)
        gates = self.gate(combined).view(combined.shape[0], -1, 1, 1)
        
        out_ch = s.shape[1]
        s_gate = gates[:, :out_ch]   # Kapija za prostorne podatke
        sp_gate = gates[:, out_ch:]  # Kapija za spektralne podatke
        return s_gate * s + sp_gate * sp

class DamageAttentionModule(nn.Module):
    """
    Modul pažnje fokusiran na regije oštećenja (Damage Attention).
    Generiše jednokanalnu mapu pažnje (vrednosti 0-1) i množi je sa ulaznim podacima
    kako bi mreža naglasila mesta gde detektuje anomalije/oštećenja.
    """
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
        refined = self.refine(attended) + x # Rezidulana veza (dodavanje originalnog ulaza)
        return refined, attn_map

class DodinaMreza(nn.Module):
    #Glavna arhitektura neuronske mreže (DodinaMreza).
    #paralelno se obrađuju slike kroz prostorne (Spatial) i spektralne (Spectral) domene
    #karakteristike se razmenjuju preko CrossBridge-a tj tu kao kouniciraju, a spajaju preko GatedFusion bloka,
    #propuštaju kroz modul pažnje i na kraju klasifikuju u klase (neoštećeno / oštećeno).
    #Takođe ima i pomoćni izlaz (aux_damage) za lokalizaciju anomalija.

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()

        # Prostorna grana (Spatial backbone)
        self.spatial_block1 = SpatialBlock(in_channels, 64)
        self.spatial_block2 = SpatialBlock(64, 128)
        self.spatial_block3 = SpatialBlock(128, 256)

        # Spektralna grana (Spectral backbone)
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

        # Mostovi za povezivanje i razmenu informacija između grana na različitim nivoima rezolucije
        self.cross1 = AsymmetricCrossBridge(64, 64, 64)
        self.cross2 = AsymmetricCrossBridge(128, 128, 128)
        self.cross3 = AsymmetricCrossBridge(256, 256, 256)

        # Finalna fuzija i modul pažnje
        self.gated_fusion = GatedFusionBlock(256, 256, 512)
        self.damage_attention = DamageAttentionModule(512)

        # Klasifikaciona glava (potpuno povezani slojevi sa Dropout-om za regularizaciju)
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
        #generisanje mape oštećenja
        self.damage_map_head = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # Prolaz kroz prostornu granu (uzimaju se i preskočene konekcije)
        s1, s1_skip = self.spatial_block1(x)
        s2, s2_skip = self.spatial_block2(s1)  
        s3, s3_skip = self.spatial_block3(s2)

        # Prolaz kroz spektralnu granu uz promene rezolucije i projekcije kanala
        sp0 = self.spectral_init(x)
        sp1 = self.spectral_block1(sp0)
        sp1_p = self.spec_proj1(self.spectral_pool1(sp1))
        sp2 = self.spectral_block2(sp1_p)
        sp2_p = self.spec_proj2(self.spectral_pool2(sp2))
        sp3 = self.spectral_block3(sp2_p)

        # Unakrsno spajanje karakteristika preko asimetričnih mostova
        c1 = self.cross1(s1_skip, sp1)
        c2 = self.cross2(s2_skip, sp2)        
        c3 = self.cross3(s3_skip, sp3)

        # Obogaćivanje prostorne mape sa informacijama iz trećeg mosta
        s3_enriched = s3 + F.adaptive_avg_pool2d(c3, s3.shape[2:])

        # Finalna fuzija prostornih i spektralnih osobina
        fused = self.gated_fusion(s3_enriched, sp3)
        # Primena Damage Attention modula
        attended, damage_map = self.damage_attention(fused)

        # Generisanje izlaza: logiti klase i pomoćne mape oštećenja
        logits = self.classifier(attended)
        aux_damage = self.damage_map_head(attended)

        return {
            'logits': logits,
            'damage_map': damage_map,
            'aux_damage': aux_damage
        }

class DodinaMrezaLoss(nn.Module):
    """
    Kompozitna funkcija gubitka (Loss function) za treniranje mreže.
    Kombinuje tri gubitka:
    1. CrossEntropyLoss za klasifikaciju (uz težinski koeficijent za balansiranje klasa).
    2. MSELoss za mapu pažnje (damage_map).
    3. BinaryCrossEntropyLoss za pomoćnu mapu oštećenja (aux_damage).
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.05, weight_damaged: float = 1.35):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight_damaged = weight_damaged

    def forward(self, outputs: dict[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
        device = outputs['logits'].device
        # Dodavanje veće težine oštećenoj klasi (indeks 1) zbog potencijalnog debalansa u podacima
        weights = torch.tensor([1.0, self.weight_damaged], device=device)
        
        # 1. Gubitak klasifikacije
        cls_loss = F.cross_entropy(outputs['logits'], labels, weight=weights)

        # 2. Gubitak za mapu pažnje (povezuje se sa labelom proširenom na dimenzije mape)
        damage_map = outputs['damage_map']
        targets_map = labels.view(-1, 1, 1, 1).expand_as(damage_map).float()
        map_loss = F.mse_loss(damage_map, targets_map)

        # 3. Pomoćni gubitak detekcije oštećenja
        aux_damage = outputs['aux_damage']
        targets_aux = labels.view(-1, 1, 1, 1).expand_as(aux_damage).float()
        aux_loss = F.binary_cross_entropy(aux_damage, targets_aux)

        # Ukupni gubitak je ponderisana suma sva tri pojedinačna gubitka
        total = self.alpha * cls_loss + self.beta * map_loss + self.gamma * aux_loss
        return {
            'total': total,
            'cls_loss': cls_loss,
            'map_loss': map_loss,
            'aux_loss': aux_loss
        }

class DamageDataset(Dataset):
    """
    Custom Dataset klasa za učitavanje slika oštećenja.
    Očekuje strukturu direktorijuma gde podfolder '0' označava neoštećene, a '1' oštećene slike.
    Primenjuje augmentaciju podataka (okretanje, rotacija, promena boja) ako je train=True.
    """
    def __init__(self, dataset_dir: str, img_size: int = 128, train: bool = True):
        # Za ovu funkciju, pošto nema validacije, uvek koristimo augmentacije
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
        
        self.samples = []
        # Učitavanje putanja i labela (0 ili 1) iz foldera
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
        return img, label


def train_dodina_mreza(
    dataset_dir: str,
    epochs_to_train: int = 10,  
    batch_size: int = 32,
    lr: float = 7e-5,          
    img_size: int = 128,
    save_dir: str = '.',
    resume: bool = True,
):
    """
    Pojednostavljena funkcija samo za treniranje mreže "DodinaMreza".
    Nema validacije, samo trening petlja i čuvanje modela na kraju svake epohe.
    """
    # Selekcija uređaja za treniranje (Grafička karta ili Procesor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inicijalizacija trening skupa podataka
    train_dataset = DamageDataset(dataset_dir, img_size=img_size, train=True)
    
    if len(train_dataset) == 0:
        print("❌ Dataset je prazan ili nije pronađen.")
        return None, None

    # Kreiranje DataLoader-a za ceo trening set
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    # Inicijalizacija modela, funkcije gubitka i AdamW optimizatora
    model = DodinaMreza(num_classes=2).to(device)
    criterion = DodinaMrezaLoss(alpha=1.0, beta=0.1, gamma=0.05, weight_damaged=1.35)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, 'dodinamreza_best.pth')

    # Logika za nastavak treniranja ukoliko već postoji sačuvan checkpoint (.pth fajl)
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Podešeno za nastavak od prethodno sačuvane epohe
        start_epoch = checkpoint.get('epoch', 0)
        
    total_epochs = start_epoch + epochs_to_train
    # Smanjivanje learning rate-a po kosinusnoj krivoj tokom epoha
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train)

    if resume and os.path.exists(checkpoint_path):
        print(f"\nPronađen sačuvan model na putanji: {checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        for g in optimizer.param_groups:
            g['lr'] = lr

        print(f"Nastavljamo od epohe {start_epoch + 1} do epohe {total_epochs} (dodatnih {epochs_to_train} epoha).")
    else:
        print(f"\nNema sačuvanog modela, počinjem trening od početka.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"{'='*60}\n")

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(start_epoch, total_epochs):
        start = time.time()

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            losses = criterion(outputs, labels)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += losses['total'].item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        scheduler.step()

        # Čuvanje modela na kraju svake epohe
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path) # Uvek se čuva pod istim imenom, poslednja epoha je sačuvana

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        elapsed = time.time() - start

        print(f"Epoch {epoch+1:3d}/{total_epochs} | "
              f"Train Loss: {train_loss:.4f} / Acc: {train_acc:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"{elapsed:.1f}s | Model sačuvan.")

    print(f"\n{'='*60}")
    print(f"Trening završen!")
    print(f"{'='*60}")

    return model, history

#ovo je sve za nameštanje google drive i putanje
if __name__ == '__main__':
    drive_dataset_path = "/content/drive/MyDrive/Projekat_Model/DATASET_TRENING"
    drive_zip_path = "/content/drive/MyDrive/Projekat_Model/DATASET_TRENING.zip"
    local_dataset_path = "/content/DATASET_TRENING"
    
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive"):
            drive.mount('/content/drive')
    except ImportError:
        print("Nije u Google Colab okruženju, preskačem mount-ovanje Drive-a.")

    if not os.path.exists(local_dataset_path):
        print("Pripremam dataset...")
        if os.path.exists(drive_zip_path):
            print("Pronađen .zip, otpakujem...")
            os.system(f'unzip -q "{drive_zip_path}" -d "/content/"')
            print("Otpakivanje uspešno završeno.")
        elif os.path.exists(drive_dataset_path):
            print("Nema .zip fajla, kopiram ceo folder (može potrajati)...")
            os.system(f'cp -r "{drive_dataset_path}" "/content/"')
            print("Kopiranje završeno.")
        else:
            print("Greška: Nema dataseta ni u zip arhivi ni kao folder na specificiranoj putanji.")
    else:
        print("Dataset je već prisutan u lokalnom direktorijumu /content/.")

    #ovde je start (Postavljeno na 10 dodatnih epoha, od 103. do 112. epohe)
    model, history = train_dodina_mreza(
        dataset_dir=local_dataset_path,
        epochs_to_train=20, #ovo da se dobro istrenira 
        batch_size=32,
        lr=7e-5,            
        img_size=128,
        save_dir="/content/drive/MyDrive/Projekat_Model",
        resume=False, # Postavljeno na False jer imamo novu arhitekturu spektralnog bloka!
    )
