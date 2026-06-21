#ovo je prva arhitektura za klasifikaciju da li je oštećenja slika ili ne
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
import warnings
from sklearn.metrics import f1_score  

warnings.filterwarnings('ignore')

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
        out_ch = s.shape[1]
        s_gate = gates[:, :out_ch]
        sp_gate = gates[:, out_ch:]
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

class DodinaMrezaLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.05, weight_damaged: float = 1.35):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight_damaged = weight_damaged

    def forward(self, outputs: dict[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
        device = outputs['logits'].device
        weights = torch.tensor([1.0, self.weight_damaged], device=device)
        
        cls_loss = F.cross_entropy(outputs['logits'], labels, weight=weights)

        damage_map = outputs['damage_map']
        targets_map = labels.view(-1, 1, 1, 1).expand_as(damage_map).float()
        map_loss = F.mse_loss(damage_map, targets_map)

        aux_damage = outputs['aux_damage']
        targets_aux = labels.view(-1, 1, 1, 1).expand_as(aux_damage).float()
        aux_loss = F.binary_cross_entropy(aux_damage, targets_aux)

        total = self.alpha * cls_loss + self.beta * map_loss + self.gamma * aux_loss
        return {
            'total': total,
            'cls_loss': cls_loss,
            'map_loss': map_loss,
            'aux_loss': aux_loss
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
        return img, label


def train_dodina_mreza(
    dataset_dir: str,
    epochs_to_train: int = 15,  
    batch_size: int = 32,
    lr: float = 7e-5,          
    img_size: int = 128,
    val_split: float = 0.2,
    save_dir: str = '.',
    resume: bool = True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = DamageDataset(dataset_dir, img_size=img_size, train=True)
    val_dataset = DamageDataset(dataset_dir, img_size=img_size, train=False)

    if len(train_dataset) == 0:
        return None, None

    unique_images = []
    for x, _ in train_dataset.samples:
        parts = os.path.basename(x).split('_')
        if len(parts) >= 2:
            unique_images.append(f"{parts[0]}_{parts[1]}")
        else:
            unique_images.append(os.path.basename(x))

    unique_images = sorted(list(set(unique_images)))

    np.random.seed(42)
    np.random.shuffle(unique_images)

    val_img_count = int(len(unique_images) * val_split)
    val_img_names = set(unique_images[:val_img_count])

    train_indices = []
    val_indices = []

    for idx, (path, _) in enumerate(train_dataset.samples):
        parts = os.path.basename(path).split('_')
        img_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else os.path.basename(path)
        if img_id in val_img_names:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = DodinaMreza(num_classes=2).to(device)
    criterion = DodinaMrezaLoss(alpha=1.0, beta=0.1, gamma=0.05, weight_damaged=1.35)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    start_epoch = 0
    best_val_acc = 0.0
    best_threshold = 0.5
    checkpoint_path = os.path.join(save_dir, 'dodinamreza_best.pth')

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        start_epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_threshold = checkpoint.get('best_threshold', 0.5)

    total_epochs = start_epoch + epochs_to_train
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train)

    if resume and os.path.exists(checkpoint_path):
        print(f"\Pronađen sačuvan model na putanji: {checkpoint_path}")

        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        for g in optimizer.param_groups:
            g['lr'] = lr

        print(f"Nastavljamo od epohe {start_epoch + 1} do epohe {total_epochs} (dodatnih {epochs_to_train} epoha). Best Acc: {best_val_acc:.2f}% | Prethodni Prag: {best_threshold:.2f}")
    else:
        print(f"\nema sač modela")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"{'='*60}\n")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

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
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                losses = criterion(outputs, labels)
                val_loss += losses['total'].item()
                
                probs_orig = F.softmax(outputs['logits'], dim=-1)

                images_flipped_h = torch.flip(images, dims=[3])
                outputs_flipped_h = model(images_flipped_h)
                probs_flipped_h = F.softmax(outputs_flipped_h['logits'], dim=-1)

                images_flipped_v = torch.flip(images, dims=[2])
                outputs_flipped_v = model(images_flipped_v)
                probs_flipped_v = F.softmax(outputs_flipped_v['logits'], dim=-1)

                probs_final = (probs_orig + probs_flipped_h + probs_flipped_v) / 3.0

                val_probs.extend(probs_final[:, 1].cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        
        val_probs_arr = np.array(val_probs)
        val_labels_arr = np.array(val_labels_list)
        best_t_epoch = 0.5
        best_f1_epoch = 0.0

        for t in np.arange(0.1, 0.9, 0.01):
            preds = (val_probs_arr >= t).astype(int)
            f1 = f1_score(val_labels_arr, preds)
            if f1 > best_f1_epoch:
                best_f1_epoch = f1
                best_t_epoch = t

        val_preds_opt = (val_probs_arr >= best_t_epoch).astype(int)
        val_correct = np.sum(val_preds_opt == val_labels_arr)
        val_total = len(val_labels_arr)
        val_acc = 100.0 * val_correct / val_total
        
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_threshold = best_t_epoch
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_threshold': best_threshold, # 
            }, os.path.join(save_dir, 'dodinamreza_best.pth'))
            marker = f" ★ BEST (Optimalni T za F1 na {best_t_epoch:.2f} daje F1: {best_f1_epoch:.4f})"
        else:
            marker = f" (Optimalni T za F1 na {best_t_epoch:.2f} daje F1: {best_f1_epoch:.4f})"

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - start

        print(f"Epoch {epoch+1:3d}/{total_epochs} | "
              f"Train Loss: {train_loss:.4f} / Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} / Acc: {val_acc:.1f}% (3-way TTA!) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"{elapsed:.1f}s{marker}")

    print(f"\n{'='*60}")
    print(f"gotojo")
    print(f"{'='*60}")

    return model, history

if __name__ == '__main__':

    drive_dataset_path = "/content/drive/MyDrive/Projekat_Model/DATASET_TRENING"
    drive_zip_path = "/content/drive/MyDrive/Projekat_Model/DATASET_TRENING.zip"
    local_dataset_path = "/content/DATASET_TRENING"

    from google.colab import drive
    if not os.path.exists("/content/drive"):
        drive.mount('/content/drive')

    if not os.path.exists(local_dataset_path):
        print("pripremam dataset")

        if os.path.exists(drive_zip_path):
            print("nadjen zip i otpakujem")
            get_ipython().system(f'unzip -q "{drive_zip_path}" -d "/content/"')
            print("otpakivanje uspešno završeno")
        elif os.path.exists(drive_dataset_path):
            print("Nema .zip fajla")
            get_ipython().system(f'cp -r "{drive_dataset_path}" "/content/"')
            print("Kopiranje done")
        else:
            print("nema dataseta")
    else:
        print("dataset u lokalnom direktorijumu /content/.")

    model, history = train_dodina_mreza(
        dataset_dir=local_dataset_path,
        epochs_to_train=15, 
        batch_size=32,
        lr=7e-5,            
        img_size=128,
        val_split=0.2,
        save_dir="/content/drive/MyDrive/Projekat_Model",
        resume=True,
    )
