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

# služi da zameni običnu konvoluciju efikasnijom konvolucijom koja koristi manje parametara i da brze i radi

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))

#ovde je arhitektura modela
# ista konvolucija se primenjuje više puta nad istim ulazom kako bi se postepeno izdvojile i primetile sitni detalji
#vilj je kao kako napraviti program koji duboko i detaljno analizira podatke, a da pritom ne zauzme previše memorijice
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

#  ovde se dakle slika prebacuje u frekvencijski domen i gledaju se visoke i niske frekvencije i tkao se odrejduje gde su detlaji sitni i gde su neke velike povrsine je
#: deli sliku na niske i visoke frekvencije (grube površine i sitne detalje) i onda ih obrađuje odvojeno.
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

#ovdde je rad sa raspodelom piksela
#gleda se slika kao prostor i gleda se gde ima neko odstupanje
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

#komunikacija i prevdo za ihz frekv u spatial i obrnuto da bi se kasnije moglo porcentii koja odluka i analiza ima veći znacaj
#prevodjenje iz prostornog u frekvencijski i obrnuto da bi mogli da se razumeju medjusobno
#kasnije se ti podaci koriste da se u sledećem bloku odredi kojem se treba više verovati
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

#posa ovde je da uzme informacije iz oba domena (prostornog i frekvencijskog) i odluči u kom delu slike će verovati kome i u kolikom procentu
#to radi tako sto ne gleda samo piskele kao spatial nego sliku kao celinu i onda odlucuje
#Iako je odluka doneta na osnovu globalnog stanja slike, ona se primenjuje na svaki piksel
#
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
        s = self.spatial_proj(spatial)  #s procenat ako slika ima kontraste velike onda s procenat je veci
        sp = self.spectral_proj( #ako ima problem sa tipa sumom ili mutne je teksture onda se aktivira
            F.interpolate(spectral, size=spatial.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        combined = torch.cat([s, sp], dim=1)
        gates = self.gate(combined).view(combined.shape[0], -1, 1, 1)
        out = s.shape[1]
        s_gate = gates[:, :out]
        sp_gate = gates[:, out:]
        return s_gate * s + sp_gate * sp

#mapiranje anomalija
#vrsi se restauracija sao gde je lokalizovao anomaliju dok zdrave delove slike ostavlja netaknutim.
#tu mini restauraiju raid preko konstektualnog popravljanja tj gleda 3*3 piksela okolo
#Na osnovu okolnih zdravih piksela, konvolucioni filteri izračunavaju (pogađaju) šta je trebalo da bude na mestu anomalije.
#
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

#koji postepeno povećava rezoluciju slike i popravlja oštećene delove pomoću informacija iz različitih izvora
class DecoderRestorationBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(     #povecavanje reyolucije
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
        x = self.upsample(x)#Uzima sliku niske rezolucije iz dubljih slojeva i povećava je duplo (skaliranje sa 2) kako bi je vratio ka originalnoj veličini
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)#tačno govori gde se na slici nalaze oštećenja
        x = torch.cat([x, skip, dm], dim=1)   #ovde se spaja>
                                               #Uvećana sliku iz prethodnog sloja dekodera
                                               #detaljne informacije visoke rezolucije direktno iz enkodera (početka mreže) koje pomažu da slika ne bude mutna
        return self.spectral(self.dense_micro(self.conv(x)))#koristi spektral i dense micro da bi iymislila sta ce restaurirati

#ovo sluyi da mreza vidi malol vise info bez gubitka rezolucije
class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int):  #Mreža istovremeno gleda sliku kroz četiri različite "lupe"
        super().__init__()
        mid = channels // 4
        self.c1 = nn.Conv2d(channels, mid, 3, padding=1, dilation=1, bias=False)
        self.c2 = nn.Conv2d(channels, mid, 3, padding=2, dilation=2, bias=False)
        self.c3 = nn.Conv2d(channels, mid, 3, padding=4, dilation=4, bias=False)
        self.c4 = nn.Conv2d(channels, mid, 3, padding=8, dilation=8, bias=False)
        self.fusion = nn.Conv2d(channels, channels, 1, bias=False)  # Skuplja informacije iz sva četiri vidna polja i spaja ih u jedno
        self.bn = nn.GroupNorm(4, channels) #Na kraju dodaje originalni ulaz na dobijeni rezultat, što stabilizuje treniranje i sprečava da mreža zaboravi originalne detalje

    def forward(self, x: Tensor) -> Tensor:
        merged = torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x)], dim=1)
        return F.relu(self.bn(self.fusion(merged)) + x)

#kao neki filter koji propusta pametne delove
#da se ne bi slika direktno iy enkodera prenela u dekoder jer bi prenela [um onda ide ovako
class GatedSkipConnection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   #Skuplja informacije sa cele slikei svodi svaku mapu karakteristika na jedan borjic
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()  #pretrvara se u ili 0 ili1  i Ako je vrednost blizu 1, ta mapa karakteristika je važna i treba je propustiti. Ako je blizu 0, ona se blokira jer sadrži nebitne info
        )

    def forward(self, skip: Tensor) -> Tensor:
        return skip * self.gate(skip)  #Na kraju, originalne karakteristike se množe sa ovim težinama, čime se filtriraju loši podaci pre nego što stignu u vaš

#Ovaj blok ima zadatak da eksplicitno natera mrežu da se fokusira na geometriju i oštre ivice predmeta na slici
#kada se slika resturira da se ne bi unistili detalljo
class EdgeBranch(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)  #vert ivice
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0) #horiz ivice
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
        return self.conv(torch.cat([F.conv2d(x, self.kx, padding=1, groups=3), F.conv2d(x, self.ky, padding=1, groups=3)], dim=1))#Mreža spaja horizontalne i vertikalne ivice (dobija se 6 kanala) i propušta ih kroz dodatne konvolucije
#Rezultat je mapa koja jasno naglašava gde se nalaze granice objekata, što pomaže dekoderu da ponovo nacrta oštre konture tamo gde su bile uništene.


#vracanje boja i kontrasta
#vraca tako sto lokalno sredjuje detalja i globalno podešava osvetljenje i to
#popunjava msm zna koja boja treba d aide takos to gleda okollne piksle
#mreža direktno uzima "skicu" i teksturu sa delova slike koji nisu uništeni i koristi ih kao šablon za popunjavanje rupa.
class ContrastColorRecovery(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        self.local_conv = nn.Sequential(     # Propušta karakteristike kroz standardne konvolucije. Ovaj deo gleda piksel po piksel i traži male, lokalne greške u bojama i teksturi koje treba ispraviti (npr. popunjavanje sitnih detalja).
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_ch // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch // 2, out_ch, 3, padding=1)
        )
        self.global_adjust = nn.Sequential(   #sabija celu sliku u jedan jedini piksel koji predstavlja prosečnu boju i osvetljenje celokupne scene. Na osnovu toga, mreža odlučuje da li je cela slika previše tamna, svetla, ili joj fali kontrasta
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch // 4, out_ch * 2, 1),
        )

    def forward(self, x: Tensor, input_img: Tensor) -> Tensor:
        local_refinement = self.local_conv(x)
        global_stats = self.global_adjust(x)
        gain, bias = torch.chunk(global_stats, 2, dim=1)
        gain = torch.sigmoid(gain).view(x.shape[0], -1, 1, 1) * 2.0 #Ovo služi kao multiplikator kontrasta. ako je preko 1.0 kontrast se pojacava ispod 1 smanjuje se na 1.0 ne menja se)
        bias = torch.tanh(bias).view(x.shape[0], -1, 1, 1) * 0.5
        adjusted = local_refinement * gain + bias
        return torch.clamp(input_img + adjusted, 0.0, 1.0)  # uzima se originalna slika i nju samo dodaje/oduzima finu korekciju
      #takodje se saseca sve sto prelazi opseg od 0.0 i 1.0

# glavni deo modela restauracije
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

        # dinamicki blok za boje
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_channels)

    def forward(self, x: Tensor, use_ccr: bool = True) -> Tensor | dict[str, Tensor]:
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

        # CCR se primenjuje na izlaz za vrhunske boje
        out = self.contrast_color_recovery(d1_fused, input_img)

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


# skup za restauraciju
class RestorationDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int = 192, train: bool = True, preload_to_ram: bool = True):
        self.img_size = img_size
        self.train = train
        self.preload_to_ram = preload_to_ram

        folder_0 = os.path.join(dataset_dir, '0')
        folder_1 = os.path.join(dataset_dir, '1')
        if not os.path.exists(folder_0) or not os.path.exists(folder_1):
            raise FileNotFoundError(f"nema 0 i 1 foldera na putanji {dataset_dir}!")

        dmg_files = sorted([f for f in os.listdir(folder_1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        self.pairs = []

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

        self.cached_pairs = []
        if self.preload_to_ram and len(self.pairs) > 0:
            print(f"[Dataset] ucitavanje {len(self.pairs)} parova za restauraciju")
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

            # Color i kontrast augmentacije
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

        return deg_t, clean_t

    def __len__(self):
        return len(self.pairs)


# Gubici za model restauracije
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

#Upoređuje ivice rekonstruisane slike sa ivicama originala. Ako su ivice na rekonstruisanoj slici mutne ili nedostaju, ova greška raste
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

#Upoređuje isključivo te zamućene mape boja. Time osigurava da restaurirani deo slike ima savršeno pogođen ton i da se ne razlikuje od okoline (npr. da trava bude iste nijanse zelene)
class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_blur = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        target_blur = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        return F.l1_loss(pred_blur, target_blur)

# Mreža mora da pogodi i krupne oblike i najsitnije teksture na slici da bi ukupna greška bila mala.
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

#ikseli gde mreža već dobro radi dobijaju malu težinu, a delovi gde žestoko greši dobijaju veliku težinu. Mreža se tako fokusira na popravljanje najtežih grešaka.
class SoftHardExampleMiningLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        error_map = torch.abs(pred - target).mean(dim=1, keepdim=True).detach()
        return 1.0 + torch.sigmoid((error_map - error_map.mean()) / (error_map.std() + 1e-6))

#Daje veću važnost oštrim detaljima (težina 0.6) nego glatkim površinama (težina 0.4), sprečavajući da slika ispadne zamućena.
class FrequencyConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_low = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        tgt_low = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        pred_high = pred - pred_low
        tgt_high = target - tgt_low
        return 0.4 * F.l1_loss(pred_low, tgt_low) + 0.6 * F.l1_loss(pred_high, tgt_high)

#konacni spoj svih funkcija gresaka
#racuna osnovnu razliku u piskelima
#brzu Furijeovu transformaciju (torch.fft.rfft2) da prebaci sliku u frekvencije. Upoređuje realni i imaginarni deo kako bi osigurala da slika nema digitalnih anomalija i artefakata (periodičnih šuma/linija).
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


# Trening funkcija
def train_restauracija(
    dataset_dir: str,
    epochs_to_train: int = 15,
    batch_size: int = 4,
    lr: float = 2e-4,
    img_size: int = 192,
    save_dir: str = '.',
    resume: bool = True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = RestorationDataset(dataset_dir, img_size=img_size, train=True, preload_to_ram=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    eval_indices = np.arange(min(100, len(train_dataset)))
    eval_subset = Subset(train_dataset, eval_indices)
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Restauracija(base_ch=32).to(device)
    criterion = RestauracijaLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train, eta_min=1e-6)

    start_epoch = 0
    best_psnr = 0.0

    checkpoint_path = os.path.join(save_dir, 'dodinarestauracijajej.pth')
    best_checkpoint_path = os.path.join(save_dir, 'dodinarestauracijabest.pth')

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        best_psnr = checkpoint.get('best_psnr', 0.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train, eta_min=1e-6)
        print(f"nastavak od epohe br {start_epoch+1}. najbolji rezultat za psnr {best_psnr:.2f} dB")
    else:
        print("nema mmodela")

    accumulation_steps = 3
    optimizer.zero_grad()

    for epoch in range(start_epoch, start_epoch + epochs_to_train):
        model.train()
        running_loss = 0.0

        for batch_idx, (deg, clean) in enumerate(train_loader):
            deg, clean = deg.to(device), clean.to(device)

            outputs = model(deg)
            loss = criterion(outputs, clean)

            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        epoch_loss = running_loss / len(train_loader)

        model.eval()
        total_psnr = 0.0
        num_batches = 0

        with torch.no_grad():
            for deg, clean in eval_loader:
                deg, clean = deg.to(device), clean.to(device)

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

        print(f"restauracija | Epoha {epoch+1:02d} | Loss: {epoch_loss:.5f} | Val PSNR: {avg_psnr:.2f} dB")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_psnr': best_psnr,
        }, checkpoint_path)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_psnr': best_psnr,
            }, best_checkpoint_path)
            print(f"  ★ novi best {best_checkpoint_path} (PSNR: {best_psnr:.2f} dB)")

    return model


# Glavni ulaz
if __name__ == '__main__':
    save_dir = "/content/drive/MyDrive/Projekat_Model"
    restauracija_path = os.path.join(save_dir, "dodinarestauracijajej.pth")

    drive_trening_zip = "/content/drive/MyDrive/Projekat_Model/trening.zip"
    lokalni_trening_path = "/content/trening"

    if not os.path.exists(lokalni_trening_path):
        if os.path.exists(drive_trening_zip):
            print(f"otpakuje se  {drive_trening_zip} u {lokalni_trening_path}...")
            get_ipython().system(f'unzip -q "{drive_trening_zip}" -d "{lokalni_trening_path}"')
            print("gotojooo")
        elif os.path.exists("/content/drive/MyDrive/Projekat_Model/trening"):
            lokalni_trening_path = "/content/drive/MyDrive/Projekat_Model/trening"

    print(f"\nkrece trening {restauracija_path}...")
    if os.path.exists(lokalni_trening_path):
        restoracioni_model = train_restauracija(
            dataset_dir=lokalni_trening_path,
            epochs_to_train=50,
            batch_size=4,
            lr=2e-4,
            img_size=192,
            save_dir=save_dir,
            resume=True
        )
    else:
        print(f"nema dataseta na {lokalni_trening_path}")
