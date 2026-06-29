import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm

def apply_anisotropic_diffusion(img, severity=0.5):
    # simulacija vlage + blagog zamućenja bez prejakog zatamnjenja

    steps = int(15 + severity * 60)   # manje agresivno nego pre
    b = 0.15 + (1 - severity) * 0.12  # jača difuzija = mekše promene
    lam = 0.12                        # malo stabilnije od 0.15

    img_f = img.astype(np.float32) / 255.0
    im = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)

    for t in range(steps):
        im_new = im.copy()

        dn = im[:-2, 1:-1] - im[1:-1, 1:-1]
        ds = im[2:, 1:-1] - im[1:-1, 1:-1]
        de = im[1:-1, 2:] - im[1:-1, 1:-1]
        dw = im[1:-1, :-2] - im[1:-1, 1:-1]

        diffusion = (
            np.exp(-1 * (dn ** 2) / (b ** 2)) * dn +
            np.exp(-1 * (ds ** 2) / (b ** 2)) * ds +
            np.exp(-1 * (de ** 2) / (b ** 2)) * de +
            np.exp(-1 * (dw ** 2) / (b ** 2)) * dw
        )

        im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + lam * diffusion

        # 🔥 ključ FIX: sprečiti “crnjenje”
        im_new = np.clip(im_new, 0.05, 0.95)

        im = im_new

    res = (im * 255).astype(np.uint8)

    # 🔥 dodatni realističan blur (umesto tamnjenja)
    blur_strength = 1 + severity * 2.5
    res = cv2.GaussianBlur(res, (0, 0), blur_strength)

    # 🔥 blago vraćanje kontrasta (da ne bude “mrtvo” sivo)
    res = cv2.normalize(res, None, 20, 235, cv2.NORM_MINMAX)

    return cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

def apply_mold_and_decay(img, severity=0.5):
    # budj koja prati putanju vlage (lokalne zelene fleke, ne globalno)

    h, w, _ = img.shape
    img_f = img.astype(np.float32) / 255.0

    # multi-scale moisture / humidity map
    layers = []
    for scale in [8, 16, 32]:
        noise = cv2.resize(
            np.random.rand(scale, scale).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )
        noise = cv2.GaussianBlur(noise, (0, 0), 6)
        layers.append(noise)

    moisture_map = (
        0.5 * layers[0] +
        0.3 * layers[1] +
        0.2 * layers[2]
    )

    # sigmoid → gde može da se razvije budj
    threshold = 0.72 - severity * 0.18
    f_map = 1 / (1 + np.exp(-12 * (moisture_map - threshold)))

    mold_core = np.zeros((h, w), np.float32)

    seeds = np.where(f_map > 0.8)
    if len(seeds[0]) > 0:
        num_seeds = min(int(15 + severity * 40), len(seeds[0]))
        idx = np.random.choice(len(seeds[0]), num_seeds, replace=False)
        mold_core[seeds[0][idx], seeds[1][idx]] = 1.0

    # spread kernel (organski rast)
    kernel = np.array([
        [0.1, 0.2, 0.1],
        [0.2, 0.0, 0.2],
        [0.1, 0.2, 0.1]
    ], dtype=np.float32)

    for _ in range(int(25 + severity * 60)):
        spread = cv2.filter2D(mold_core, -1, kernel)
        mold_core = mold_core * 0.82 + spread * (f_map + 0.25)

    mold_core = np.clip(mold_core * 3.0, 0, 1)

    # 🔥 real mold color (greenish organic patches)
    mold_color = np.array([0.02, 0.10, 0.04])

    result = img_f.copy()
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - mold_core * 0.85)
            + mold_color[c] * mold_core * 0.85
        )

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def apply_chemical_aging(img, severity=0.5):
    # realistična hemijska degradacija (oksidacija + fleke + starenje papira)

    h, w, _ = img.shape
    img_f = img.astype(np.float32) / 255.0

    lab = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    Y, X = np.ogrid[:h, :w]

    # =========================
    # 📜 paper aging field (edge + random oxidation)
    # =========================
    edge_decay = np.minimum(
        np.minimum(X / w, (w - X) / w),
        np.minimum(Y / h, (h - Y) / h)
    )

    edge_decay = 1 - edge_decay  # više na ivicama

    noise = cv2.resize(np.random.rand(14, 14).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), 7)

    aging_field = edge_decay * (0.6 + 0.9 * noise)

    # =========================
    # ☕ oxidation + chemical burn
    # =========================
    lab[:, :, 0] -= (18 + 40 * severity) * aging_field   # darkening
    lab[:, :, 1] += (10 + 30 * severity) * aging_field   # red shift
    lab[:, :, 2] += (30 + 90 * severity) * aging_field   # yellowing

    stain_seed = cv2.GaussianBlur(np.random.rand(h, w).astype(np.float32), (0, 0), 12)

    stain = stain_seed * aging_field
    stain = cv2.GaussianBlur(stain, (0, 0), 10)

    lab[:, :, 0] -= stain * 25 * severity

    vignette = np.power(edge_decay, 1.8)

    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    result = result.astype(np.float32) / 255.0

    result *= vignette[..., None]

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def apply_fft_lpf(img, severity=0.5):
    # degradacija visokih frekvencija uz lokalni gubitak detalja

    h, w, _ = img.shape
    img_f = img.astype(np.float32) / 255.0

    sigma = max(5, 22 - severity * 15)

    result_channels = []

    for ch in cv2.split(img_f):

        dft = np.fft.fftshift(np.fft.fft2(ch))

        y, x = np.ogrid[-h//2:h-h//2, -w//2:w-w//2]
        dist2 = x**2 + y**2

        mask = np.exp(-dist2 / (2 * sigma**2))

        filtered = np.real(
            np.fft.ifft2(
                np.fft.ifftshift(dft * mask)
            )
        )

        # veoma velike nepravilne zone
        texture = cv2.resize(
            np.random.rand(4, 4).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        texture = cv2.GaussianBlur(texture, (0, 0), 18)
        texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)

        # samo pojedini delovi slike gube detalje
        local_mask = np.clip((texture - 0.35) * 2.3, 0, 1)

        degraded = (
            ch * (1 - local_mask * 0.9)
            + filtered * (local_mask * 0.9)
        )

        # dodatni blur samo gde postoji degradacija
        blur = cv2.GaussianBlur(
            degraded,
            (11, 11),
            2.8 + severity * 2.0
        )

        degraded = degraded * (1 - local_mask * 0.65) + blur * (local_mask * 0.65)

        # lokalni pad kontrasta
        mean = cv2.GaussianBlur(degraded, (0, 0), 12)

        degraded = (
            mean +
            (degraded - mean) *
            (0.85 - local_mask * 0.45)
        )

        # veoma blaga promena osvetljenja
        illumination = cv2.GaussianBlur(
            np.random.rand(h, w).astype(np.float32),
            (0, 0),
            40
        )

        illumination = (illumination - 0.5) * 0.10 * severity

        degraded = degraded + illumination

        degraded = np.clip(degraded, 0, 1)

        result_channels.append(degraded)

    result = cv2.merge(result_channels)

    return (result * 255).astype(np.uint8)


def apply_cracks(img, severity=0.5):
#pomocu random walk slg ismulacija pukotina na slici
    h, w, _ = img.shape
    result = img.copy()

    num_cracks = int(3 + severity * 12)

    for _ in range(num_cracks):

        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        steps = random.randint(
            int(50 * max(severity, 0.1)),
            int(200 * max(severity, 0.1))
        )

        direction = random.uniform(0, 2 * np.pi)

        points = [(x, y)]

        for step in range(steps):

            direction += random.gauss(0, 0.25)

            step_size = random.randint(2, 5)

            x = int(x + step_size * np.cos(direction))
            y = int(y + step_size * np.sin(direction))

            x = int(np.clip(x, 0, w - 1))
            y = int(np.clip(y, 0, h - 1))

            points.append((x, y))

            # manje grananje
            if random.random() < 0.04 * severity:

                branch_dir = direction + random.choice([-1, 1]) * random.uniform(0.5, 1.2)

                bx, by = x, y

                for _ in range(random.randint(10, 30)):

                    branch_dir += random.gauss(0, 0.18)

                    bx = int(bx + 3 * np.cos(branch_dir))
                    by = int(by + 3 * np.sin(branch_dir))

                    bx = int(np.clip(bx, 0, w - 1))
                    by = int(np.clip(by, 0, h - 1))

                    cv2.circle(
                        result,
                        (bx, by),
                        1,
                        (35, 28, 20),
                        -1
                    )

        for i in range(1, len(points)):

            thickness = random.choices(
                [1, 2, 3],
                weights=[70, 22, 8]
            )[0]

            darkness = random.randint(18, 55)

            color = (
                darkness,
                max(darkness - 6, 0),
                max(darkness - 10, 0)
            )

            cv2.line(
                result,
                points[i - 1],
                points[i],
                color,
                thickness,
                cv2.LINE_AA
            )

            # sitna proširenja duž pukotine
            if random.random() < 0.30:

                cv2.circle(
                    result,
                    points[i],
                    1,
                    (
                        min(darkness + 15, 255),
                        min(darkness + 10, 255),
                        min(darkness + 5, 255)
                    ),
                    -1
                )

    return result

def apply_water_stains(img, severity=0.5):
    # braon/plave mrlje + blur + tamnjenje + “wet diffusion”

    h, w, _ = img.shape
    img_f = img.astype(np.float32) / 255.0
    result = img_f.copy()

    num_stains = int(2 + severity * 6)

    for _ in range(num_stains):

        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        radius = random.randint(
            int(40 + severity * 30),
            int(100 + severity * 80)
        )

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        mask = np.clip(1 - dist / radius, 0, 1)

        # organic diffusion inside stain
        blur_noise = cv2.GaussianBlur(
            np.random.rand(h, w).astype(np.float32),
            (0, 0),
            10
        )

        mask = mask * (0.6 + 0.8 * blur_noise)
        mask = cv2.GaussianBlur(mask, (0, 0), 6)

        # 🎨 brown-blue dirty water tone
        stain_color = np.array([
            random.uniform(0.25, 0.40),  # R (brownish)
            random.uniform(0.28, 0.42),  # G
            random.uniform(0.35, 0.55)   # B (bluish tint)
        ])

        strength = 0.6 + severity * 0.4

        # darken + tint + blur effect
        for c in range(3):
            result[:, :, c] = (
                result[:, :, c] * (1 - mask * strength)
                + stain_color[c] * mask * strength
            )

        # extra dark center (wet pooling)
        result -= mask[..., None] * 0.08 * severity

    # final wet blur
    result = cv2.GaussianBlur(result, (7, 7), 1.2)

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def apply_paint_flaking(img, severity=0.5):
#simulaciaj ljutšenja boje i kao da se planto ispod otkriva
    h, w, _ = img.shape
    img_f = img.astype(np.float32) / 255.0
    result = img_f.copy()
    num_flakes = int(5 + severity * 20)

    for _ in range(num_flakes):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        num_points = random.randint(4, 8)
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points + random.uniform(-0.3, 0.3)
            r = random.randint(int(5 + severity * 5), int(15 + severity * 20))
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            points.append([px, py])

        points = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [points], 1.0)

        canvas_color = np.array([0.92, 0.88, 0.80])
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - mask * 0.9) + canvas_color[c] * mask * 0.9

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def apply_dust_and_scratches(img, severity=0.5):
#simulacije rpašine tj crno/sivih tačkica i ogrebotrinica kao kombinacija ta dva za ostecene slike
    h, w, _ = img.shape
    result = img.copy()

#tackice prašine ranodm idu 
    num_dust =int(200 + severity * 800)

    for _ in range(num_dust):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        color = random.choice([
            (255, 255, 255),  # bela prašina
            (0, 0, 0)         # crna prašina wi
        ])

        result[y, x] = color

    # ogrebotinice :3
    num_scratches = int(5 + severity * 25)

    for _ in range(num_scratches):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)

        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(10, int(80 + severity * 120))

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        color = (random.randint(180, 255),) * 3

        thickness = 1 if severity < 0.5 else 2

        cv2.line(result, (x1, y1), (x2, y2), color, thickness)

#blagi kontrast noise 
    noise = np.random.normal(0, 0.03 * severity, (h, w, 3))
    result = np.clip(result.astype(np.float32) / 255.0 + noise, 0, 1)

    return (result * 255).astype(np.uint8)


def apply_combined_damage(img, severity=0.5):
#kombinovano oštećenje - simuira bap mega proapdanje slike 
    comb = apply_chemical_aging(img, severity)
    comb = apply_cracks(comb, severity)
    comb = apply_anisotropic_diffusion(comb, severity * 0.7)
    return comb

#generisanje 2 dataseta
def build_balanced_dataset(source_dir, output_dir, target_size=(224, 224), desc_msg="Procesiranje"):
    #kreira se folder nula gde je 6 cistih slika samo trnasformisano da bi dataset bio balansiran i 6/9 oštećenja random izbarano
    
    os.makedirs(os.path.join(output_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '1'), exist_ok=True)

#sva oštećenja koja idu na slike
    damage_functions = [
        apply_anisotropic_diffusion,
        apply_mold_and_decay,
        apply_chemical_aging,
        apply_fft_lpf,
        apply_cracks,
        apply_paint_flaking,
        apply_water_stains,
        apply_dust_and_scratches,
        apply_combined_damage,
    ]

    all_images = []
    for root, dirs, files_in_dir in os.walk(source_dir):
        for file in files_in_dir:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                all_images.append(os.path.join(root, file))

    img_counter = 0

    for img_path in tqdm(all_images, desc=desc_msg):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize i konverzija u RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)

        img_flipped = cv2.flip(img, 1)

        # Kreiranje rotacija za čistu originalnu sliku (0, 90, 180 stepeni)
        img_rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_rot180 = cv2.rotate(img, cv2.ROTATE_180)

        # Kreiranje rotacija za čistu flipovanu sliku (0, 90, 180 stepeni)
        img_flipped_rot90 = cv2.rotate(img_flipped, cv2.ROTATE_90_CLOCKWISE)
        img_flipped_rot180 = cv2.rotate(img_flipped, cv2.ROTATE_180)

        base_name = f"img_{img_counter:05d}"

        # ──── FOLDER 0: Čiste slike (Ukupno 6 slika po bazi) ────
        cv2.imwrite(os.path.join(output_dir, '0', f"{base_name}_clean.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, '0', f"{base_name}_clean_rot90.jpg"), cv2.cvtColor(img_rot90, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, '0', f"{base_name}_clean_rot180.jpg"), cv2.cvtColor(img_rot180, cv2.COLOR_RGB2BGR))

        cv2.imwrite(os.path.join(output_dir, '0', f"{base_name}_flip.jpg"), cv2.cvtColor(img_flipped, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, '0', f"{base_name}_flip_rot90.jpg"), cv2.cvtColor(img_flipped_rot90, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, '0', f"{base_name}_flip_rot180.jpg"), cv2.cvtColor(img_flipped_rot180, cv2.COLOR_RGB2BGR))

        # ──── FOLDER 1: Oštećene slike (Ukupno 6 slika po bazi) ────
        # Bira se 6 nasumičnih filtera od 9 ponuđenih
        selected_filters = random.sample(damage_functions, 6)

        for i, damage_func in enumerate(selected_filters):
            random_severity = random.uniform(0.5, 1.0)
            source_img = img if i < 3 else img_flipped
            suffix = "orig" if i < 3 else "flip"

            try:
                damaged_img = damage_func(source_img, severity=random_severity)
                damaged_path = os.path.join(
                    output_dir, '1', f"{base_name}_dmg{i}_{suffix}_{damage_func.__name__}.jpg"
                )
                cv2.imwrite(damaged_path, cv2.cvtColor(damaged_img, cv2.COLOR_RGB2BGR))
            except Exception as e:
                continue

        img_counter += 1



def pokreni_ceo_proces(izvorni_dir, izlazna_baza, prosek_treninga=0.8):
    #deli se na testi trening 
    print(f"{'=' * 60}")
    print(f"{'=' * 60}")
    
    # Privremeni folderi za podelu originala
    temp_trening_izvor = os.path.join(izlazna_baza, "temp_izvor_trening")
    temp_test_izvor = os.path.join(izlazna_baza, "temp_izvor_testiranje")
    
    # Konačni izlazni folderi za model
    trening_izlaz = os.path.join(izlazna_baza, "DATASET_TRENING")
    test_izlaz = os.path.join(izlazna_baza, "DATASET_TEST")

    # prikupljanej sbih  originalnih slika
    sve_slike = []
    for root, dirs, files in os.walk(izvorni_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                sve_slike.append(os.path.join(root, file))

    total_images = len(sve_slike)
    print(f"Pronađeno originalnih slika u izvoru: {total_images}")
    
    if total_images == 0:
        print("nema slika izvornom folderu")
        return
#ranodm mesanje mdoela
    random.seed(42)
    random.shuffle(sve_slike)

    granica = int(total_images * prosek_treninga)
    originali_trening = sve_slike[:granica]
    originali_test = sve_slike[granica:]

    print(f"Podela originala -> Trening deo: {len(originali_trening)} | Test deo: {len(originali_test)}")

    def kopiraj_fajlove(lista_putanja, ciljni_dir):
        os.makedirs(ciljni_dir, exist_ok=True)
        for putanja in lista_putanja:
            rel_path = os.path.relpath(putanja, izvorni_dir)
            dest = os.path.join(ciljni_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(putanja, dest)

    print("\npripreema oriverremnih foldera za trening i test...")
    kopiraj_fajlove(originali_trening, temp_trening_izvor)
    kopiraj_fajlove(originali_test, temp_test_izvor)

    print("\nGenerisanje trening dataseta")
    build_balanced_dataset(
        source_dir=temp_trening_izvor,
        output_dir=trening_izlaz,
        target_size=(224, 224),
        desc_msg="Generisanje Trening Skupa"
    )

    # 4. Generisanje DATASET_TEST
    print("\ngenerisanej test dataseta  ")
    build_balanced_dataset(
        source_dir=temp_test_izvor,
        output_dir=test_izlaz,
        target_size=(224, 224),
        desc_msg="Generisanje Testnog Skupa"
    )

    shutil.rmtree(temp_trening_izvor)
    shutil.rmtree(temp_test_izvor)

    print("\n Kompresujem kreirane skupove u ZIP arhive...")
    try:
        shutil.make_archive(trening_izlaz, 'zip', trening_izlaz)
        print("DATASET_TRENING.zip  napravljen")    
        shutil.make_archive(test_izlaz, 'zip', test_izlaz)
        print("DATASET_TEST.zip napralvjen.")
    except Exception as e:
        print(f"greska tokom zipovanja: {e}")

    # Statistika na kraju
    num_train_0 = len(os.listdir(os.path.join(trening_izlaz, '0')))
    num_train_1 = len(os.listdir(os.path.join(trening_izlaz, '1')))
    num_test_0 = len(os.listdir(os.path.join(test_izlaz, '0')))
    num_test_1 = len(os.listdir(os.path.join(test_izlaz, '1')))

    print(f"\n{'=' * 60}")
    print(f"gotojoooo")
    print(f"{'=' * 60}")
    print(f"1. TRENING DATASET (DATASET_TRENING):")
    print(f"   - Čiste slike (Folder 0):    {num_train_0}")
    print(f"   - Oštećene slike (Folder 1): {num_train_1}")
    print(f"   - Ukupno u treningu:         {num_train_0 + num_train_1}")
    print(f"2. TEST DATASET (DATASET_TEST):")
    print(f"   - Čiste slike (Folder 0):    {num_test_0}")
    print(f"   - Oštećene slike (Folder 1): {num_test_1}")
    print(f"   - Ukupno u testu:            {num_test_0 + num_test_1}")
    print(f"Arhive se nalaze na: {izlazna_baza}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    izvorni_test_folder = r"C:\Users\PC\gitara\slike.projekat\data" 
    baza_izlaznih_foldera = r"C:\Users\PC\gitara\slike.projekat"

    pokreni_ceo_proces(
        izvorni_dir=izvorni_test_folder,
        izlazna_baza=baza_izlaznih_foldera,
        prosek_treninga=0.8  # 80% originalnih slika ide u trening, preostalih 20% u test
    )
