import zipfile, os, pathlib, shutil, collections, json

z = zipfile.ZipFile("chula_main.zip")
OUT_DIR = pathlib.Path("data/anerbc_raw")
N_DIR, A_DIR = OUT_DIR / "Normal", OUT_DIR / "Anemia"
prefix = "Chula-RBC-12-Dataset-main/"

print("Organizing dataset...")
z.extractall("temp_unzip")

root = pathlib.Path("temp_unzip") / prefix
labels_dir = root / "Label"
images_dir = root / "Dataset"

shutil.rmtree(N_DIR, ignore_errors=True)
shutil.rmtree(A_DIR, ignore_errors=True)
N_DIR.mkdir(parents=True, exist_ok=True)
A_DIR.mkdir(parents=True, exist_ok=True)

counts = collections.Counter()
for img_p in images_dir.glob("*"):
    if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    stem = img_p.stem
    lbl_p = labels_dir / f"{stem}.txt"
    classes = set()
    if lbl_p.exists():
        for line in lbl_p.read_text(errors="ignore").strip().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2].isdigit():
                classes.add(int(parts[2]))
    is_anemic = bool(classes - {0})
    target = (A_DIR if is_anemic else N_DIR) / f"{stem}.jpg"
    shutil.copy(img_p, target)
    counts["Anemia" if is_anemic else "Normal"] += 1

shutil.rmtree("temp_unzip", ignore_errors=True)
print("Organized dataset counts:", json.dumps(dict(counts)))
