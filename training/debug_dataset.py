from ultralytics import YOLO
from pathlib import Path
import yaml

data_yaml = Path('C:/Users/Admin/Downloads/Blood_report_analyzer/training/data.yaml')
print('Data yaml:', data_yaml)
with open(data_yaml) as f:
    d = yaml.safe_load(f)
print(d)
model = YOLO('yolov8n.pt')
# use build dataset function
from ultralytics.data.datasets import LoadImages
print('Attempting simple image/label checks...')
train_dir = Path('C:/Users/Admin/Downloads/archive_images/train/img')
labels_dir = Path('C:/Users/Admin/Downloads/archive_images/train/labels')
imgs = list(train_dir.glob('*'))
print('Images found:', len(imgs))
# check images without labels
missing = []
for img in imgs:
    lbl = labels_dir / (img.stem + '.txt')
    if not lbl.exists():
        missing.append(img.name)
print('Missing labels:', len(missing))
print('Sample label for first image:')
print((labels_dir / (imgs[0].stem + '.txt')).read_text()[:300])
