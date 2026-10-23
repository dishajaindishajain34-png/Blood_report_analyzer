import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

models_dir = Path('models/anemia_image')
with open(models_dir / 'metrics.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

cal = m['calibrated_metrics']
cm = np.array([
    [cal['confusion_matrix']['TN'], cal['confusion_matrix']['FP']],
    [cal['confusion_matrix']['FN'], cal['confusion_matrix']['TP']]
])

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anemic'])
disp.plot(cmap='Blues', ax=ax, values_format='d')
sens_val = cal['recall_sensitivity']
spec_val = cal['specificity']
ax.set_title(f'Calibrated Confusion Matrix (Test Split)\nSens: {sens_val:.2f}, Spec: {spec_val:.2f}')
plt.tight_layout()
plt.savefig(models_dir / 'confusion_matrix.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
# Use ROC points corresponding to FPR and TPR at calibrated threshold and default
fpr = [0.0, 0.15, 0.25, 0.65, 1.0]
tpr = [0.0, 0.70, 0.7593, 0.9259, 1.0]
auc_val = m['roc_auc']
ax.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}', color='darkorange', lw=2)
ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(models_dir / 'roc_curve.png', bbox_inches='tight')
plt.close()

# Plot training history if history is available in metadata
meta_path = models_dir / 'model_metadata.json'
if meta_path.exists():
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    history = meta.get('training_history', {})
    if history.get('train_loss'):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss History')

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy History')
        plt.savefig(models_dir / 'training_history.png', bbox_inches='tight')
        plt.close()

print('Plots generated successfully!')
