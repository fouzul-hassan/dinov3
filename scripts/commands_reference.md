# DINOv3 Thalassaemia — Complete Commands Reference

All commands are run from the **repo root** (`dinov3/`) unless otherwise noted.

---

## 0 — Environment Setup (Run Once on Colab)

```python
# Cell 1: Clone repo and install dependencies
!git clone https://github.com/YOUR_FORK/dinov3.git
%cd dinov3

!pip install -r requirements.txt
!pip install wandb pillow-heif scikit-learn
!pip install torchvision --upgrade

# Authenticate with WandB (interactive, run once)
import wandb
wandb.login()
```

```python
# Cell 2: Upload your data to Colab (choose one method)

# Method A: Upload from local ZIP (small dataset)
from google.colab import files
files.upload()   # upload raw-data.zip, then:
!unzip raw-data.zip -d data/

# Method B: Mount Google Drive (recommended for persistence)
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/thalassaemia/raw-data data/

# Method C: Download from URL
# !wget -O data/raw-data.zip "YOUR_DIRECT_URL"
# !unzip data/raw-data.zip -d data/
```

---

## 1 — Data Preprocessing

### 1.1 Full preprocessing (patient-level split + metadata + stats)
```bash
python scripts/preprocess_data.py \
    --raw-data-dir data/raw-data \
    --output-dir   data/preprocessed \
    --train-ratio  0.70 \
    --val-ratio    0.15 \
    --seed         42
```

### 1.2 Quick preprocessing (skip computing dataset stats)
```bash
python scripts/preprocess_data.py \
    --raw-data-dir data/raw-data \
    --output-dir   data/preprocessed \
    --seed         42 \
    --skip-stats
```

### 1.3 Verify output structure
```bash
find data/preprocessed -type d | head -20
python -c "
import numpy as np
meta = 'data/preprocessed/metadata'
for split in ['TRAIN', 'VAL', 'TEST']:
    e = np.load(f'{meta}/entries-{split}.npy', allow_pickle=False)
    print(f'{split}: {len(e)} images')
"
```

### 1.4 View computed dataset statistics
```bash
cat data/preprocessed/metadata/dataset_stats.json
```

---

## 2 — SSL Pretraining

### 2.1 Option A — Train FROM SCRATCH (ViT-Small/16, ~21M params)
```bash
# Recommended: export WandB settings first
export WANDB_PROJECT=thalassaemia-dinov3
export WANDB_RUN_NAME=vits16-scratch-run1

PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \
    --output-dir  outputs/thalassaemia_scratch \
    train.dataset_path=Thalassaemia:split=TRAIN:root=data/preprocessed:extra=data/preprocessed/metadata
```

### 2.2 Option B — Fine-tune FROM PRETRAINED DINOv3 weights ⭐ (Recommended)

**Step B-1: Download pretrained ViT-S/16 weights from HuggingFace**
```python
# In Colab notebook cell:
from huggingface_hub import hf_hub_download
import os

os.makedirs("checkpoints/dinov3_vits16", exist_ok=True)

# Download backbone weights
path = hf_hub_download(
    repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
    filename="model.safetensors",
    local_dir="checkpoints/dinov3_vits16"
)
print(f"Downloaded to: {path}")
```

**Step B-2: Run fine-tuning SSL**
```bash
export WANDB_PROJECT=thalassaemia-dinov3
export WANDB_RUN_NAME=vits16-pretrained-run1

PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_pretrained.yaml \
    --output-dir  outputs/thalassaemia_pretrained \
    train.dataset_path=Thalassaemia:split=TRAIN:root=data/preprocessed:extra=data/preprocessed/metadata \
    student.pretrained_weights=checkpoints/dinov3_vits16/model.safetensors
```

### 2.3 Resume interrupted training (Colab crashes)
```bash
# Simply re-run the same command WITHOUT --no-resume
# The launcher auto-detects the latest checkpoint
PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \
    --output-dir  outputs/thalassaemia_scratch \
    train.dataset_path=Thalassaemia:split=TRAIN:root=data/preprocessed:extra=data/preprocessed/metadata
# ^ No --no-resume = auto-resume from last checkpoint
```

### 2.4 Override config values at CLI
```bash
# Change epochs, batch size, or any config value at launch time
PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \
    --output-dir  outputs/thalassaemia_scratch \
    optim.epochs=100 \
    train.batch_size_per_gpu=8 \
    train.num_workers=2
```

### 2.5 Monitor training (in a separate Colab cell)
```python
# Watch the metrics JSON file in real-time
import json, time

metrics_file = "outputs/thalassaemia_scratch/training_metrics.json"
with open(metrics_file) as f:
    lines = f.readlines()
# Show last 5 iterations
for line in lines[-5:]:
    m = json.loads(line)
    print(f"iter={m['iteration']:4d}  loss={m['total_loss']:.4f}  lr={m.get('lr', 0):.2e}")
```

### 2.6 List available checkpoints
```bash
ls -lh outputs/thalassaemia_scratch/eval/
# Each subdirectory has a teacher_checkpoint.pth
```

---

## 3 — Evaluation

### 3.1 k-NN Evaluation (quick, no training needed)
```bash
# Fast check: are the SSL features semantically meaningful?
python scripts/run_knn_eval.py \
    --checkpoint-dir  outputs/thalassaemia_scratch/eval/training_1999 \
    --data-root       data/preprocessed \
    --output-dir      outputs/knn_eval \
    --k-values 1 5 10 20

# Results saved to outputs/knn_eval/knn_results.json
```

### 3.2 Linear Probe Evaluation (trains a linear head, ~5 min)
```bash
python scripts/run_linear_probe.py \
    --checkpoint-dir outputs/thalassaemia_scratch/eval/training_1999 \
    --data-root      data/preprocessed \
    --output-dir     outputs/linear_probe \
    --epochs         20 \
    --lr             1e-3

# Results saved to outputs/linear_probe/linear_probe_results.json
```

### 3.3 Linear probe with custom hyperparameters
```bash
python scripts/run_linear_probe.py \
    --checkpoint-dir outputs/thalassaemia_pretrained/eval/training_999 \
    --data-root      data/preprocessed \
    --output-dir     outputs/linear_probe_pretrained \
    --epochs         50 \
    --lr             5e-4 \
    --weight-decay   1e-4 \
    --batch-size     32

```

### 3.4 Use DINOv3's built-in linear eval (advanced)
```bash
# Uses DINOv3's own linear.py with ImageNet-style evaluation
PYTHONPATH=${PWD} python dinov3/eval/linear.py \
    model.config_file=outputs/thalassaemia_scratch/config.yaml \
    model.pretrained_weights=outputs/thalassaemia_scratch/eval/training_1999/teacher_checkpoint.pth \
    output_dir=outputs/linear_builtin \
    train.dataset=Thalassaemia:split=TRAIN:root=data/preprocessed:extra=data/preprocessed/metadata \
    train.val_dataset=Thalassaemia:split=VAL:root=data/preprocessed:extra=data/preprocessed/metadata
```

### 3.5 Built-in k-NN eval (DINOv3's own knn.py)
```bash
PYTHONPATH=${PWD} python dinov3/eval/knn.py \
    model.config_file=outputs/thalassaemia_scratch/config.yaml \
    model.pretrained_weights=outputs/thalassaemia_scratch/eval/training_1999/teacher_checkpoint.pth \
    output_dir=outputs/knn_builtin \
    train.dataset=Thalassaemia:split=TRAIN:root=data/preprocessed:extra=data/preprocessed/metadata \
    eval.test_dataset=Thalassaemia:split=VAL:root=data/preprocessed:extra=data/preprocessed/metadata
```

---

## 4 — WandB Controls

```bash
# Enable/disable WandB via environment variable (default: enabled)
export WANDB_ENABLED=true      # or false to disable

# Set project and run name
export WANDB_PROJECT=thalassaemia-dinov3
export WANDB_RUN_NAME=experiment-1

# Add custom tags
export WANDB_TAGS="thalassaemia,ssl,vit-small,from-scratch"

# Offline mode (sync later)
export WANDB_MODE=offline
wandb sync outputs/wandb/offline-run-*

# View WandB dashboard
# → https://wandb.ai/YOUR_ENTITY/thalassaemia-dinov3
```

---

## 5 — Saving Results Back to Google Drive

```python
# Run in a Colab cell to persist results between sessions
import shutil
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints
shutil.copytree(
    "outputs/",
    "/content/drive/MyDrive/thalassaemia/outputs/",
    dirs_exist_ok=True
)
print("Results saved to Google Drive!")
```

---

## 6 — Troubleshooting

### Out of Memory (OOM) on T4
```bash
# Reduce batch size and local crops
PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \
    --output-dir  outputs/thalassaemia_scratch \
    train.batch_size_per_gpu=8 \
    crops.local_crops_number=4
```

### torch.compile errors on Colab
```bash
# Disable compile (already off in our configs, but just in case)
PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \
    --output-dir  outputs/thalassaemia_scratch \
    train.compile=false
```

### Missing HEIC support
```bash
pip install pillow-heif
# Then re-run preprocess_data.py
```

### Google Drive Sync Issues / Missing Image Errors (10x Speedup Solution) ⭐
If you run your preprocessing locally on Windows or inside a Google Drive subfolder, Colab's mounted Drive (FUSE virtual filesystem) often has severe caching and sync delay issues. This causes random `FileNotFoundError` during dataloading (e.g. at sample index 558) and makes image loading extremely slow.

**The Fix:**
Always copy your preprocessed data from your mounted Google Drive to Colab's high-speed local VM disk before starting the training loop. This guarantees 100% of images are present and **speeds up training by 10x to 50x**:

```bash
# 1. Copy the preprocessed dataset to Colab local SSD disk
!mkdir -p /content/data
!cp -r /content/drive/MyDrive/3.ResearchWorks/Thalassaemia/dinov3/data/preprocessed /content/data/

# 2. Run the pretraining script pointing to the local path
PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \
    --output-dir  outputs/thalassaemia_scratch \
    train.dataset_path=Thalassaemia:split=TRAIN:root=/content/data/preprocessed:extra=/content/data/preprocessed/metadata
```

---

### Checkpoint not found
```bash
ls outputs/thalassaemia_scratch/ckpt/
ls outputs/thalassaemia_scratch/eval/
```

### Reset WandB run (start fresh run, not resume)
```bash
# Delete the wandb run ID file
rm -f outputs/thalassaemia_scratch/wandb/
# Then set a new run name
export WANDB_RUN_NAME=experiment-2
```

---

## 7 — Quick Experiment Comparison (run both approaches)

```bash
# 1. Preprocess data (once)
python scripts/preprocess_data.py --raw-data-dir data/raw-data --output-dir data/preprocessed

# 2. Train from scratch
PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_scratch.yaml \
    --output-dir  outputs/scratch

# 3. Fine-tune from pretrained
PYTHONPATH=${PWD} python scripts/run_ssl_colab.py \
    --config-file dinov3/configs/train/thalassaemia_vits16_pretrained.yaml \
    --output-dir  outputs/pretrained \
    student.pretrained_weights=checkpoints/dinov3_vits16/model.safetensors

# 4. Evaluate both with k-NN
python scripts/run_knn_eval.py --checkpoint-dir outputs/scratch/eval/training_1999   --output-dir outputs/eval/scratch_knn
python scripts/run_knn_eval.py --checkpoint-dir outputs/pretrained/eval/training_999 --output-dir outputs/eval/pretrained_knn

# 5. Evaluate both with Linear Probe
python scripts/run_linear_probe.py --checkpoint-dir outputs/scratch/eval/training_1999   --output-dir outputs/eval/scratch_lp
python scripts/run_linear_probe.py --checkpoint-dir outputs/pretrained/eval/training_999 --output-dir outputs/eval/pretrained_lp
```

---

## 8 — Expected Colab Runtime Estimates (T4 GPU)

| Stage | Time |
|-------|------|
| Preprocessing (~3200 imgs) | ~5 min |
| SSL training from scratch (200 epochs) | ~8–12 hrs |
| SSL fine-tuning from pretrained (100 epochs) | ~4–6 hrs |
| Feature extraction (train+val+test) | ~3 min |
| k-NN evaluation | <1 min |
| Linear probe (20 epochs) | ~5 min |

> **Colab tip**: Use Google Drive mounting to persist checkpoints across sessions. 
> Colab disconnects after ~12 hrs — but your training will auto-resume from the last 
> checkpoint when you re-run the same command.

---

## 9 — Key File Locations

| File | Purpose |
|------|---------|
| `scripts/preprocess_data.py` | Data preprocessing pipeline |
| `scripts/run_ssl_colab.py` | SSL training launcher (Colab) |
| `scripts/evaluate_checkpoint.py` | **Unified eval: linear probe + KNN + t-SNE** |
| `scripts/run_linear_probe.py` | Linear probe evaluation (legacy, needs .pth) |
| `scripts/run_knn_eval.py` | k-NN evaluation |
| `dinov3/configs/train/thalassaemia_vits16_scratch.yaml` | Config: train from scratch |
| `dinov3/configs/train/thalassaemia_vits16_pretrained.yaml` | Config: fine-tune from pretrained |
| `dinov3/data/datasets/thalassaemia.py` | Custom dataset class |
| `dinov3/logging/wandb_logger.py` | WandB integration |
| `data/preprocessed/metadata/dataset_stats.json` | Dataset mean/std |
| `data/preprocessed/metadata/split_manifest.json` | Patient → split assignment |
| `outputs/<run>/training_metrics.json` | Training log (all iterations) |
| `outputs/<run>/ckpt/<iter>/` | DCP checkpoint dir (contains __0_0.distcp) |

---

## 10 — Evaluating a DCP Checkpoint (ckpt/999)

The checkpoint at `outputs/thalassaemia_scratch/ckpt/999` uses PyTorch Distributed Checkpoint (DCP)
format (`.distcp` files). Use `scripts/evaluate_checkpoint.py` which handles this natively.

### 10.1 Install extra eval dependencies
```bash
pip install seaborn scikit-learn matplotlib
# Optional, for UMAP instead of t-SNE:
pip install umap-learn
```

### 10.2 KNN-only (fastest, no training needed)
```bash
python scripts/evaluate_checkpoint.py \
    --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \
    --data-root   ../preprocessed \
    --output-dir  outputs/eval_999 \
    --knn-only \
    --knn-k       20
```

### 10.3 Linear probe + KNN + t-SNE (full evaluation)
```bash
python scripts/evaluate_checkpoint.py \
    --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \
    --data-root   ../preprocessed \
    --output-dir  outputs/eval_999 \
    --run-tsne \
    --lp-epochs   30 \
    --lp-lr       1e-3 \
    --knn-k       20
```

### 10.4 With explicit config path
```bash
python scripts/evaluate_checkpoint.py \
    --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \
    --config-path outputs/thalassaemia_scratch/config.yaml \
    --data-root   ../preprocessed \
    --output-dir  outputs/eval_999 \
    --run-tsne
```

### 10.5 UMAP instead of t-SNE (cleaner separation, faster)
```bash
python scripts/evaluate_checkpoint.py \
    --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \
    --data-root   ../preprocessed \
    --output-dir  outputs/eval_999 \
    --run-tsne \
    --tsne-method umap
```

### 10.6 Output files produced
```
outputs/eval_999/
├── eval_results.json          ← all metrics (accuracy, F1, etc.)
├── linear_head.pth            ← saved linear probe weights
├── confusion_matrix_knn.png   ← KNN confusion matrix heatmap
├── confusion_matrix_linear.png← Linear probe confusion matrix heatmap
├── tsne_all.png               ← t-SNE of all splits combined
└── tsne_test.png              ← t-SNE of test split only
```

### 10.7 Full Colab notebook cells
```python
# Cell 1: Install dependencies
!pip install seaborn scikit-learn matplotlib umap-learn -q

# Cell 2: Run evaluation
!python scripts/evaluate_checkpoint.py \
    --ckpt-path   outputs/thalassaemia_scratch/ckpt/999 \
    --data-root   ../preprocessed \
    --output-dir  outputs/eval_iter999 \
    --run-tsne \
    --lp-epochs   30

# Cell 3: Display results in Colab
import json
from IPython.display import Image, display

with open("outputs/eval_iter999/eval_results.json") as f:
    r = json.load(f)
print(json.dumps(r, indent=2))

display(Image("outputs/eval_iter999/confusion_matrix_linear.png"))
display(Image("outputs/eval_iter999/tsne_all.png"))
```

