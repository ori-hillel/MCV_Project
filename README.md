# Surgical Phase and Gaze Prior Reproduction Guide

This project reproduces the full pipeline used for the surgical phase recognition model:

1. Generate pseudo-gaze supervision from the paper-inspired gaze teacher.
2. Train the gaze student model.
3. Train the V2 phase model with gaze-conditioned fusion.
4. Evaluate the final model on the test split.

The repository is designed to run from a working directory that contains the project files and the dataset. The files and folders can be named however you want as long as the structure below is preserved.

## Repository layout

Expected project root:

```text
<repo-root>/
├── dataset.py
├── model_v2.py
├── gaze_student.py
├── gaze_data.py
├── paper_gaze_teacher.py
├── generate_pseudo_gaze.py
├── train_gaze_student.py
├── train_v2.py
├── evaluate_from_weights.py
├── loss.py
├── utils.py
├── images/
├── annotations/
├── checkpoints/
├── paper_gaze_outputs/
├── README.md
├── requirements.txt
└── ...
```

Notes:
- `images/` and `annotations/` are dataset folders.
- `checkpoints/` holds training checkpoints.
- `paper_gaze_outputs/` holds pseudo-gaze targets and the gaze-student checkpoint.

## Using the provided checkpoints on a separate drive

The checkpoint folder for the informed model from the paper and the `paper_gaze_outputs` folder are available on the project drive. If you do not want to retrain the gaze prior or the phase model, you can skip the training steps and directly use those saved artifacts for evaluation.on.

In that case, you only need to:

1. copy or point the project to the checkpoint folder containing `best_model_v2.pt`
2. copy or point the project to the gaze checkpoint folder containing `best_gaze_student.pt`
3. run `evaluate_from_weights.py` against the dataset root

This is the fastest way to reproduce the reported evaluation without retraining.

## Quick start without retraining

If you do not want to reproduce the full training pipeline, you can skip to evaluation using the supplied checkpoint folders on the project drive.

The required artifacts are:
- the phase-model checkpoint: `checkpoints/best_model_v2.pt`
- the gaze prior checkpoint: `paper_gaze_outputs/best_gaze_student.pt`
- the dataset root containing `images/` and `annotations/`

To run evaluation only:

```powershell
python evaluate_from_weights.py --weights "checkpoints\best_model_v2.pt" --gaze-weights "paper_gaze_outputs\best_gaze_student.pt" --root "." --batch-size 16 --device cuda
```

This reproduces the reported evaluation without retraining the model.

## Requirements

### Hardware
- NVIDIA GPU strongly recommended (CUDA-capable)
- Minimum: 12 GB VRAM; recommended: 24 GB+
- Python 3.10 or 3.11

### Software
Install the following before running the project:

```bash
# Windows (PowerShell)
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy matplotlib seaborn scikit-learn pillow tqdm
```

If you do not want to use the CUDA wheel, use the standard PyTorch installation instead:

```bash
python -m pip install torch torchvision torchaudio
```

Optional but useful:

```bash
python -m pip install jupyter notebook
```

## Dataset structure

Your dataset should be organized as:

```text
<dataset-root>/
├── images/
│   ├── video_01/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── ...
├── annotations/
│   ├── bbox/
│   │   ├── by_split/
│   │   │   ├── hands/
│   │   │   │   ├── train.json
│   │   │   │   ├── val.json
│   │   │   │   └── test.json
│   │   │   └── tools/
│   │   │       ├── train.json
│   │   │       ├── val.json
│   │   │       └── test.json
│   │   └── phase/
│   │       └── *.csv
│   └── ...
└── ...
```

The code expects the dataset root to contain:
- `images/`
- `annotations/bbox/by_split/...`
- `annotations/bbox/phase/...`

## Step 1: Generate pseudo-gaze targets

This creates the teacher-derived gaze supervision used to train the gaze student.

### Train split

```powershell
python generate_pseudo_gaze.py --root-dir "." --split train --output "paper_gaze_outputs\train_pseudo_gaze.pt" --fps 30 --grid-size 14 --device cuda
```

### Validation split

```powershell
python generate_pseudo_gaze.py --root-dir "." --split val --output "paper_gaze_outputs\val_pseudo_gaze.pt" --fps 30 --grid-size 14 --device cuda
```

### Test split (optional)

```powershell
python generate_pseudo_gaze.py --root-dir "." --split test --output "paper_gaze_outputs\test_pseudo_gaze.pt" --fps 30 --grid-size 14 --device cuda
```

## Step 2: Train the gaze student model

This trains the `ViTGazeStudent` model using the generated pseudo-gaze targets.

```powershell
python train_gaze_student.py --train-targets "paper_gaze_outputs\train_pseudo_gaze.pt" --val-targets "paper_gaze_outputs\val_pseudo_gaze.pt" --output "paper_gaze_outputs\best_gaze_student.pt" --epochs 20 --batch-size 32 --num-workers 4
```

Recommended default values are already set in the script, so you can usually shorten this to:

```powershell
python train_gaze_student.py --train-targets "paper_gaze_outputs\train_pseudo_gaze.pt" --val-targets "paper_gaze_outputs\val_pseudo_gaze.pt" --output "paper_gaze_outputs\best_gaze_student.pt"
```

This saves the best gaze checkpoint as:

```text
paper_gaze_outputs\best_gaze_student.pt
```

## Step 3: Train the phase model

The phase-model training script writes checkpoints into a local `checkpoints/` directory relative to the current working directory.

To keep checkpoints on a dedicated drive, run the training from a working directory that contains the repo files and a `checkpoints` folder in the same location as the working directory. In practice, the simplest method is to keep the repository checkout on the same drive as the artifacts, or copy/symlink the local `checkpoints` directory to the target drive.

From the repo root:

```powershell
python train_v2.py
```

This script will:
- load the dataset from the project root
- use `paper_gaze_outputs/best_gaze_student.pt` as the gaze prior
- save best checkpoints under `checkpoints/`
- update `checkpoints/best_model_v2.pt` whenever the test Jaccard improves

The script also saves score-tagged checkpoints such as:

```text
checkpoints/best_test_jaccard_0.2636.pt
```

and keeps the active best checkpoint as:

```text
checkpoints/best_model_v2.pt
```

## Step 4: Evaluate the trained phase model

After training, evaluate the best saved model:

```powershell
python evaluate_from_weights.py --weights "checkpoints\best_model_v2.pt" --gaze-weights "paper_gaze_outputs\best_gaze_student.pt" --root "." --batch-size 16 --device cuda
```

If the files are in the repo root, the default call is enough:

```powershell
python evaluate_from_weights.py
```

This script prints:
- raw Jaccard IoU
- smoothed Jaccard IoU
- accuracy, precision, recall, and loss
- and saves the confusion matrix as `test_confusion_matrix.png`

## Full reproduction summary

A complete from-scratch workflow is:

```powershell
# 1) Generate pseudo-gaze supervision
python generate_pseudo_gaze.py --root-dir "." --split train --output "paper_gaze_outputs\train_pseudo_gaze.pt" --fps 30 --grid-size 14 --device cuda
python generate_pseudo_gaze.py --root-dir "." --split val --output "paper_gaze_outputs\val_pseudo_gaze.pt" --fps 30 --grid-size 14 --device cuda

# 2) Train gaze model
python train_gaze_student.py --train-targets "paper_gaze_outputs\train_pseudo_gaze.pt" --val-targets "paper_gaze_outputs\val_pseudo_gaze.pt" --output "paper_gaze_outputs\best_gaze_student.pt"

# 3) Train phase model
python train_v2.py

# 4) Evaluate phase model
python evaluate_from_weights.py --weights "checkpoints\best_model_v2.pt" --gaze-weights "paper_gaze_outputs\best_gaze_student.pt" --root "."
```

## Important notes

- The gaze model depends on the pseudo-gaze teacher and the generated pseudo-gaze `.pt` files.
- The phase model depends on the trained gaze checkpoint `best_gaze_student.pt`.
- The evaluation script depends on `best_model_v2.pt` and the gaze checkpoint.
- Use a clean Python environment to avoid version mismatches.
- The exact names of the folders can vary, but the folder structure and file names inside the root must match the project assumptions.

## Troubleshooting

### CUDA not detected
Check that:
- your NVIDIA driver is installed
- CUDA is enabled in the environment
- PyTorch matches your CUDA version

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### Dataset not found
Make sure the root path contains:
- `images/`
- `annotations/bbox/by_split/...`
- `annotations/bbox/phase/...`

### Missing training outputs
Ensure the pseudo-gaze targets were generated before running `train_gaze_student.py`.

### Checkpoints not saving
The phase training script writes to `checkpoints/` in the current working directory. Run the script from a directory that has write access and keep the `checkpoints` directory in the same working folder as the repo.

## Expected final outputs

After a successful run, you should have:

```text
paper_gaze_outputs/
├── train_pseudo_gaze.pt
├── val_pseudo_gaze.pt
├── best_gaze_student.pt

checkpoints/
├── best_model_v2.pt
├── best_test_jaccard_xxxx.pt
├── model_architecture_xxxx.py
```

and the evaluation result files such as:

```text
test_confusion_matrix.png
```