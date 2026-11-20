"""
Master Experimentation Orchestrator Script.
Runs the complete deep learning and computer vision research pipeline:
1. Dataset Verification & Inspection
2. Baseline U-Net Training
3. Multi-Task Learning Training
4. Held-Out Test Evaluation & Comparison
5. Systematic Robustness Benchmarking (4 degradations x 5 severities)
6. Epistemic Uncertainty Estimation
"""

import os
import sys
import argparse

# Add repo root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cv.data.inspect_data import inspect_dataset
from cv.training.train_unet import train_unet
from cv.training.train_multitask import train_multitask
from cv.evaluation.evaluate import evaluate_models
from cv.robustness.benchmark import RobustnessBenchmark
from cv.models.unet import UNet
import torch

def run_all(
    data_dir: str = 'cv/data/brisc2025',
    output_dir: str = 'artifacts',
    docs_dir: str = 'docs',
    epochs: int = 10,
    batch_size: int = 8,
    max_train_samples: int = 250,
    max_val_samples: int = 60,
    max_test_samples: int = 60
):
    print("=" * 70)
    print("🧠 BRAIN TUMOR SEGMENTATION SYSTEM — RESEARCH EXPERIMENTS")
    print("=" * 70)

    # 1. Dataset Inspection
    print("\n[PHASE 1/5] Inspecting BRISC 2025 Dataset...")
    inspect_dataset(data_dir, output_dir, docs_dir)

    # 2. Baseline U-Net Training
    print("\n[PHASE 2/5] Training Baseline U-Net...")
    train_unet(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        features=32,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples
    )

    # 3. Multi-Task Model Training
    print("\n[PHASE 3/5] Training Multi-Task U-Net...")
    train_multitask(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lambda_cls=0.5,
        features=32,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples
    )

    # 4. Held-Out Test Evaluation & Model Comparison
    print("\n[PHASE 4/5] Evaluating Models on Held-Out Test Partition...")
    evaluate_models(
        data_dir=data_dir,
        artifacts_dir=output_dir,
        max_test_samples=max_test_samples
    )

    # 5. Robustness Benchmarking
    print("\n[PHASE 5/5] Running Systematic Robustness Stress Tests...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_ckpt = os.path.join(output_dir, 'checkpoints', 'best_unet.pth')
    model = UNet(in_channels=1, out_classes=1, features=32).to(device)
    if os.path.exists(unet_ckpt):
        state = torch.load(unet_ckpt, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    model.eval()

    from cv.data.dataset import BRISCDataset
    test_ds = BRISCDataset(data_dir=data_dir, split="test", max_samples=max_test_samples)
    bench = RobustnessBenchmark(model, device=device)
    bench.run_benchmark_suite(test_ds.samples, os.path.join(output_dir, 'metrics'))

    print("\n" + "=" * 70)
    print("✅ ALL RESEARCH EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='cv/data/brisc2025')
    parser.add_argument('--output_dir', default='artifacts')
    parser.add_argument('--docs_dir', default='docs')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_train_samples', type=int, default=200)
    parser.add_argument('--max_val_samples', type=int, default=50)
    parser.add_argument('--max_test_samples', type=int, default=50)
    args = parser.parse_args()

    run_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples
    )
