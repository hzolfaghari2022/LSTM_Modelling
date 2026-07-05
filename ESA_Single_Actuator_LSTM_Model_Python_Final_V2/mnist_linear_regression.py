#!/usr/bin/env python3
"""
MNIST Linear Classifier with PyTorch.

Goal
----
Train a simple linear classifier on MNIST and show/save the results.

Model structure
---------------
MNIST image: 28 x 28
Flatten: 784 features
Linear layer: 784 -> 10 classes

This is NOT a CNN.
This is a basic linear classification baseline.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# User settings
# ============================================================
DATA_FOLDER = "data"
FIG_FOLDER = "figures"
OUTPUT_FOLDER = "outputs"

BATCH_SIZE = 64
MAX_EPOCHS = 10
INITIAL_LEARN_RATE = 1e-3
RANDOM_SEED = 42

# Set True to show figures on screen after training
SHOW_FIGURES = True

# GitHub settings
DO_GITHUB_PUSH = False
REPO_SSH = "git@github.com:hzolfaghari2022/Machine_Learning_Complete.git"
REPO_NAME = "Machine_Learning_Complete"
BRANCH_NAME = "main"
TARGET_FOLDER_NAME = "MNIST_Linear_Classifier"


# ============================================================
# Utility functions
# ============================================================
def run_command(command: list[str], cwd: Path | None = None) -> Tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout


def save_figure(fig: plt.Figure, fig_folder: Path, base_name: str) -> None:
    fig_folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_folder / f"{base_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_folder / f"{base_name}.pdf", bbox_inches="tight")

    if not SHOW_FIGURES:
        plt.close(fig)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred) * 100.0)


# ============================================================
# MNIST data loading
# ============================================================
def load_mnist_data(script_dir: Path) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST train and test datasets.

    Original image shape:
        [1, 28, 28]

    The model will flatten it internally to:
        [784]
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_root = script_dir / DATA_FOLDER

    train_dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


# ============================================================
# Linear classification model
# ============================================================
class MNISTLinearClassifier(nn.Module):
    """
    Simple linear classifier for MNIST.

    Input:
        image tensor with shape [batch, 1, 28, 28]

    Flattened:
        [batch, 784]

    Output:
        logits with shape [batch, 10]
    """

    def __init__(self) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ============================================================
# Training and evaluation
# ============================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:

    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * images.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    average_loss = total_loss / len(train_loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:

    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += float(loss.detach().cpu()) * images.size(0)

            predictions = torch.argmax(logits, dim=1)

            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)

    return average_loss, accuracy, y_true, y_pred


# ============================================================
# Figures
# ============================================================
def plot_training_history(history: list[dict[str, float]], fig_folder: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    test_acc = [row["test_accuracy"] for row in history]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(epochs, train_loss, linewidth=1.8, label="Training loss")
    ax.plot(epochs, test_loss, "--", linewidth=1.8, label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("MNIST Linear Classifier Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, fig_folder, "Fig01_Training_Loss")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(epochs, train_acc, linewidth=1.8, label="Training accuracy")
    ax.plot(epochs, test_acc, "--", linewidth=1.8, label="Test accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MNIST Linear Classification Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, fig_folder, "Fig02_Accuracy")


def plot_example_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    fig_folder: Path,
) -> None:

    model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        predictions = torch.argmax(logits, dim=1).detach().cpu()

    images_cpu = images.detach().cpu()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    axes = axes.ravel()

    for i in range(10):
        image = images_cpu[i, 0].numpy()
        true_label = int(labels[i])
        pred_label = int(predictions[i])

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}", fontsize=10)
        axes[i].axis("off")

    fig.suptitle("Example MNIST Predictions: Linear Classifier", fontsize=14)
    save_figure(fig, fig_folder, "Fig03_Example_Predictions")


def plot_confusion_matrix_simple(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fig_folder: Path,
) -> None:

    cm = np.zeros((10, 10), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title("MNIST Confusion Matrix: Linear Classifier")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    save_figure(fig, fig_folder, "Fig04_Confusion_Matrix")


# ============================================================
# GitHub push
# ============================================================
def github_push(source_folder: Path) -> None:
    print("====================================================")
    print("Preparing GitHub SSH push")
    print("====================================================")

    github_root = Path(tempfile.gettempdir()) / "Python_GitHub_Repos"
    repo_folder = github_root / REPO_NAME
    target_folder = repo_folder / TARGET_FOLDER_NAME

    github_root.mkdir(parents=True, exist_ok=True)

    if not repo_folder.exists():
        status, out = run_command(["git", "clone", REPO_SSH, str(repo_folder)])
        if status != 0:
            raise RuntimeError(f"Git clone failed using SSH.\n{out}")

    run_command(["git", "remote", "set-url", "origin", REPO_SSH], cwd=repo_folder)
    run_command(["git", "branch", "-M", BRANCH_NAME], cwd=repo_folder)

    run_command(
        ["git", "stash", "push", "-u", "-m", "auto-stash-before-mnist-linear-pull"],
        cwd=repo_folder,
    )

    status, out = run_command(["git", "pull", "origin", BRANCH_NAME, "--rebase"], cwd=repo_folder)
    if status != 0:
        print("Warning: git pull had an issue:")
        print(out)

    target_folder.mkdir(parents=True, exist_ok=True)

    patterns = [
        "*.py",
        "*.txt",
        "*.csv",
        "*.pt",
        "*.pdf",
        "*.png",
        "*.md",
        ".gitignore",
    ]

    for pattern in patterns:
        for file in source_folder.glob(pattern):
            if file.is_file():
                shutil.copy2(file, target_folder / file.name)

    source_fig = source_folder / FIG_FOLDER
    target_fig = target_folder / FIG_FOLDER

    if source_fig.is_dir():
        target_fig.mkdir(parents=True, exist_ok=True)
        for src_file in source_fig.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(source_fig)
                dst_file = target_fig / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)

    source_output = source_folder / OUTPUT_FOLDER
    target_output = target_folder / OUTPUT_FOLDER

    if source_output.is_dir():
        target_output.mkdir(parents=True, exist_ok=True)
        for src_file in source_output.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(source_output)
                dst_file = target_output / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)

    run_command(["git", "config", "user.name", "Hussein Zolfaghari"], cwd=repo_folder)
    run_command(["git", "config", "user.email", "h.zolfaghari2015@gmail.com"], cwd=repo_folder)

    status, out = run_command(["git", "add", "."], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(f"Git add failed:\n{out}")

    status, _ = run_command(["git", "diff", "--cached", "--quiet"], cwd=repo_folder)

    if status != 0:
        status, out = run_command(
            ["git", "commit", "-m", "Add MNIST linear classifier"],
            cwd=repo_folder,
        )
        if status != 0:
            raise RuntimeError(f"Git commit failed:\n{out}")

    status, out = run_command(["git", "push", "-u", "origin", BRANCH_NAME], cwd=repo_folder)
    if status != 0:
        raise RuntimeError(f"Git push failed using SSH.\n{out}")

    print("Files pushed successfully to GitHub.")


# ============================================================
# Main workflow
# ============================================================
def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    fig_folder = script_dir / FIG_FOLDER
    output_folder = script_dir / OUTPUT_FOLDER

    fig_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    print("====================================================")
    print("MNIST PyTorch Linear Classification")
    print("====================================================")

    train_loader, test_loader = load_mnist_data(script_dir)

    images, labels = next(iter(train_loader))

    print(f"Training image batch shape: {images.shape}")
    print(f"Training label batch shape: {labels.shape}")
    print("Each image is 1 x 28 x 28 and will be flattened to 784 features.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = MNISTLinearClassifier().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARN_RATE)

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        test_loss, test_acc, y_true, y_pred = evaluate(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        })

        print(
            f"Epoch {epoch:3d} | "
            f"Train loss: {train_loss:.6f} | "
            f"Train acc: {train_acc:.2f}% | "
            f"Test loss: {test_loss:.6f} | "
            f"Test acc: {test_acc:.2f}%"
        )

    final_test_loss, final_test_acc, y_true, y_pred = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("====================================================")
    print("Final test results")
    print("====================================================")
    print(f"Final test loss: {final_test_loss:.6f}")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print("====================================================")

    # Save trained model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "MNISTLinearClassifier",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": INITIAL_LEARN_RATE,
            "final_test_loss": final_test_loss,
            "final_test_accuracy": final_test_acc,
        },
        output_folder / "mnist_linear_classifier_model.pt",
    )

    # Save training history
    history_file = output_folder / "mnist_linear_training_history.csv"
    with history_file.open("w", encoding="utf-8") as fid:
        fid.write("epoch,train_loss,train_accuracy,test_loss,test_accuracy\n")
        for row in history:
            fid.write(
                f"{row['epoch']},"
                f"{row['train_loss']:.8f},"
                f"{row['train_accuracy']:.8f},"
                f"{row['test_loss']:.8f},"
                f"{row['test_accuracy']:.8f}\n"
            )

    # Save summary
    summary_file = output_folder / "mnist_linear_summary.txt"
    with summary_file.open("w", encoding="utf-8") as fid:
        fid.write("MNIST PyTorch Linear Classification Summary\n")
        fid.write("==========================================\n\n")
        fid.write("Dataset: MNIST handwritten digits\n")
        fid.write("Image size: 28 x 28 grayscale\n")
        fid.write("Flattened input size: 784\n")
        fid.write("Number of classes: 10\n")
        fid.write("Model: Linear layer, 784 -> 10\n")
        fid.write(f"Batch size: {BATCH_SIZE}\n")
        fid.write(f"Epochs: {MAX_EPOCHS}\n")
        fid.write(f"Learning rate: {INITIAL_LEARN_RATE}\n")
        fid.write(f"Final test loss: {final_test_loss:.8f}\n")
        fid.write(f"Final test accuracy: {final_test_acc:.4f}%\n")

    # Create README
    readme_file = script_dir / "README.md"
    with readme_file.open("w", encoding="utf-8") as fid:
        fid.write("# MNIST PyTorch Linear Classifier\n\n")
        fid.write("This project trains a simple linear classifier on the MNIST handwritten digit dataset.\n\n")
        fid.write("## Model\n\n")
        fid.write("```text\n")
        fid.write("28 x 28 image -> flatten to 784 features -> linear layer 784 -> 10 classes\n")
        fid.write("```\n\n")
        fid.write("This model is a linear classification baseline, not a CNN.\n\n")
        fid.write("## Run\n\n")
        fid.write("```bash\n")
        fid.write("python mnist_linear_classifier.py\n")
        fid.write("```\n\n")
        fid.write("## Requirements\n\n")
        fid.write("```bash\n")
        fid.write("pip install torch torchvision matplotlib numpy\n")
        fid.write("```\n")

    # Create requirements.txt
    requirements_file = script_dir / "requirements.txt"
    with requirements_file.open("w", encoding="utf-8") as fid:
        fid.write("torch\n")
        fid.write("torchvision\n")
        fid.write("matplotlib\n")
        fid.write("numpy\n")

    # Create .gitignore
    gitignore_file = script_dir / ".gitignore"
    with gitignore_file.open("w", encoding="utf-8") as fid:
        fid.write("data/\n")
        fid.write("__pycache__/\n")
        fid.write("*.pyc\n")

    # Create figures
    plot_training_history(history, fig_folder)
    plot_example_predictions(model, test_loader, device, fig_folder)
    plot_confusion_matrix_simple(y_true, y_pred, fig_folder)

    print("Files saved successfully.")
    print(f"Figures folder: {fig_folder}")
    print(f"Outputs folder: {output_folder}")

    if SHOW_FIGURES:
        plt.show()

    if DO_GITHUB_PUSH:
        github_push(script_dir)


if __name__ == "__main__":
    main()