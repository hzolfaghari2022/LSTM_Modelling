"""
Training loop, balanced sampling, inference helpers and error metrics.
"""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    BALANCE_POWER,
    BATCH_SIZE,
    DISPLACEMENT_LOSS_WEIGHT,
    DISPLACEMENT_PEAK_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FINAL_FINE_TUNE_EPOCHS,
    FINAL_FINE_TUNE_LEARNING_RATE,
    FORCE_LOSS_WEIGHT,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MINIMUM_IMPROVEMENT,
    MIN_LEARNING_RATE,
    SAMPLES_PER_EPOCH,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
)
from data_utils import WindowDataset


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def balanced_sampler(dataset, samples_per_epoch=SAMPLES_PER_EPOCH,
                     balance_power=BALANCE_POWER):
    """
    Draw training windows so that short records are not buried.

    The wideband chirp holds twenty thousand samples while a step response
    holds eight hundred. Without reweighting the loss is dominated by one
    record and the transient cases are effectively ignored.
    """
    record_of_window = dataset.record_of_window
    counts = np.bincount(record_of_window)
    counts = np.where(counts == 0, 1, counts)
    weight_per_record = (1.0 / counts) ** balance_power
    weights = weight_per_record[record_of_window]
    weights = weights / weights.sum()

    draws = min(samples_per_epoch, len(dataset)) if samples_per_epoch <= 0 else samples_per_epoch

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(draws),
        replacement=True,
    )


def make_loaders(records, data):
    """Build the training and validation loaders."""
    train_dataset = WindowDataset(records, data["training_pairs"])
    validation_dataset = WindowDataset(records, data["validation_pairs"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=balanced_sampler(train_dataset),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader, train_dataset, validation_dataset


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

def accuracy_focused_loss(prediction, target):
    """Weighted mean squared error with extra emphasis on large excursions."""
    squared_error = (prediction - target) ** 2

    peak_weight = 1.0 + DISPLACEMENT_PEAK_WEIGHT * torch.abs(target[:, 0])

    displacement_loss = torch.mean(peak_weight * squared_error[:, 0])
    force_loss = torch.mean(squared_error[:, 1])

    return (
        DISPLACEMENT_LOSS_WEIGHT * displacement_loss
        + FORCE_LOSS_WEIGHT * force_loss
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    """One pass over a loader. Training when an optimizer is supplied."""
    training = optimizer is not None
    model.train() if training else model.eval()

    non_blocking = device.type == "cuda"
    running_sum = 0.0
    running_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for input_batch, target_batch in loader:
            input_batch = input_batch.to(device, non_blocking=non_blocking)
            target_batch = target_batch.to(device, non_blocking=non_blocking)

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(input_batch)
            loss = accuracy_focused_loss(prediction, target_batch)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP_NORM
                )
                optimizer.step()

            running_sum += float(loss.item()) * len(input_batch)
            running_count += len(input_batch)

    return running_sum / max(running_count, 1)


def train_model(model, records, data, device):
    """Train with early stopping on the held out validation blocks."""
    train_loader, validation_loader, train_dataset, validation_dataset = make_loaders(
        records, data
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    training_history = []
    validation_history = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        training_loss = run_epoch(model, train_loader, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, device)

        training_history.append(training_loss)
        validation_history.append(validation_loss)
        scheduler.step(validation_loss)

        if validation_loss < best_loss - MINIMUM_IMPROVEMENT:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"train={training_loss:.6f} | "
            f"validation={validation_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping. Best epoch = {best_epoch}", flush=True)
            break

    model.load_state_dict(best_weights)

    return (
        np.asarray(training_history, dtype=np.float64),
        np.asarray(validation_history, dtype=np.float64),
        best_epoch,
    )


def fine_tune_on_development_data(model, records, data, device):
    """
    Short refresh on training plus validation windows.

    Internal test blocks and every pure test record stay untouched.
    """
    if FINAL_FINE_TUNE_EPOCHS <= 0:
        return np.asarray([], dtype=np.float64)

    combined_pairs = list(data["training_pairs"]) + list(data["validation_pairs"])
    dataset = WindowDataset(records, combined_pairs)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=balanced_sampler(dataset),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINAL_FINE_TUNE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    for epoch in range(FINAL_FINE_TUNE_EPOCHS):
        epoch_loss = run_epoch(model, loader, device, optimizer)
        history.append(epoch_loss)
        print(
            f"Final fine tune {epoch + 1:02d}/{FINAL_FINE_TUNE_EPOCHS} | "
            f"development loss={epoch_loss:.6f}",
            flush=True,
        )

    return np.asarray(history, dtype=np.float64)


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

@torch.no_grad()
def predict_pairs(model, records, pairs, device, batch_size=1024):
    """Run the frozen model over an explicit list of window indices."""
    model.eval()
    dataset = WindowDataset(records, pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    parts = []
    for input_batch, _ in loader:
        input_batch = input_batch.to(device, non_blocking=device.type == "cuda")
        parts.append(model(input_batch).cpu().numpy())

    return np.concatenate(parts, axis=0)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def calculate_metrics(measured, predicted, negligible_range=0.0):
    """
    Standard system identification error measures.

    negligible_range guards the relative measures. A coefficient of
    determination is a comparison against the variance of the reference
    signal, so when that reference is flat at zero, as the Lorentz force is
    in the zero current records, R2 and the fit percentage carry no
    information and are reported as not available. RMSE, MAE and the maximum
    absolute error stay meaningful in that situation and should be read
    instead.
    """
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = measured - predicted

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    maximum_absolute_error = float(np.max(np.abs(error)))

    signal_range = float(measured.max() - measured.min())
    degenerate = signal_range < negligible_range

    centred = measured - measured.mean()
    variance_sum = float(np.sum(centred ** 2))

    if degenerate or variance_sum <= 0.0:
        r2 = np.nan
        fit = np.nan
        normalised_rmse = np.nan
    else:
        r2 = 1.0 - float(np.sum(error ** 2)) / variance_sum
        norm_denominator = float(np.linalg.norm(centred))
        fit = 100.0 * (1.0 - float(np.linalg.norm(error)) / norm_denominator)
        normalised_rmse = 100.0 * rmse / signal_range if signal_range > 0 else np.nan

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MaxAbsError": maximum_absolute_error,
        "SignalRange": signal_range,
        "NRMSE_percent": normalised_rmse,
        "R2": r2,
        "Fit_percent": fit,
        "ReferenceIsFlat": bool(degenerate),
    }
