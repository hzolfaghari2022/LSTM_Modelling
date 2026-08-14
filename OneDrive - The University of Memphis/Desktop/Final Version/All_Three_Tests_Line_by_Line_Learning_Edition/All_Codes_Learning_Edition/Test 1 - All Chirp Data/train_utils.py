# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Training, prediction, and metric functions."""

# Import copying support so the best model weights can be saved independently.
import copy

# Import NumPy for numerical arrays, normalization, errors, and metrics.
import numpy as np
# Import PyTorch for tensors, the LSTM model, training, and prediction.
import torch
# Import selected names from torch.utils.data instead of importing its complete namespace.
from torch.utils.data import DataLoader, TensorDataset

# Import selected names from config instead of importing its complete namespace.
from config import (
    # Pass `BATCH_SIZE` as the next value required by the surrounding call or collection.
    BATCH_SIZE,
    # Pass `DISPLACEMENT_LOSS_WEIGHT` as the next value required by the surrounding call or collection.
    DISPLACEMENT_LOSS_WEIGHT,
    # Pass `DISPLACEMENT_PEAK_WEIGHT` as the next value required by the surrounding call or collection.
    DISPLACEMENT_PEAK_WEIGHT,
    # Pass `EARLY_STOPPING_PATIENCE` as the next value required by the surrounding call or collection.
    EARLY_STOPPING_PATIENCE,
    # Pass `EPOCHS` as the next value required by the surrounding call or collection.
    EPOCHS,
    # Pass `FORCE_LOSS_WEIGHT` as the next value required by the surrounding call or collection.
    FORCE_LOSS_WEIGHT,
    # Pass `FINAL_FINE_TUNE_EPOCHS` as the next value required by the surrounding call or collection.
    FINAL_FINE_TUNE_EPOCHS,
    # Pass `FINAL_FINE_TUNE_LEARNING_RATE` as the next value required by the surrounding call or collection.
    FINAL_FINE_TUNE_LEARNING_RATE,
    # Pass `LEARNING_RATE` as the next value required by the surrounding call or collection.
    LEARNING_RATE,
    # Pass `MINIMUM_IMPROVEMENT` as the next value required by the surrounding call or collection.
    MINIMUM_IMPROVEMENT,
    # Pass `MIN_LEARNING_RATE` as the next value required by the surrounding call or collection.
    MIN_LEARNING_RATE,
    # Pass `SCHEDULER_FACTOR` as the next value required by the surrounding call or collection.
    SCHEDULER_FACTOR,
    # Pass `SCHEDULER_PATIENCE` as the next value required by the surrounding call or collection.
    SCHEDULER_PATIENCE,
    # Pass `WEIGHT_DECAY` as the next value required by the surrounding call or collection.
    WEIGHT_DECAY,
# Close the current function call, tuple, or grouped expression.
)


# Build mini-batches and define the two-output loss
def make_loaders(data):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Create training and validation mini-batches."""

    # Create the training DataLoader that groups normalized training windows into shuffled mini-batches.
    train_loader = DataLoader(
        # Call `TensorDataset`; the following indented continuation lines provide its arguments.
        TensorDataset(
            # Select `data["x_train"]` from the current array, tensor, table, or dictionary.
            data["x_train"],
            # Select `data["y_train"]` from the current array, tensor, table, or dictionary.
            data["y_train"],
        # Close the current function call, tuple, or grouped expression.
        ),
        # Set the number of windows processed in one mini-batch; the configured value controls the memory/speed tradeoff.
        batch_size=BATCH_SIZE,
        # Choose whether DataLoader randomizes sample order: True for training, False for deterministic validation or testing.
        shuffle=True,
    # Close the current function call, tuple, or grouped expression.
    )

    # Create the validation DataLoader that evaluates every validation window without shuffling.
    validation_loader = DataLoader(
        # Call `TensorDataset`; the following indented continuation lines provide its arguments.
        TensorDataset(
            # Select `data["x_validation"]` from the current array, tensor, table, or dictionary.
            data["x_validation"],
            # Select `data["y_validation"]` from the current array, tensor, table, or dictionary.
            data["y_validation"],
        # Close the current function call, tuple, or grouped expression.
        ),
        # Set the number of windows processed in one mini-batch; the configured value controls the memory/speed tradeoff.
        batch_size=BATCH_SIZE,
        # Choose whether DataLoader randomizes sample order: True for training, False for deterministic validation or testing.
        shuffle=False,
    # Close the current function call, tuple, or grouped expression.
    )

    # Return this value to the code that called the current function.
    return train_loader, validation_loader


# Define the accuracy_focused_loss function; its indented lines form the function body.
def accuracy_focused_loss(prediction, target):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Weighted squared error for standardized displacement and force."""

    # Compute the elementwise squared residual for both standardized outputs.
    squared_error = (
        # Subtract the measured targets from predictions to obtain residuals before squaring.
        prediction - target
    # Square each residual so positive and negative errors are penalized and large errors count more.
    ) ** 2

    # The 0.35 setting gives large displacement peaks some extra importance.
    peak_weight = (
        # Start at 1.0 so every sample keeps its full ordinary loss weight before peak emphasis is added.
        1.0
        # Continue the previous expression by adding or concatenating this value.
        + DISPLACEMENT_PEAK_WEIGHT
        # Use absolute standardized displacement (output column 0) to identify large target peaks.
        * torch.abs(target[:, 0])
    # Close the current function call, tuple, or grouped expression.
    )

    # Average the peak-weighted squared error for output column 0, which is displacement.
    displacement_loss = torch.mean(
        # Pass `peak_weight` as the next value required by the surrounding call or collection.
        peak_weight
        # Multiply the previous quantity by this factor to form the current weighted term.
        * squared_error[:, 0]
    # Close the current function call, tuple, or grouped expression.
    )

    # Average the squared error for output column 1, which is force.
    force_loss = torch.mean(
        # Select `squared_error[:, 1]` from the current array, tensor, table, or dictionary.
        squared_error[:, 1]
    # Close the current function call, tuple, or grouped expression.
    )

    # Return this value to the code that called the current function.
    return (
        # Pass `DISPLACEMENT_LOSS_WEIGHT` as the next value required by the surrounding call or collection.
        DISPLACEMENT_LOSS_WEIGHT
        # Multiply the previous quantity by this factor to form the current weighted term.
        * displacement_loss
        # Continue the previous expression by adding or concatenating this value.
        + FORCE_LOSS_WEIGHT
        # Multiply the previous quantity by this factor to form the current weighted term.
        * force_loss
    # Close the current function call, tuple, or grouped expression.
    )


# Train with validation-based learning-rate control and early stopping
def train_model(model, data, device):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Train the model and restore the best validation epoch."""

    # Build both mini-batch iterators and unpack the two returned objects.
    train_loader, validation_loader = make_loaders(data)

    # Create the Adam optimizer that updates the trainable model parameters.
    optimizer = torch.optim.Adam(
        # Call `model.parameters`; the following indented continuation lines provide its arguments.
        model.parameters(),
        # Pass the learning rate that controls the size of each optimizer update.
        lr=LEARNING_RATE,
        # Pass the small L2 penalty used to discourage unnecessarily large weights.
        weight_decay=WEIGHT_DECAY,
    # Close the current function call, tuple, or grouped expression.
    )

    # Create the validation-driven scheduler that reduces the learning rate at a plateau.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # Pass `optimizer` as the next value required by the surrounding call or collection.
        optimizer,
        # Use 'min' because a smaller validation loss represents improvement.
        mode="min",
        # Pass the multiplicative learning-rate reduction factor from the configuration.
        factor=SCHEDULER_FACTOR,
        # Pass the number of plateau epochs allowed before the scheduler changes the learning rate.
        patience=SCHEDULER_PATIENCE,
        # Pass the lower learning-rate limit so the scheduler cannot reduce it indefinitely.
        min_lr=MIN_LEARNING_RATE,
    # Close the current function call, tuple, or grouped expression.
    )

    # Create storage for the loss recorded after each training epoch.
    training_history = []
    # Create storage for the loss recorded after each validation epoch.
    validation_history = []

    # Start the best validation loss at infinity so the first finite loss becomes the initial best.
    best_loss = float("inf")
    # Initialize the index of the best validation epoch before training begins.
    best_epoch = 0
    # Start without saved weights; a deep copy is stored after validation improves.
    best_weights = None
    # Count consecutive validation epochs that do not improve enough.
    epochs_without_improvement = 0

    # Repeat the following indented block once for each item in this iterable.
    for epoch in range(EPOCHS):
        # Training mode enables dropout and parameter updates.
        model.train()
        # Initialize or update the sample-weighted sum used to calculate the epoch's mean training loss.
        training_sum = 0.0

        # Repeat the following indented block once for each item in this iterable.
        for input_batch, target_batch in train_loader:
            # Move the current input mini-batch to the model's CPU or GPU device.
            input_batch = input_batch.to(device)
            # Move the corresponding measured-output mini-batch to the same device.
            target_batch = target_batch.to(device)

            # Clear gradients left from the previous mini-batch because PyTorch accumulates them by default.
            optimizer.zero_grad()  # Clear the previous batch gradients.

            # Run the current input batch through the LSTM to obtain two predicted outputs.
            prediction = model(input_batch)

            # Calculate the weighted displacement-and-force error for the current batch.
            loss = accuracy_focused_loss(
                # Pass `prediction` as the next value required by the surrounding call or collection.
                prediction,
                # Pass `target_batch` as the next value required by the surrounding call or collection.
                target_batch,
            # Close the current function call, tuple, or grouped expression.
            )

            # Backpropagate the loss to calculate gradients for all trainable parameters.
            loss.backward()  # Calculate gradients for every parameter.

            # Limit large gradients before changing the parameters.
            torch.nn.utils.clip_grad_norm_(
                # Call `model.parameters`; the following indented continuation lines provide its arguments.
                model.parameters(),
                # Clip the total gradient norm at 1.0, a conservative recurrent-network stability threshold and tunable heuristic.
                max_norm=1.0,
            # Close the current function call, tuple, or grouped expression.
            )

            # Use Adam and the current gradients to update every trainable parameter once.
            optimizer.step()  # Apply one Adam update.

            # Add this batch's mean loss multiplied by its sample count; this
            # gives an exact epoch mean even when the final batch is smaller.
            training_sum += (
                # Call `loss.item`; the following indented continuation lines provide its arguments.
                loss.item()
                # Multiply the previous quantity by this factor to form the current weighted term.
                * len(input_batch)
            # Close the current function call, tuple, or grouped expression.
            )

        # Divide the accumulated training loss by the number of training samples.
        training_loss = (
            # Pass `training_sum` as the next value required by the surrounding call or collection.
            training_sum
            # Use the expression `/ len(data["x_train"])` as the next part of the surrounding Python statement.
            / len(data["x_train"])
        # Close the current function call, tuple, or grouped expression.
        )

        # Validation checks separate data without changing the model.
        model.eval()
        # Initialize or update the sample-weighted sum used to calculate the epoch's mean validation loss.
        validation_sum = 0.0

        # Open this managed context and release its resources automatically afterward.
        with torch.no_grad():
            # Repeat the following indented block once for each item in this iterable.
            for input_batch, target_batch in validation_loader:
                # Move the current input mini-batch to the model's CPU or GPU device.
                input_batch = input_batch.to(device)
                # Move the corresponding measured-output mini-batch to the same device.
                target_batch = target_batch.to(device)

                # Run the current input batch through the LSTM to obtain two predicted outputs.
                prediction = model(input_batch)

                # Calculate the weighted displacement-and-force error for the current batch.
                loss = accuracy_focused_loss(
                    # Pass `prediction` as the next value required by the surrounding call or collection.
                    prediction,
                    # Pass `target_batch` as the next value required by the surrounding call or collection.
                    target_batch,
                # Close the current function call, tuple, or grouped expression.
                )

                # Add this validation batch's sample-weighted loss to the epoch sum.
                validation_sum += (
                    # Call `loss.item`; the following indented continuation lines provide its arguments.
                    loss.item()
                    # Multiply the previous quantity by this factor to form the current weighted term.
                    * len(input_batch)
                # Close the current function call, tuple, or grouped expression.
                )

        # Divide the accumulated validation loss by the number of validation samples.
        validation_loss = (
            # Pass `validation_sum` as the next value required by the surrounding call or collection.
            validation_sum
            # Use the expression `/ len(data["x_validation"])` as the next part of the surrounding Python statement.
            / len(data["x_validation"])
        # Close the current function call, tuple, or grouped expression.
        )

        # Call `training_history.append`; the following indented continuation lines provide its arguments.
        training_history.append(training_loss)
        # Call `validation_history.append`; the following indented continuation lines provide its arguments.
        validation_history.append(validation_loss)

        # Lower the learning rate if validation has stopped improving.
        scheduler.step(validation_loss)

        # Evaluate this condition and run the following indented block only when it is true.
        if (
            # Pass `validation_loss` as the next value required by the surrounding call or collection.
            validation_loss
            # Use the expression `< best_loss` as the next part of the surrounding Python statement.
            < best_loss
            # Use the expression `- MINIMUM_IMPROVEMENT` as the next part of the surrounding Python statement.
            - MINIMUM_IMPROVEMENT
        # Begin the indented block controlled by this statement.
        ):
            # Start the best validation loss at infinity so the first finite loss becomes the initial best.
            best_loss = validation_loss
            # Initialize the index of the best validation epoch before training begins.
            best_epoch = epoch + 1
            # Start without saved weights; a deep copy is stored after validation improves.
            best_weights = copy.deepcopy(
                # Call `model.state_dict`; the following indented continuation lines provide its arguments.
                model.state_dict()
            # Close the current function call, tuple, or grouped expression.
            )
            # Count consecutive validation epochs that do not improve enough.
            epochs_without_improvement = 0
        # Run the following block when the preceding condition was false.
        else:
            # Update `epochs_without_improvement` in place: add `1`.
            epochs_without_improvement += 1

        # Read the learning rate from Adam's first and only parameter group.
        current_learning_rate = (
            # Index 0 selects that parameter group and `lr` selects its rate.
            optimizer.param_groups[0]["lr"]
        # Close the current function call, tuple, or grouped expression.
        )

        # Print this progress or result message in the terminal.
        print(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"train={training_loss:.6f} | "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"validation={validation_loss:.6f} | "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"lr={current_learning_rate:.2e}",
            # Pass `True` as the `flush` argument of the surrounding function call.
            flush=True,
        # Close the current function call, tuple, or grouped expression.
        )

        # Evaluate this condition and run the following indented block only when it is true.
        if (
            # Pass `epochs_without_improvement` as the next value required by the surrounding call or collection.
            epochs_without_improvement
            # Use the expression `>= EARLY_STOPPING_PATIENCE` as the next part of the surrounding Python statement.
            >= EARLY_STOPPING_PATIENCE
        # Begin the indented block controlled by this statement.
        ):
            # Print this progress or result message in the terminal.
            print(
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                "Early stopping. "
                # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
                f"Best epoch = {best_epoch}",
                # Pass `True` as the `flush` argument of the surrounding function call.
                flush=True,
            # Close the current function call, tuple, or grouped expression.
            )
            # Leave the current loop immediately.
            break

    # Evaluate this condition and run the following indented block only when it is true.
    if best_weights is None:
        # Stop this operation and report the stated error condition.
        raise RuntimeError(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            "Training did not produce a valid model state."
        # Close the current function call, tuple, or grouped expression.
        )

    # Continue with the best validation epoch instead of the last epoch.
    model.load_state_dict(best_weights)

    # Return this value to the code that called the current function.
    return (
        # Call `np.asarray`; the following indented continuation lines provide its arguments.
        np.asarray(training_history),
        # Call `np.asarray`; the following indented continuation lines provide its arguments.
        np.asarray(validation_history),
        # Pass `best_epoch` as the next value required by the surrounding call or collection.
        best_epoch,
    # Close the current function call, tuple, or grouped expression.
    )


# Refit briefly on training and validation after model selection
def fine_tune_on_all_development_data(model, data, device):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Use training and validation windows for final low-rate updates."""

    # Test windows are deliberately not included here.
    all_inputs = torch.cat(
        # Begin the grouped expression or collection continued on the following lines.
        [
            # Select `data["x_train"]` from the current array, tensor, table, or dictionary.
            data["x_train"],
            # Select `data["x_validation"]` from the current array, tensor, table, or dictionary.
            data["x_validation"],
        # Close the current list or index expression.
        ],
        # Select the tensor dimension along which values are concatenated or reduced.
        dim=0,
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate `torch.cat(` and store the result in `all_targets` for the following steps.
    all_targets = torch.cat(
        # Begin the grouped expression or collection continued on the following lines.
        [
            # Select `data["y_train"]` from the current array, tensor, table, or dictionary.
            data["y_train"],
            # Select `data["y_validation"]` from the current array, tensor, table, or dictionary.
            data["y_validation"],
        # Close the current list or index expression.
        ],
        # Select the tensor dimension along which values are concatenated or reduced.
        dim=0,
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate `DataLoader(` and store the result in `loader` for the following steps.
    loader = DataLoader(
        # Call `TensorDataset`; the following indented continuation lines provide its arguments.
        TensorDataset(
            # Pass `all_inputs` as the next value required by the surrounding call or collection.
            all_inputs,
            # Pass `all_targets` as the next value required by the surrounding call or collection.
            all_targets,
        # Close the current function call, tuple, or grouped expression.
        ),
        # Use 512 windows per prediction batch. No gradients are stored here,
        # so a batch larger than the 128 training batch improves speed without
        # changing predictions; reduce it only if memory is insufficient.
        batch_size=BATCH_SIZE,
        # Choose whether DataLoader randomizes sample order: True for training, False for deterministic validation or testing.
        shuffle=True,
    # Close the current function call, tuple, or grouped expression.
    )

    # Create the Adam optimizer that updates the trainable model parameters.
    optimizer = torch.optim.Adam(
        # Call `model.parameters`; the following indented continuation lines provide its arguments.
        model.parameters(),
        # Pass the learning rate that controls the size of each optimizer update.
        lr=FINAL_FINE_TUNE_LEARNING_RATE,
        # Pass the small L2 penalty used to discourage unnecessarily large weights.
        weight_decay=WEIGHT_DECAY,
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate `[]` and store the result in `fine_tune_history` for the following steps.
    fine_tune_history = []

    # Repeat the following indented block once for each item in this iterable.
    for epoch in range(
        # Pass `FINAL_FINE_TUNE_EPOCHS` as the next value required by the surrounding call or collection.
        FINAL_FINE_TUNE_EPOCHS
    # Begin the indented block controlled by this statement.
    ):
        # Switch the model to training mode, which enables dropout.
        model.train()
        # Evaluate `0.0` and store the result in `running_sum` for the following steps.
        running_sum = 0.0

        # Repeat the following indented block once for each item in this iterable.
        for input_batch, target_batch in loader:
            # Move the current input mini-batch to the model's CPU or GPU device.
            input_batch = input_batch.to(device)
            # Move the corresponding measured-output mini-batch to the same device.
            target_batch = target_batch.to(device)

            # Clear gradients left from the previous mini-batch because PyTorch accumulates them by default.
            optimizer.zero_grad()

            # Run the current input batch through the LSTM to obtain two predicted outputs.
            prediction = model(input_batch)

            # Calculate the weighted displacement-and-force error for the current batch.
            loss = accuracy_focused_loss(
                # Pass `prediction` as the next value required by the surrounding call or collection.
                prediction,
                # Pass `target_batch` as the next value required by the surrounding call or collection.
                target_batch,
            # Close the current function call, tuple, or grouped expression.
            )

            # Backpropagate the loss to calculate gradients for all trainable parameters.
            loss.backward()

            # Call `torch.nn.utils.clip_grad_norm_`; the following indented continuation lines provide its arguments.
            torch.nn.utils.clip_grad_norm_(
                # Call `model.parameters`; the following indented continuation lines provide its arguments.
                model.parameters(),
                # Clip the total gradient norm at 1.0, a conservative recurrent-network stability threshold and tunable heuristic.
                max_norm=1.0,
            # Close the current function call, tuple, or grouped expression.
            )

            # Use Adam and the current gradients to update every trainable parameter once.
            optimizer.step()

            # Update `running_sum` in place: add `(`.
            running_sum += (
                # Call `loss.item`; the following indented continuation lines provide its arguments.
                loss.item()
                # Multiply the previous quantity by this factor to form the current weighted term.
                * len(input_batch)
            # Close the current function call, tuple, or grouped expression.
            )

        # Evaluate `(` and store the result in `epoch_loss` for the following steps.
        epoch_loss = (
            # Pass `running_sum` as the next value required by the surrounding call or collection.
            running_sum
            # Use the expression `/ len(all_inputs)` as the next part of the surrounding Python statement.
            / len(all_inputs)
        # Close the current function call, tuple, or grouped expression.
        )

        # Call `fine_tune_history.append`; the following indented continuation lines provide its arguments.
        fine_tune_history.append(
            # Pass `epoch_loss` as the next value required by the surrounding call or collection.
            epoch_loss
        # Close the current function call, tuple, or grouped expression.
        )

        # Print this progress or result message in the terminal.
        print(
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"Final fine-tune "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"{epoch + 1:02d}/"
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"{FINAL_FINE_TUNE_EPOCHS} | "
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"development loss="
            # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
            f"{epoch_loss:.6f}",
            # Pass `True` as the `flush` argument of the surrounding function call.
            flush=True,
        # Close the current function call, tuple, or grouped expression.
        )

    # Return this value to the code that called the current function.
    return np.asarray(
        # Pass `fine_tune_history` as the next value required by the surrounding call or collection.
        fine_tune_history
    # Close the current function call, tuple, or grouped expression.
    )


# Run prediction and calculate physical-unit metrics
@torch.no_grad()
# Define the predict function; its indented lines form the function body.
def predict(model, x, device):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Predict normalized outputs in memory-safe batches."""

    # Switch the model to evaluation mode, which disables dropout.
    model.eval()

    # Evaluate `DataLoader(` and store the result in `loader` for the following steps.
    loader = DataLoader(
        # Call `TensorDataset`; the following indented continuation lines provide its arguments.
        TensorDataset(x),
        # Set the number of windows processed in one mini-batch; the configured value controls the memory/speed tradeoff.
        batch_size=512,
        # Choose whether DataLoader randomizes sample order: True for training, False for deterministic validation or testing.
        shuffle=False,
    # Close the current function call, tuple, or grouped expression.
    )

    # Evaluate `[]` and store the result in `prediction_parts` for the following steps.
    prediction_parts = []

    # Repeat the following indented block once for each item in this iterable.
    for (input_batch,) in loader:
        # Call `prediction_parts.append`; the following indented continuation lines provide its arguments.
        prediction_parts.append(
            # Call `model`; the following indented continuation lines provide its arguments.
            model(
                # Call `input_batch.to`; the following indented continuation lines provide its arguments.
                input_batch.to(device)
            # Use the expression `).cpu().numpy()` as the next part of the surrounding Python statement.
            ).cpu().numpy()
        # Close the current function call, tuple, or grouped expression.
        )

    # Return this value to the code that called the current function.
    return np.concatenate(
        # Pass `prediction_parts` as the next value required by the surrounding call or collection.
        prediction_parts,
        # Concatenate along axis 0, the sample/window dimension, while keeping
        # the two output columns unchanged.
        axis=0,
    # Close the current function call, tuple, or grouped expression.
    )


# Define the calculate_metrics function; its indented lines form the function body.
def calculate_metrics(measured, predicted):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Return MSE, RMSE, MAE, R2, and Fit percentage."""

    # Subtract prediction from measurement to obtain the pointwise prediction error.
    error = measured - predicted

    # Calculate mean squared error, which is the average squared prediction error.
    mse = float(np.mean(error ** 2))
    # Take the square root of MSE to return the error to the output's physical unit.
    rmse = float(np.sqrt(mse))
    # Calculate the mean absolute error, which is less dominated by isolated peaks than RMSE.
    mae = float(np.mean(np.abs(error)))

    # Calculate measured variation around the mean, the R-squared denominator.
    r2_denominator = float(
        # Call `np.sum`; the following indented continuation lines provide its arguments.
        np.sum(
            # Begin the grouped expression or collection continued on the following lines.
            (measured - measured.mean()) ** 2
        # Close the current function call, tuple, or grouped expression.
        )
    # Close the current function call, tuple, or grouped expression.
    )

    # Calculate the coefficient of determination relative to variation around the measured mean.
    r2 = (
        # Perfect prediction has zero residual ratio, so R-squared starts at one.
        1
        # Subtract the total squared prediction-error energy.
        - float(np.sum(error ** 2))
        # Normalize error by measured variation so R-squared is dimensionless.
        / r2_denominator
        # Evaluate this condition and run the following indented block only when it is true.
        if r2_denominator > 0
        # A constant measured signal has no R-squared denominator, so use NaN.
        else np.nan
    # Close the current function call, tuple, or grouped expression.
    )

    # Calculate the measured-deviation norm used by the Fit metric.
    fit_denominator = float(
        # Call `np.linalg.norm`; the following indented continuation lines provide its arguments.
        np.linalg.norm(
            # Use the expression `measured - measured.mean()` as the next part of the surrounding Python statement.
            measured - measured.mean()
        # Close the current function call, tuple, or grouped expression.
        )
    # Close the current function call, tuple, or grouped expression.
    )

    # Calculate the percentage fit from error energy relative to measured variation.
    fit = (
        # Multiply the dimensionless ratio by 100 to report Fit as percent.
        100
        # Multiply the previous quantity by this factor to form the current weighted term.
        * (
            # A zero error norm should give a Fit ratio of one.
            1
            # Subtract the prediction-error norm.
            - float(np.linalg.norm(error))
            # Normalize by the measured signal's deviation norm.
            / fit_denominator
        # Close the current function call, tuple, or grouped expression.
        )
        # Evaluate this condition and run the following indented block only when it is true.
        if fit_denominator > 0
        # A constant measured signal has no Fit denominator, so use NaN.
        else np.nan
    # Close the current function call, tuple, or grouped expression.
    )

    # Return this value to the code that called the current function.
    return {
        # Store the 'MSE' field in the current dictionary.
        "MSE": mse,
        # Store the 'RMSE' field in the current dictionary.
        "RMSE": rmse,
        # Store the 'MAE' field in the current dictionary.
        "MAE": mae,
        # Store the 'R2' field in the current dictionary.
        "R2": r2,
        # Store the 'Fit_percent' field in the current dictionary.
        "Fit_percent": fit,
    # Close the current dictionary.
    }
