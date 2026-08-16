import copy


import numpy as np

import torch

from torch.utils.data import DataLoader, TensorDataset


from config import (

    BATCH_SIZE,

    DISPLACEMENT_LOSS_WEIGHT,

    DISPLACEMENT_PEAK_WEIGHT,

    EARLY_STOPPING_PATIENCE,

    EPOCHS,

    FORCE_LOSS_WEIGHT,

    FINAL_FINE_TUNE_EPOCHS,

    FINAL_FINE_TUNE_LEARNING_RATE,

    LEARNING_RATE,

    MINIMUM_IMPROVEMENT,

    MIN_LEARNING_RATE,

    SCHEDULER_FACTOR,

    SCHEDULER_PATIENCE,

    WEIGHT_DECAY,

)


def make_loaders(data):


    train_loader = DataLoader(

        TensorDataset(

            data["x_train"],

            data["y_train"],

        ),

        batch_size=BATCH_SIZE,

        shuffle=True,

        pin_memory=torch.cuda.is_available(),

    )


    validation_loader = DataLoader(

        TensorDataset(

            data["x_validation"],

            data["y_validation"],

        ),

        batch_size=BATCH_SIZE,

        shuffle=False,

        pin_memory=torch.cuda.is_available(),

    )


    return train_loader, validation_loader


def accuracy_focused_loss(prediction, target):


    squared_error = (

        prediction - target

    ) ** 2


    peak_weight = (

        1.0

        + DISPLACEMENT_PEAK_WEIGHT

        * torch.abs(target[:, 0])

    )


    displacement_loss = torch.mean(

        peak_weight

        * squared_error[:, 0]

    )


    force_loss = torch.mean(

        squared_error[:, 1]

    )


    return (

        DISPLACEMENT_LOSS_WEIGHT

        * displacement_loss

        + FORCE_LOSS_WEIGHT

        * force_loss

    )


def train_model(model, data, device):


    train_loader, validation_loader = make_loaders(data)


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

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

    best_weights = None

    epochs_without_improvement = 0


    for epoch in range(EPOCHS):

        model.train()

        training_sum = 0.0


        for input_batch, target_batch in train_loader:

            input_batch = input_batch.to(

                device,

                non_blocking=device.type == "cuda",

            )

            target_batch = target_batch.to(

                device,

                non_blocking=device.type == "cuda",

            )


            optimizer.zero_grad()


            prediction = model(input_batch)


            loss = accuracy_focused_loss(

                prediction,

                target_batch,

            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0,

            )


            optimizer.step()


            training_sum += (

                loss.item()

                * len(input_batch)

            )


        training_loss = (

            training_sum

            / len(data["x_train"])

        )


        model.eval()

        validation_sum = 0.0


        with torch.no_grad():

            for input_batch, target_batch in validation_loader:

                input_batch = input_batch.to(

                    device,

                    non_blocking=device.type == "cuda",

                )

                target_batch = target_batch.to(

                    device,

                    non_blocking=device.type == "cuda",

                )


                prediction = model(input_batch)


                loss = accuracy_focused_loss(

                    prediction,

                    target_batch,

                )


                validation_sum += (

                    loss.item()

                    * len(input_batch)

                )


        validation_loss = (

            validation_sum

            / len(data["x_validation"])

        )


        training_history.append(training_loss)

        validation_history.append(validation_loss)


        scheduler.step(validation_loss)


        if (

            validation_loss

            < best_loss

            - MINIMUM_IMPROVEMENT

        ):

            best_loss = validation_loss

            best_epoch = epoch + 1

            best_weights = copy.deepcopy(

                model.state_dict()

            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        current_learning_rate = (

            optimizer.param_groups[0]["lr"]

        )


        print(

            f"Epoch {epoch + 1:03d}/{EPOCHS} | "

            f"train={training_loss:.6f} | "

            f"validation={validation_loss:.6f} | "

            f"lr={current_learning_rate:.2e}",

            flush=True,

        )


        if (

            epochs_without_improvement

            >= EARLY_STOPPING_PATIENCE

        ):

            print(

                "Early stopping. "

                f"Best epoch = {best_epoch}",

                flush=True,

            )

            break


    if best_weights is None:

        raise RuntimeError(

            "Training did not produce a valid model state."

        )


    model.load_state_dict(best_weights)


    return (

        np.asarray(training_history),

        np.asarray(validation_history),

        best_epoch,

    )


def fine_tune_on_all_development_data(model, data, device):


    all_inputs = torch.cat(

        [

            data["x_train"],

            data["x_validation"],

        ],

        dim=0,

    )


    all_targets = torch.cat(

        [

            data["y_train"],

            data["y_validation"],

        ],

        dim=0,

    )


    loader = DataLoader(

        TensorDataset(

            all_inputs,

            all_targets,

        ),


        batch_size=BATCH_SIZE,

        shuffle=True,

        pin_memory=torch.cuda.is_available(),

    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=FINAL_FINE_TUNE_LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )


    fine_tune_history = []


    for epoch in range(

        FINAL_FINE_TUNE_EPOCHS

    ):

        model.train()

        running_sum = 0.0


        for input_batch, target_batch in loader:

            input_batch = input_batch.to(

                device,

                non_blocking=device.type == "cuda",

            )

            target_batch = target_batch.to(

                device,

                non_blocking=device.type == "cuda",

            )


            optimizer.zero_grad()


            prediction = model(input_batch)


            loss = accuracy_focused_loss(

                prediction,

                target_batch,

            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0,

            )


            optimizer.step()


            running_sum += (

                loss.item()

                * len(input_batch)

            )


        epoch_loss = (

            running_sum

            / len(all_inputs)

        )


        fine_tune_history.append(

            epoch_loss

        )


        print(

            f"Final fine-tune "

            f"{epoch + 1:02d}/"

            f"{FINAL_FINE_TUNE_EPOCHS} | "

            f"development loss="

            f"{epoch_loss:.6f}",

            flush=True,

        )


    return np.asarray(

        fine_tune_history

    )


@torch.no_grad()


def predict(model, x, device):


    model.eval()


    loader = DataLoader(

        TensorDataset(x),

        batch_size=512,

        shuffle=False,

        pin_memory=torch.cuda.is_available(),

    )


    prediction_parts = []


    for (input_batch,) in loader:

        prediction_parts.append(

            model(

                input_batch.to(

                    device,

                    non_blocking=device.type == "cuda",

                )

            ).cpu().numpy()

        )


    return np.concatenate(

        prediction_parts,


        axis=0,

    )


def calculate_metrics(measured, predicted):


    error = measured - predicted


    mse = float(np.mean(error ** 2))

    rmse = float(np.sqrt(mse))

    mae = float(np.mean(np.abs(error)))


    r2_denominator = float(

        np.sum(

            (measured - measured.mean()) ** 2

        )

    )


    r2 = (

        1

        - float(np.sum(error ** 2))

        / r2_denominator

        if r2_denominator > 0

        else np.nan

    )


    fit_denominator = float(

        np.linalg.norm(

            measured - measured.mean()

        )

    )


    fit = (

        100

        * (

            1

            - float(np.linalg.norm(error))

            / fit_denominator

        )

        if fit_denominator > 0

        else np.nan

    )


    return {

        "MSE": mse,

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2,

        "Fit_percent": fit,

    }
