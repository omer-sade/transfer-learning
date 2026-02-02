import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from dataModule import FlowersDataModule
from yolov5 import YOLOv5Classifier

IMG_DIR = "/home/zloofe/flower2/data/jpg"
MAT_FILE = "/home/zloofe/flower2/data/imagelabels.mat"
N_TRIALS = 15
TUNE_EPOCHS = 8


def objective(trial: optuna.trial.Trial):
    """
    Optuna will run this function multiple times with different parameters.
    """

    # 1. Ask Optuna for hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 128])

    # Preprocessing params
    rotation = trial.suggest_categorical("rotation", [0, 15, 30])
    flip_p = trial.suggest_categorical("flip_prob", [0.0, 0.5])

    # 2. Setup DataModule with the suggested params
    dm = FlowersDataModule(
        data_dir=IMG_DIR,
        mat_file=MAT_FILE,
        batch_size=batch_size,
        rotation_deg=rotation,  # Passing the tuning param
        flip_prob=flip_p  # Passing the tuning param
    )
    dm.setup()

    # 3. Setup Model
    model = YOLOv5Classifier(
        num_classes=102,
        lr=lr
    )

    # 4. Logger & Pruning Callback
    # The PruningCallback stops unpromising trials early to save time
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    logger = TensorBoardLogger("tb_logs0", name="yolo_tuning", version=f"trial_{trial.number}")

    trainer = pl.Trainer(
        max_epochs=TUNE_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        enable_checkpointing=False,  # Disable checkpoints during tuning to save disk space
        callbacks=[pruning_callback],
        log_every_n_steps=10
    )

    # 5. Train
    trainer.fit(model, datamodule=dm)

    # 6. Return the metric to minimize
    return trainer.callback_metrics["val_loss"].item()


def main():
    print("--- Starting Hyperparameter & Preprocessing Tuning ---")

    # Create Study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 30)
    print(f"Best Trial Found: {study.best_trial.number}")
    print(f"Best Val Loss: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print("=" * 30 + "\n")

    # --- Re-train with Best Parameters ---
    print("--- Starting Full Training with Best Parameters ---")
    best_params = study.best_params

    dm = FlowersDataModule(
        data_dir=IMG_DIR,
        mat_file=MAT_FILE,
        batch_size=best_params["batch_size"],
        rotation_deg=best_params["rotation"],
        flip_prob=best_params["flip_prob"]
    )
    dm.setup()

    model = YOLOv5Classifier(
        num_classes=102,
        lr=best_params["lr"]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_yolo_best',
        filename='best-tuned-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=15,  # Full training epochs
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger("tb_logs0", name="yolov5_final")
    )

    trainer.fit(model, datamodule=dm)

    print("--- Testing Best Model ---")
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == '__main__':
    main()