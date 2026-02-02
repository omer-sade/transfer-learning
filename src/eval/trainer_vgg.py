import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from dataModule import FlowersDataModule
from vgg import VGG19Classifier

# --- Constants ---
IMG_DIR = "/home/zloofe/flower2/data/jpg"
MAT_FILE = "/home/zloofe/flower2/data/imagelabels.mat"
N_TRIALS = 15
TUNE_EPOCHS = 8
FULL_EPOCHS = 8


def objective(trial: optuna.trial.Trial):
    """
    Optuna objective function to tune VGG19 hyperparameters.
    """
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # VGG usually likes lower LR than YOLO
    batch_size = trial.suggest_categorical("batch_size", [32, 64])  # Adjust based on GPU memory

    # Preprocessing / Augmentation params
    rotation = trial.suggest_categorical("rotation", [0, 15, 30])
    flip_p = trial.suggest_categorical("flip_prob", [0.0, 0.5])

    # 2. Init DataModule with suggested params
    dm = FlowersDataModule(
        data_dir=IMG_DIR,
        mat_file=MAT_FILE,
        batch_size=batch_size,
        rotation_deg=rotation,
        flip_prob=flip_p
    )
    dm.setup()


    model = VGG19Classifier(
        num_classes=102,
        learning_rate=lr,
    )

    # 4. Logger & Pruning Callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    logger = TensorBoardLogger("tb_logs1", name="vgg_tuning", version=f"trial_{trial.number}")

    trainer = pl.Trainer(
        max_epochs=TUNE_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        enable_checkpointing=False,  # Save space
        callbacks=[pruning_callback],
        log_every_n_steps=10
    )

    # 5. Train
    try:
        trainer.fit(model, datamodule=dm)
    except Exception as e:
        # Handle potential VRAM OOM errors gracefully by pruning the trial
        print(f"Trial failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    # 6. Return metric
    return trainer.callback_metrics["val_loss"].item()


def main():
    print("--- Starting VGG19 Hyperparameter Tuning ---")

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 30)
    print(f"Best Trial Found: {study.best_trial.number}")
    print(f"Best Val Loss: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print("=" * 30 + "\n")

    # --- 2. Re-train with Best Parameters ---
    print("--- Starting Full Training with Best Parameters ---")
    best_params = study.best_params

    # Setup DataModule with best params
    dm = FlowersDataModule(
        data_dir=IMG_DIR,
        mat_file=MAT_FILE,
        batch_size=best_params["batch_size"],
        rotation_deg=best_params["rotation"],
        flip_prob=best_params["flip_prob"]
    )
    dm.setup()

    # Setup Model with best LR and FULL epochs
    model = VGG19Classifier(
        num_classes=102,
        learning_rate=best_params["lr"],
    )

    # Callbacks for final training
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_vgg_best',
        filename='best-vgg-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=FULL_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger("tb_logs0", name="vgg19_final")
    )

    trainer.fit(model, datamodule=dm)

    print("--- Testing Best Model ---")
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == '__main__':
    main()