# Run as uv run -m model.continue_train
import argparse
from .project_config import WANDB_DEFAULTS, DEFAULT_MODEL_NAME
from .harness import TrainingOverrides, ModelTrainerBase, ModelBase, WandbHelper, TrainingState
import wandb
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training a model')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        '--end-epoch',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size for training (default: None, uses model default)'
    )
    parser.add_argument(
        '--batch-limit',
        type=int,
        default=None,
        help='Override the batch count per epoch/validation (default: None, uses full batch)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override the default learning rate for training (default: None, uses model default)'
    )
    parser.add_argument(
        '--ignore-dataset-cache',
        action='store_true',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Set a random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        '--print-after-batches',
        type=int,
        default=None,
        help="Print a log message after N batches (default: None, uses model default)",
    )
    parser.add_argument(
        '--validate-after-epochs',
        type=int,
        default=1,
        help="Run validation after every N epochs (default: 1)",
    )
    parser.add_argument(
        '--immediate-validation',
        action='store_true',
        help='Run validation immediately after loading the model'
    )
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping during training'
    )
    wandb_arguments_helper = WANDB_DEFAULTS.add_arg_parser_arguments(
        parser,
        add_model_source_args=True,
        add_upload_model_args=True,
    )
    args = parser.parse_args()

    wandb_args = wandb_arguments_helper.handle_arguments(args)

    overrides = TrainingOverrides(
        override_to_epoch=args.end_epoch,
        override_batch_size=args.batch_size,
        override_batch_limit=args.batch_limit,
        override_learning_rate=args.learning_rate,
        validate_after_epochs=args.validate_after_epochs,
        recalculate_running_loss_after_batches=args.print_after_batches,
        seed=args.seed,
        use_dataset_cache=not args.ignore_dataset_cache,
    )

    try:
        model_file_path = wandb_args.load_model_get_filepath()

        model, trainer_state_dict, training_config_dict = ModelBase.load_advanced(model_path=model_file_path)

        wandb_args.start_run_if_required(
            model=model,
            training_config=training_config_dict,
            trainer_state_dict=trainer_state_dict,
            overrides=overrides,
        )

        trainer = ModelTrainerBase.load(
            model=model,
            config=training_config_dict,
            state=TrainingState.from_dict(trainer_state_dict),
            overrides=overrides,
        )

        if args.immediate_validation:
            print("Immediate validation enabled, running validation before training:")
            trainer.validate()

        if trainer.epoch >= trainer.config.epochs:
            raise ValueError(f"Model {trainer.model.model_name} has already finished training its {trainer.config.epochs} epochs. Use --end-epoch <new_total_epochs> to continue training.")

        results = trainer.train()

        wandb_args.upload_latest_and_best_model_snapshots(trainer.model.model_name)
    finally:
        if wandb.run is not None:
            wandb.finish()



        
