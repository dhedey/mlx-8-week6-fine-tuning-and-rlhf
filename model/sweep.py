#!/usr/bin/env python3
# Run as uv run -m model.sweep

import wandb
import os

import model.models as models
from .harness import ModelBase, select_device, WandbHelper, WandbArguments, TrainingOverrides

from .project_config import WANDB_DEFAULTS, DEFINED_MODELS, DEFAULT_MODEL_NAME, SweepDefinition, ModelDefinition

def load_sweep_definition(model_name, sweep_config_name) -> tuple[ModelDefinition, SweepDefinition]:
    if model_name not in DEFINED_MODELS:
        raise ValueError(f"Model '{model_name}' is not defined. The choices are: {list(DEFINED_MODELS.keys())}")
    model_definition = DEFINED_MODELS[model_name]

    if sweep_config_name not in model_definition.sweep_configs:
        raise ValueError(f"Model '{model_name}' does not define sweep config '{sweep_config_name}'")

    sweep_definition = model_definition.sweep_configs[sweep_config_name]

    # Set some programmatic parameters on the sweep which we will load up later
    if "name" not in sweep_definition.config:
        sweep_definition.config["name"] = f"{model_name}::{sweep_config_name}"
    sweep_definition.config["parameters"]["base_model_name"] = { "value": model_name }
    sweep_definition.config["parameters"]["sweep_config_name"] = { "value": sweep_config_name }

    return model_definition, sweep_definition

def sweep_single_run(sweep_id, wandb_arguments: WandbArguments, enable_early_stopping: bool, early_stopping_patience: int):
    try:
        # We have to init before we can read the sweep config...
        # But we can update the config later with our standard fields, once we can compute them all!
        wandb.init()

        model_name = wandb.config.base_model_name
        sweep_config_name = wandb.config.sweep_config_name

        print(f"Loading sweep parametrization mapper from config {sweep_config_name} under model {model_name}")

        model_definition, sweep_definition = load_sweep_definition(model_name, sweep_config_name)

        model_config = sweep_definition.model_config_mapper(wandb.config)
        training_config = sweep_definition.training_config_mapper(wandb.config)

        sweep_model_name = f"{model_name}__sweep-{sweep_id}__run-{wandb.run.id}"

        print(f"Model will be known as \"{sweep_model_name}\"")

        device = select_device()
        model = model_definition.model(
            model_name=sweep_model_name,
            config=model_config,
        ).to(device)

        overrides = TrainingOverrides(
            print_detailed_parameter_counts=True,
            # TODO: Add others, via override handling
        )
        # TODO: Make this into an (automated) override
        training_config.early_stopping = enable_early_stopping
        training_config.early_stopping_patience = early_stopping_patience

        wandb_arguments.update_run_name_and_config(
            run_id=wandb.run.id,
            model=model,
            training_config=training_config.to_dict(),
            trainer_state_dict=None,
            overrides=overrides,
            sweep_id=sweep_id,
        )
        print()

        trainer = model_definition.trainer(
            model=model,
            config=training_config,
            overrides=overrides,
        )

        results = trainer.train()
        
        # Log final metrics
        wandb.log({
            "final_train_average_loss": results.last_training_epoch.average_loss,
            "final_validation_average_loss": results.last_validation.train_comparable_loss,
            "best_train_average_loss": results.best_training_epoch.average_loss,
            "final_validation_objective": results.last_validation.objective,
            "best_validation_objective": results.best_validation.objective,
            "total_epochs": results.total_epochs,
        })

        wandb_arguments.upload_latest_and_best_model_snapshots(sweep_model_name)
        
        print(f"✅ Sweep run completed!")
        
    except Exception as e:
        print(f"❌ Sweep run failed: {e}")
        # Log the failure
        wandb.log({"status": "failed", "error": str(e)})
        raise
    
    finally:
        WandbHelper.finish()


def main():
    """
    Main function with different sweep options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter sweeps')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        '--sweep-config',
        type=str,
        default="default",
        help="The name of the sweep configuration to use under the model definition (default is \"default\")",
    )
    parser.add_argument('--count', type=int, default=20,
                        help='Number of sweep runs (default: 20)')
    parser.add_argument('--sweep-id', type=str,
                        help='Join existing sweep by ID instead of creating new one')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show the configuration without running')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping for all sweep runs')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Number of epochs to wait before early stopping (default: 5)')
    wandb_arguments_helper = WANDB_DEFAULTS.add_arg_parser_arguments(
        parser,
        add_model_source_args=False,
        add_upload_model_args=True,
        require_wandb=True,
    )
    
    args = parser.parse_args()

    wandb_arguments = wandb_arguments_helper.handle_arguments(args)

    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")

    # Run sweep
    run_function = lambda sweep_id: sweep_single_run(sweep_id, wandb_arguments, args.early_stopping, args.early_stopping_patience)

    if args.sweep_id:
        if args.dry_run:
            raise ValueError("Dry run mode cannot be used with an existing sweep id")
        wandb_arguments.continue_sweep(
            args.sweep_id,
            function=run_function,
            count=args.count,
        )
        return

    _, sweep_definition = load_sweep_definition(args.model, args.sweep_config)
    config = sweep_definition.config

    if args.dry_run:
        print("\n🔍 Sweep configuration:")
        import json
        print(json.dumps(config, indent=2))
        return

    wandb_arguments.run_new_sweep(
        config,
        function=run_function,
        count=args.count,
    )

if __name__ == '__main__':
    main()
