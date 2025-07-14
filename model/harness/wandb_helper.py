from typing import Optional, Callable

from . import TrainingOverrides
from .base_trainer import TrainingState, ModelTrainerBase
from .base_model import ModelBase
from dataclasses import dataclass
import os
import wandb

class WandbHelper:
    @staticmethod
    def is_running() -> bool:
        return wandb.run is not None

    @staticmethod
    def finish() -> None:
        if WandbHelper.is_running():
            wandb.finish()

    @staticmethod
    def upload_latest_and_best_model_snapshots(model_name: str, model_source_run: Optional[wandb.sdk.wandb_run.Run] = None):
        if not WandbHelper.is_running():
            print("Wandb is not enabled, so skipping of model snapshots")
            return

        WandbHelper.upload_model_snapshot_if_exists(
            model_name=model_name, # The latest snapshot
            artifact_name=f"{model_name}-final",
            artifact_description=f"Final model: {model_name} (run {wandb.run.id})",
            model_source_run=model_source_run,
        )
        # TODO: Compare best validation and only upload if it's better
        WandbHelper.upload_model_snapshot_if_exists(
            model_name=f"{model_name}-best", # The best snapshot
            artifact_name=f"{model_name}-best",
            artifact_description=f"Best validation model: {model_name} (run {wandb.run.id})",
            model_source_run=model_source_run,
        )
        print()

    @staticmethod
    def upload_model_snapshot_if_exists(
        model_name: str,
        artifact_name: str,
        artifact_description: str,
        model_source_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ):
        model_path = ModelBase.resolve_path(model_name)

        if ModelBase.exists(model_path=model_path):
            print(f"Uploading {model_name} from {model_path}")
            WandbHelper.upload_model_snapshot(
                model_path=model_path,
                artifact_name=artifact_name,
                artifact_description=artifact_description,
                model_source_run=model_source_run,
            )
        else:
            print(f"Could not find {model_name} at {model_path}, skipping upload")

    @staticmethod
    def upload_model_snapshot(
        model_path,
        artifact_name,
        artifact_description,
        model_source_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ):
        model, training_state_dict, training_config_dict = ModelBase.load_advanced(model_path=model_path)

        training_state = TrainingState.from_dict(training_state_dict)
        latest_train_results = training_state.latest_training_results
        latest_validation_results = training_state.latest_validation_results

        artifact_metadata = {
            "model_name": model.model_name,
            "model_class": model.__class__.__name__,
            "model_config": model.config.to_dict(),
            "training_config": training_config_dict,
            "training_state": training_state_dict,
            "validation_objective": latest_validation_results.objective,
            "validation_loss": latest_validation_results.train_comparable_loss,
            "train_loss": latest_train_results.average_loss,
            "total_epochs": latest_train_results.epoch,
        }

        if model_source_run is not None:
            artifact_metadata["source_run_id"] = model_source_run.id

        WandbHelper.upload_model_artifact(
            file_path=model_path,
            artifact_name=artifact_name,
            metadata=artifact_metadata,
            description=artifact_description
        )

    @staticmethod
    def upload_model_artifact(
        file_path: str,
        artifact_name: str,
        description: str,
        metadata: dict = None,
        artifact_file_name: str = "model.pt",
    ):
        """
        Upload a model as a wandb artifact.

        Args:
            file_path: Path to the saved model file
            artifact_name: Optional custom artifact name (defaults to model_name)
            description: Description for the artifact
            metadata: Optional metadata dictionary to include with the artifact
            artifact_file_name: Optional file name for the model inside the artifact. (default: model.pt)
        """
        if not os.path.exists(file_path):
            print(f"⚠️ Model file not found: {file_path}")
            return None

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=description,
            metadata=metadata or {}
        )

        # Add the model file
        artifact.add_file(file_path, name=artifact_file_name)

        # Log the artifact
        wandb.log_artifact(artifact)
        print(f"📦 Uploaded model artifact: {artifact_name} containing {artifact_file_name}")

        return artifact

@dataclass
class ModelSourceArguments:
    model_source: str
    model_name: str
    continue_run_id: Optional[str]
    wandb_artifact_name: Optional[str]
    wandb_artifact_type: str
    wandb_artifact_version: str


@dataclass
class WandbArguments:
    enabled: bool
    wandb_entity: str
    wandb_project: str
    no_upload: bool
    model_source: Optional[ModelSourceArguments]
    model_source_run: Optional[wandb.sdk.wandb_run.Run] = None

    def load_model_get_filepath(self) -> str:
        if self.model_source is None:
            raise ValueError("Model source must be specified to call this")

        args = self.model_source

        if args.continue_run_id is not None:
            wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                id=self.model_source.continue_run_id,
                resume="must"
            )
            print(f"🏃 Resumed wandb run {self.model_source.continue_run_id}")

        match self.model_source.model_source:
            case "local":
                model_file_path = ModelBase.model_path(model_name=args.model_name)
            case "wandb":
                if args.wandb_artifact_name:
                    artifact_name = args.wandb_artifact_name
                    print(f"Loading provided artifact name: {artifact_name}")
                elif wandb.run is not None:
                    model_name = wandb.run.config["model_name"]
                    if "sweep_config_name" in wandb.run.config:
                        model_name = f"{model_name}-run-{wandb.run.id}"
                    if model_name != args.model_name:
                        print(f"WARNING: The model name {model_name} from the resumed run will be loaded. The conflicting resolved model name {args.model_name} will be ignored")
                    artifact_name = f"{model_name}-{args.wandb_artifact_type}"
                    print(f"Loading artifact name from the continued run model name: {artifact_name}")
                else:
                    model_name = args.model_name
                    artifact_name = f"{model_name}-{args.wandb_artifact_type}"
                    print(f"Assuming artifact name from the provided model name: {artifact_name}")

                full_artifact_identifier = f"{self.wandb_entity}/{self.wandb_project}/{artifact_name}:{args.wandb_artifact_version}"
                try:
                    if wandb.run is not None:
                        # noinspection PyTypeChecker
                        artifact = wandb.run.use_artifact(full_artifact_identifier)
                    else:
                        api = wandb.Api()
                        artifact = api.artifact(full_artifact_identifier)
                except wandb.errors.CommError as e:
                    print(f"Error: Could not find wandb artifact '{full_artifact_identifier}''. {e}")
                    exit(1)

                self.model_source_run = artifact.logged_by()
                download_path = artifact.download()
                pt_files = [f for f in os.listdir(download_path) if f.endswith('.pt')]
                if not pt_files:
                    print(f"Error: No .pt file found in downloaded artifact at {download_path}")
                    exit(1)
                model_file_path = os.path.join(download_path, pt_files[0])
                print(f"Found downloaded model file: {model_file_path}")
            case _:
                raise ValueError(f"Unknown model_source {self.model_source.model_source}")

        return model_file_path

    def start_run_if_required(self, model: ModelBase, training_config: dict, trainer_state_dict: Optional[dict], overrides: TrainingOverrides, run_id: Optional[str] = None, sweep_id: Optional[str] = None):
        if not self.enabled or wandb.run is not None:
            return

        if run_id is None:
            run_id = wandb.util.generate_id()

        run_name, config = self.create_run_config(
            run_id=run_id,
            model=model,
            training_config=training_config,
            trainer_state_dict=trainer_state_dict,
            overrides=overrides,
            sweep_id=sweep_id,
        )

        wandb.init(
            id=run_id,
            entity=self.wandb_entity,
            project=self.wandb_project,
            name=run_name,
            config=config,
        )
        print(f"🏃 Started wandb run {wandb.run.id}")

    def update_run_name_and_config(self, run_id: str, model: ModelBase, training_config: dict, trainer_state_dict: Optional[dict], overrides: TrainingOverrides, sweep_id: Optional[str] = None):
        run_name, config = self.create_run_config(
            run_id=run_id,
            model=model,
            training_config=training_config,
            trainer_state_dict=trainer_state_dict,
            overrides=overrides,
            sweep_id=sweep_id,
        )
        api = wandb.Api()
        run = api.run(f"{self.wandb_entity}/{self.wandb_project}/{run_id}")
        run.name = run_name
        for key, value in config.items():
            run.config[key] = value
        run.update()
        print(f"🏃 Updated wandb run {run_id} name and configuration")

    def create_run_config(self, model: ModelBase, training_config: dict, trainer_state_dict: Optional[dict], overrides: TrainingOverrides, run_id: str, sweep_id: Optional[str]):
        model_name = model.model_name
        from_epoch = trainer_state_dict["epoch"] if trainer_state_dict is not None else 0

        config = {
            "model_name": model_name,
            "model_class": model.__class__.__name__,
            "model_config": model.config.to_dict(),
            "training_config": training_config,
            "from_epoch": from_epoch,
            "overrides": overrides.to_dict(),
        }

        if self.model_source_run is not None:
            run_name = f"{run_id}|cont:{run_id}|{model_name}"
            config["source_run_id"] = self.model_source_run.id
        elif sweep_id is not None:
            run_name = f"{run_id}|sweep:{sweep_id}|{model_name}"
        else:
            run_name = f"{run_id}|train|{model_name}"

        return run_name, config

    def run_new_sweep(self, config, function: Callable[[str], None], count=10) -> str:
        print(f"🔧 Creating sweep with {config['method']} optimization under {self.wandb_entity}/{self.wandb_project}...")
        print(f"📊 Target metric: {config['metric']['name']} ({config['metric']['goal']})")
        sweep_id = wandb.sweep(config, entity=self.wandb_entity, project=self.wandb_project)
        print(f"✅ Sweep created with ID: {sweep_id}")
        print(f"🌐 View sweep at: https://wandb.ai/{self.wandb_entity}/{self.wandb_project}/sweeps/{sweep_id}")
        print(f"🏃 Starting sweep agent with {count} runs...")
        wandb.agent(sweep_id, lambda: function(sweep_id), entity=self.wandb_entity, project=self.wandb_project, count=count)
        print(f"🎉 Sweep completed!")
        return sweep_id

    def continue_sweep(self, sweep_id: str, function: Callable[[str], None], count=10) -> str:
        print(f"🔄 Continuing existing sweep {sweep_id} under {self.wandb_entity}/{self.wandb_project} for {count} runs...")
        wandb.agent(sweep_id, lambda: function(sweep_id), entity=self.wandb_entity, project=self.wandb_project, count=count)
        print(f"🎉 Sweep completed!")
        return sweep_id

    def upload_latest_and_best_model_snapshots(self, model_name):
        if self.no_upload:
            print("Skipping uploading of model snapshots, due to config.should_upload_artifacts = False")
            return
        WandbHelper.upload_latest_and_best_model_snapshots(model_name=model_name, model_source_run=self.model_source_run)

class WandbArgumentsHelper:
    def __init__(self, require_wandb: bool, default_enable: bool, uploading: bool, downloading: bool):
        self.require_wandb = require_wandb
        self.default_enable = default_enable
        self.uploading = uploading
        self.downloading = downloading

    def handle_arguments(self, args) -> WandbArguments:
        other_args_present = args.wandb_entity is not None or args.wandb_project is not None
        if self.uploading:
            other_args_present = other_args_present or args.wandb_no_upload is not None
        if self.downloading:
            other_args_present = other_args_present or args.wandb_run or args.wandb_artifact_name is not None or args.wandb_artifact_type is not None or args.wandb_artifact_version is not None

        if self.require_wandb:
            enabled = True
        else:
            if self.default_enable:
                if args.wandb_disable:
                    if other_args_present:
                        raise ValueError("You cannot both provide --disable-wandb and other wandb parameters")
                    enabled = False
                else:
                    enabled = True
            else:
                enabled = args.wandb_enable or other_args_present

        if enabled:
            # Force a login prompt early in the process
            # This means that even if creation of a run is delayed (by e.g. initializing datasets) we will already
            # be logged in
            wandb.login(verify = True)

        model_source_arguments = None

        if self.downloading:
            match args.model_source:
                case "default":
                    if enabled:
                        model_source = "wandb"
                    else:
                        model_source = "local"
                case "wandb":
                    model_source = "wandb"
                case "local":
                    model_source = "local"
                case _:
                    raise ValueError(f"Unknown model_source {args.model_source}")

            model_source_arguments = ModelSourceArguments(
                model_name = args.model,
                model_source = model_source,
                continue_run_id = args.wandb_run,
                wandb_artifact_name=args.wandb_artifact_name,
                wandb_artifact_type = args.wandb_artifact_type,
                wandb_artifact_version = args.wandb_artifact_version,
            )

        return WandbArguments(
            enabled = enabled,
            no_upload = args.wandb_no_upload,
            wandb_entity = args.wandb_entity,
            wandb_project = args.wandb_project,
            model_source = model_source_arguments
        )

class WandbDefaults:
    def __init__(self, default_entity, default_project):
        self.default_entity = default_entity
        self.default_project = default_project
        self.default_enable = os.environ.get('WANDB_DEFAULT_ENABLED', "").lower() == "true"

    def add_arg_parser_arguments(self, parser, add_model_source_args: bool, add_upload_model_args: bool, require_wandb: bool = False) -> WandbArgumentsHelper:
        if not require_wandb:
            if self.default_enable:
                parser.add_argument(
                    '--wandb-disable',
                    action='store_true',
                    help='Disable using wandb'
                )
            else:
                parser.add_argument(
                    '--wandb-enable',
                    # Aliases
                    '--wandb',
                    action='store_true',
                    help='Enable wandb logging'
                )
        parser.add_argument(
            '--wandb-entity',
            default=self.default_entity,
            help='W&B entity name (used if --from-wandb is set)'
        )
        parser.add_argument(
            '--wandb-project',
            default=self.default_project,
            help=f'W&B project name (default: {self.default_project})'
        )
        if add_upload_model_args:
            parser.add_argument(
                '--wandb-no-upload',
                action='store_true',
                help='Disable artifact uploading to W&B (if wandb is enabled)'
            )
        if add_model_source_args:
            parser.add_argument(
                '--model-source',
                choices=["default", "local", "wandb"],
                default="default",
                help='Where to load the model from (defaults to wandb if it exists)'
            )
            parser.add_argument(
                '--wandb-run',
                # Aliases
                '--wandb-continue-run',
                type=str,
                default=None,
                help='Continue from the specified wandb run ID.'
            )
            parser.add_argument(
                '--wandb-artifact-name',
                type=str,
                default=None,
                help='Continue by reading a specific artifact name. Not needed if --wandb-continue-run is provided.'
            )
            parser.add_argument(
                '--wandb-artifact-type',
                type=str,
                choices=["best", "final"],
                default="best",
                help='Which model type to download. Not used if --wandb-artifact-name is specified.'
            )
            parser.add_argument(
                '--wandb-artifact-version',
                type=str,
                default="latest",
                help='The version of the artifact to load. Not used if --wandb-artifact-name is specified.'
            )

        return WandbArgumentsHelper(
            require_wandb=require_wandb,
            default_enable=self.default_enable,
            uploading=add_upload_model_args,
            downloading=add_model_source_args
        )
