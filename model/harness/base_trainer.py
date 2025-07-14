import inspect
import time
from dataclasses import dataclass
from typing import Optional, Self
from itertools import islice
from abc import ABC, abstractmethod

import torch
import wandb
from pydantic import Field
from torch import optim as optim
from tqdm.auto import tqdm

from .base_model import ModelBase
from .base_processor import ProcessorBase
from .utility import PersistableData

class TrainingConfig(PersistableData):
    batch_size: int
    epochs: int
    learning_rate: float
    dataset_num_workers: int = 2
    shuffle_validation_set: bool = True
    save_only_grad_weights: bool = False
    warmup_epochs: int = 0 # Number of epochs to warm up the learning rate
    recalculate_running_loss_after_batches: int = 10
    custom_validate_after_batches: Optional[int] = None
    batch_limit: Optional[int] = None
    optimizer: str = "AdamW"
    """
    The optimizer to use for training.
    Currently supports "Adam" and "AdamW".
    In future, it could support any under https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
    """
    optimizer_params: Optional[dict] = None
    """
    The parameters to use with the optimizer. You don't need to set the learning rate, this is overridden from the learning_rate field.
    """
    schedulers: list[str | tuple[str, dict]] = Field(default_factory=list, description="List of (scheduler_name, scheduler_params) tuples")
    """
    The scheduler/s to use for training along with their parameters.
    The names should be any scheduler name under the `torch.optim.lr_scheduler` namespace, e.g. "LRScheduler"
    If a list of scheduler names is provided, they will be wrapped in a chained scheduler.
    See https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate for more details.
    """
    early_stopping: bool = False
    early_stopping_patience: int = 5
    verbose: bool = False

class ValidationResults(PersistableData):
    epoch: int
    train_comparable_loss: float = 0.0 # Default to 0.0 to support backwards compatibility
    """
    The average training loss is a measure of how well the model performs on the validation set.
    It should be comparable with the average loss from a training epoch, and can act as a measure of
    overfitting.
    """

    custom: dict
    """
    Custom results, depending on the trainer
    """

    @property
    def objective(self) -> float:
        """
        The validation objective is a measure of how well the model performs on the validation set (high is good)
        My default, it's `1 / average_training_loss` but could be any other better metric, so long as it is comparable between epochs.
        """
        if "objective" in self.custom:
            return self.custom["objective"]
        else:
            if self.train_comparable_loss < 0:
                raise ValueError("Loss is expected to be positive")
            eps = 0.00001
            return 1/max(self.train_comparable_loss, eps)


@dataclass
class BatchResults:
    total_loss: torch.Tensor # Singleton float tensor
    num_samples: int
    intermediates: dict


class EpochTrainingResults(PersistableData):
    epoch: int
    average_loss: float
    num_samples: int


class FullTrainingResults(PersistableData):
    total_epochs: int
    last_validation: ValidationResults
    best_validation: ValidationResults
    last_training_epoch: EpochTrainingResults
    best_training_epoch: EpochTrainingResults


class TrainingState(PersistableData):
    epoch: int
    optimizer_state: dict
    model_trainer_class_name: str
    total_training_time_seconds: float
    latest_training_results: EpochTrainingResults
    best_training_results: Optional[EpochTrainingResults] = None # None to support backwards compatibility
    all_training_results: list[EpochTrainingResults] = Field(default_factory=list) # Default to support backwards compatibility
    latest_validation_results: ValidationResults
    best_validation_results: Optional[ValidationResults] = None # None to support backwards compatibility
    all_validation_results: list[ValidationResults] = Field(default_factory=list) # Default to support backwards compatibility
    scheduler_state: Optional[dict] = None # None to support backwards compatibility


class TrainingOverrides(PersistableData):
    override_batch_size: Optional[int] = None
    override_batch_limit: Optional[int] = None
    override_to_epoch: Optional[int] = None
    override_learning_rate: Optional[float] = None
    recalculate_running_loss_after_batches: Optional[int] = None
    validate_after_epochs: int = 1
    seed: int = 42
    use_dataset_cache: bool = True
    print_detailed_parameter_counts: bool = False


def create_composite_scheduler(
    optimizer: optim.Optimizer,
    schedulers: list[str | tuple[str, dict]],
) -> Optional[optim.lr_scheduler.LRScheduler]:
    def map_scheduler(scheduler: str | tuple[str, dict]) -> optim.lr_scheduler.LRScheduler:
        if isinstance(scheduler, str):
            return create_scheduler(optimizer, scheduler, {})
        else:
            return create_scheduler(optimizer, scheduler[0], scheduler[1])

    match len(schedulers):
        case 0:
            return None
        case 1:
            return map_scheduler(schedulers[0])
        case _:
            # This doesn't work with `ReduceLROnPlateau`...
            # Maybe we could create our own ChainedScheduler where it does work, by intelligently passing through the metrics field
            return optim.lr_scheduler.ChainedScheduler([
                map_scheduler(scheduler) for scheduler in schedulers
            ])


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    scheduler_params: dict,
) -> optim.lr_scheduler.LRScheduler:
    match scheduler_type:
        case "LRScheduler":
            return optim.lr_scheduler.LRScheduler(optimizer, **scheduler_params)
        case "LambdaLR":
            return optim.lr_scheduler.LambdaLR(optimizer, **scheduler_params)
        case "MultiplicativeLR":
            return optim.lr_scheduler.MultiplicativeLR(optimizer, **scheduler_params)
        case "StepLR":
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        case "MultiStepLR":
            return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
        case "LinearLR":
            return optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
        case "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
        case "PolynomialLR":
            return optim.lr_scheduler.PolynomialLR(optimizer, **scheduler_params)
        case "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        case "SequentialLR":
            return optim.lr_scheduler.SequentialLR(optimizer, **scheduler_params)
        case "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        case "CyclicLR":
            return optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)
        case "OneCycleLR":
            return optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
        case "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
        case _:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def create_optimizer(
    model: ModelBase,
    optimizer_name: str,
    optimizer_params: Optional[dict],
    learning_rate: float,
) -> optim.Optimizer:
    if optimizer_params is None:
        optimizer_params = {}

    optimizer_params["lr"] = learning_rate

    match optimizer_name.lower():
        case "adam":
            return optim.Adam(model.parameters(), **optimizer_params)
        case "adamw":
            return optim.AdamW(model.parameters(), **optimizer_params)
        case _:
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

def loss_change_description(old, new):
    change = (new - old) / old
    if change >= 0:
        return f"{change:.2%} worse"
    else:
        return f"{(-change):.2%} better"

class ModelTrainerBase(ABC):
    registered_types: dict[str, type[Self]] = {}
    config_class: type[TrainingConfig] = TrainingConfig

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in ModelTrainerBase.registered_types:
            raise ValueError(f"ModelTrainer {cls.__name__} is a duplicate classname. Use a new class name.")
        ModelTrainerBase.registered_types[cls.__name__] = cls

        # Register the config class in cls.config_class
        init_signature = inspect.signature(cls.__init__)
        init_params = init_signature.parameters

        if "config" not in init_params:
            raise ValueError(f"ModelTrainer {cls.__name__} must have a 'config' parameter in its __init__ method.")

        config_param_class = init_params["config"].annotation

        if not issubclass(config_param_class, TrainingConfig):
            raise ValueError(f"ModelTrainer {cls.__name__} has a 'config' parameter in its __init__ method called {config_param_class}, but this class does not derive from TrainingConfig.")

        cls.config_class = config_param_class

    def __init__(
            self,
            model: ModelBase,
            config: TrainingConfig,
            continuation: Optional[TrainingState] = None,
            overrides: Optional[TrainingOverrides] = None,
        ):

        self.model = model
        self.config = config

        learnable_weights_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_weights_count = sum(p.numel() for p in model.parameters())
        print(f"Initializing {self.__class__.__name__} for {model.__class__.__name__} named \"{model.model_name}\" (learnable weights = {learnable_weights_count:,} total weights = {total_weights_count:,})")
        print()

        if overrides is None:
            overrides = TrainingOverrides()

        torch.manual_seed(overrides.seed)

        if overrides.print_detailed_parameter_counts:
            self.model.print_detailed_parameter_counts()

        self.validate_after_epochs = overrides.validate_after_epochs

        if overrides.override_to_epoch is not None:
            self.config.epochs = overrides.override_to_epoch
            print(f"Overriding training end epoch to {self.config.epochs}")

        if overrides.override_batch_size is not None:
            self.config.batch_size = overrides.override_batch_size
            print(f"Overriding batch size to {self.config.batch_size}")

        if overrides.override_batch_limit is not None:
            self.config.batch_limit = overrides.override_batch_limit
            print(f"Overriding batch limit to {self.config.batch_limit}")

        if overrides.override_learning_rate is not None:
            self.config.learning_rate = overrides.override_learning_rate
            print(f"Overriding learning rate to {self.config.learning_rate}")

        if overrides.recalculate_running_loss_after_batches is not None:
            self.config.recalculate_running_loss_after_batches = overrides.recalculate_running_loss_after_batches
            print(f"Overriding print after batches to {self.config.recalculate_running_loss_after_batches}")

        self.optimizer = create_optimizer(self.model, self.config.optimizer, self.config.optimizer_params, self.config.learning_rate)

        schedulers = self.config.schedulers.copy()
        if config.warmup_epochs > 0:
            print(f"{config.warmup_epochs} warm-up epochs were defined, so configuring an initial linear scheduler")
            if len(schedulers) > 0 and isinstance(schedulers[0], tuple) and schedulers[0][0] == "LinearLR":
                # We already added a scheduler and persisted it (before there was a copy on the line above)
                pass
            else:
                schedulers.insert(0, ("LinearLR", { "total_iters": config.warmup_epochs }))

        self.scheduler = create_composite_scheduler(self.optimizer, schedulers)

        if continuation is not None:
            self.epoch = continuation.epoch
            self.optimizer.load_state_dict(continuation.optimizer_state)
            if overrides.override_learning_rate is not None:
                # After loading the state dict, we need to set the override again
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = overrides.override_learning_rate
            if self.scheduler is not None and continuation.scheduler_state is not None:
                self.scheduler.load_state_dict(continuation.scheduler_state)
            self.total_training_time_seconds = continuation.total_training_time_seconds
            self.latest_training_results = continuation.latest_training_results
            self.best_training_results = continuation.best_training_results if continuation.best_training_results is not None else continuation.latest_training_results
            self.all_training_results = continuation.all_training_results
            self.latest_validation_results = continuation.latest_validation_results
            self.best_validation_results = continuation.best_validation_results if continuation.best_validation_results is not None else continuation.latest_validation_results
            self.all_validation_results = continuation.all_validation_results
            print(f"Resuming training from saved state after epoch {self.epoch}")
            print()
        else:
            self.epoch = 0
            self.total_training_time_seconds = 0.0
            self.latest_training_results: Optional[EpochTrainingResults] = None
            self.best_training_results: Optional[EpochTrainingResults] = None
            self.all_training_results: list[EpochTrainingResults] = []
            self.latest_validation_results: Optional[ValidationResults] = None
            self.best_validation_results: Optional[ValidationResults] = None
            self.all_validation_results: list[ValidationResults] = []

        self.processor = self.create_processor()
        dataset_num_workers = config.dataset_num_workers if self.processor.supports_multi_processing(model.device) else 0
        print(f"Preparing datasets (batch_size = {config.batch_size}, parallelization = {dataset_num_workers if dataset_num_workers > 0 else None})")

        each_dataset = self.processor.create_datasets()
        self.train_data_loader = torch.utils.data.DataLoader(
            each_dataset.train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=dataset_num_workers,
            pin_memory=model.device == 'cuda',
            collate_fn=self.processor.collate,
        )
        self.validation_data_loader = torch.utils.data.DataLoader(
            each_dataset.validation,
            batch_size=config.batch_size,
            shuffle=config.shuffle_validation_set,
            num_workers=dataset_num_workers,
            pin_memory=model.device == 'cuda',
            collate_fn=self.processor.collate,
        )
        # noinspection PyTypeChecker
        train_size = str(len(each_dataset.train)) if hasattr(each_dataset.train, "__len__") else "[unknown]"
        print(f"Training data: {train_size} across {len(self.train_data_loader)} batches")
        # noinspection PyTypeChecker
        validation_size = str(len(each_dataset.validation)) if hasattr(each_dataset.validation, "__len__") else "[unknown]"
        print(f"Validation data: {validation_size} across {len(self.validation_data_loader)} batches")
        print()

    @abstractmethod
    def create_processor(self) -> ProcessorBase:
        """This is run in the super constructor, after self.config and self.model are set"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def process_batch(self, raw_batch) -> BatchResults:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def start_custom_validation(self) -> dict:
        """
        This method may be overridden by trainers wanting custom validation.
        It should return initialized custom validation metrics which are fed into the `custom_validate_batch` method.
        """
        return {}

    def custom_validate_batch(self, custom_validation_metrics: dict, batch_results: BatchResults, batch_num: int, total_batches: int):
        """
        This method may be overridden by trainers wanting custom validation.
        It is called after applying the model to a batch of the validation set, and may mutate the
        custom_validation_metrics dictionary to add information about this batch.
        batch_num is 1-indexed, so batch_num==total_batches is the last batch
        """
        pass

    def finalize_custom_validation(self, total_samples: int, total_batches: int, custom_validation_metrics: dict) -> dict:
        """
        This method may be overridden by trainers wanting custom validation.
        By default, this returns the custom validation metrics as-is, but can be overridden to perform final
        calculations on the metrics.
        If you have a better measure of validation success than the train_comparable_loss, you can return a "loss" in
        the dictionary.
        You may also choose to perform additional custom validation here / print additional stuff.
        """
        return custom_validation_metrics

    @classmethod
    def load_with_model(cls, model_name: Optional[str] = None, overrides: Optional[TrainingOverrides] = None, device: Optional[str] = None, model_path: Optional[str] = None) -> Self:
        model, state_dict, config_dict = ModelBase.load_advanced(model_name=model_name, device=device, model_path=model_path)
        return cls.load(
            model=model,
            config=config_dict,
            state=TrainingState.from_dict(state_dict),
            overrides=overrides,
        )

    @classmethod
    def load(cls, model: ModelBase, config: dict, state: TrainingState, overrides: Optional[TrainingOverrides] = None) -> Self:
        trainer_class_name = state.model_trainer_class_name

        registered_types = ModelTrainerBase.registered_types
        if trainer_class_name not in registered_types:
            raise ValueError(f"Trainer class {trainer_class_name} is not a known ModelTrainer. Available classes: {list(registered_types.keys())}")
        trainer_class: type[Self] = registered_types[trainer_class_name]

        if not issubclass(trainer_class, cls):
            raise ValueError(f"The trainer was attempted to be loaded with {cls.__name__}.load(..) with a trainer class of \"{trainer_class_name}\"), but {trainer_class_name} is not a subclass of {cls}.")

        config = trainer_class.config_class.from_dict(config)

        trainer = trainer_class(
            model=model,
            config=config,
            continuation=state,
            overrides=overrides,
        )

        return trainer

    def train(self) -> FullTrainingResults:
        while self.epoch < self.config.epochs:
            self.epoch += 1
            print(f"======================== EPOCH {self.epoch}/{self.config.epochs} ========================")
            epoch_start_time = time.time()
            previous_train_results = self.latest_training_results
            previous_validation_results = self.latest_validation_results
            previous_best_validation_results = self.best_validation_results

            self.train_epoch()

            if self.epoch % self.validate_after_epochs == 0 or self.epoch == self.config.epochs or self.latest_validation_results is None:
                self.validate()

                if wandb.run is not None:
                    log_data = {
                        "epoch": self.epoch,
                    }
                    def add_prefixed(output: dict, source: dict, prefix: str):
                        for key, value in source.items():
                            match key:
                                case "epoch": # Ignore
                                    continue
                                case "custom": # Flatten "custom" keys into the other data
                                    add_prefixed(output, value, prefix)
                                    continue
                                case str():
                                    if key.startswith(prefix):
                                        output[key] = value
                                    else:
                                        output[f"{prefix}{key}"] = value
                                case _:
                                    raise ValueError(f"Unknown prefix type {prefix} in training.")

                    add_prefixed(log_data, self.latest_validation_results.to_dict(), "validation_")
                    add_prefixed(log_data, self.latest_training_results.to_dict(), "train_")

                    wandb.log(log_data)

            assert self.latest_training_results is not None
            assert self.latest_validation_results is not None
            assert self.best_validation_results is not None

            time_elapsed = time.time() - epoch_start_time

            print()
            print(f"Epoch {self.epoch} complete in {time_elapsed:#.1f}s")

            train_average_loss = self.latest_training_results.average_loss
            if previous_train_results is not None:
                train_change = f" ({loss_change_description(previous_train_results.average_loss, train_average_loss)})"
            else:
                train_change = ""

            if self.latest_validation_results.epoch == self.epoch:
                validation_average_train_loss = self.latest_validation_results.train_comparable_loss
                validation_objective = self.latest_validation_results.objective
                overfitting_measure = (validation_average_train_loss - train_average_loss) / validation_average_train_loss if validation_average_train_loss > 0 else float('inf')

                if previous_validation_results is not None:
                    validation_change = f" ({loss_change_description(previous_validation_results.train_comparable_loss, validation_average_train_loss)})"
                else:
                    validation_change = ""
                print(f"> Train Loss: {train_average_loss:#.3g}{train_change} | Validation Loss: {validation_average_train_loss:#.3g}{validation_change} | Overfitting: {overfitting_measure:.2%}")

                if previous_best_validation_results is None or previous_best_validation_results.epoch == self.epoch:
                    print(f"> Validation Objective: {validation_objective:#.3g} [RECORD]")
                else:
                    if previous_best_validation_results.objective > 0:
                        improvement = (validation_objective - previous_best_validation_results.objective) / previous_best_validation_results.objective
                    else:
                        improvement = float('inf')
                    last_best_epochs_ago = self.epoch - previous_best_validation_results.epoch
                    pluralised_epochs = "epoch" if last_best_epochs_ago == 1 else "epochs"
                    if validation_objective > previous_best_validation_results.objective:
                        print(f"> Validation Objective: {validation_objective:#.3g} ({improvement:.2%} better than {last_best_epochs_ago} {pluralised_epochs} ago) [RECORD]")
                    elif validation_objective == previous_best_validation_results.objective:
                        print(f"> Validation Objective: {validation_objective:#.3g} (Equal to {last_best_epochs_ago} {pluralised_epochs} ago)")
                    else:
                        print(f"> Validation Objective: {validation_objective:#.3g} ({(-improvement):.2%} worse than {last_best_epochs_ago} {pluralised_epochs} ago)")
            else:
                print(f"> Train Loss: {train_average_loss:#.3g}{train_change} | No validation was performed this epoch")

            if self.scheduler is not None:
                # Handle the parameters for the ReduceLROnPlateau scheduler
                scheduler_step_parameters = inspect.signature(self.scheduler.step).parameters
                if "metrics" in scheduler_step_parameters.keys():
                    self.scheduler.step(self.latest_validation_results.validation_loss)
                else:
                    self.scheduler.step()

            self.save_model()
            print()

            if self.config.early_stopping:
                best_results_epochs_ago = self.best_validation_results.epoch - self.latest_validation_results.epoch

                if best_results_epochs_ago > self.config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {self.latest_validation_results.epoch}, because the best validation results occurred at epoch {self.best_validation_results.epoch}, over {self.config.early_stopping_patience} epochs ago")
                    break

        print("======================== Training complete ========================")

        return FullTrainingResults(
            total_epochs=self.config.epochs,
            last_validation=self.latest_validation_results,
            best_validation=self.best_validation_results,
            last_training_epoch=self.latest_training_results,
            best_training_epoch=self.best_training_results,
        )

    def train_epoch(self):
        self.model.train()

        running_total_every = self.config.recalculate_running_loss_after_batches
        custom_validate_every = self.config.custom_validate_after_batches

        total_batches = len(self.train_data_loader)

        if self.config.batch_limit is not None and total_batches >= self.config.batch_limit:
            total_batches = self.config.batch_limit

        epoch_loss = 0.0
        epoch_samples = 0
        running_loss = 0.0
        running_samples = 0
        recent_avg_loss = None

        batch_num = 0
        start_epoch_time_at = time.time()

        batch_limit = "" if self.config.batch_limit is None else f", batch_limit={self.config.batch_limit}"
        print(f"Beginning training (batch_size={self.config.batch_size}{batch_limit})")

        def loader_description():
            recent_loss = f"{recent_avg_loss:#.3g}" if recent_avg_loss is not None else "-N/A-"
            avg_loss = f"{(epoch_loss / epoch_samples):#.3g}" if epoch_samples > 0 else "-N/A-"
            return f"Training batch {batch_num}/{total_batches} | Loss avg={avg_loss} recent={recent_loss}"

        loader = tqdm(islice(self.train_data_loader, total_batches), total=total_batches, desc=loader_description())

        for raw_batch in loader:
            batch_num += 1
            loader.set_description(loader_description())

            self.optimizer.zero_grad()
            batch_results = self.process_batch(raw_batch)

            loss = batch_results.total_loss
            running_samples += batch_results.num_samples
            running_loss += loss.item()
            epoch_samples += batch_results.num_samples
            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_num % running_total_every == 0 or batch_num == total_batches:
                recent_avg_loss = running_loss / running_samples
                running_loss = 0.0
                running_samples = 0
                loader.set_description(loader_description())
                
            if custom_validate_every is not None and batch_num % custom_validate_every == 0:
                print()
                print(f"Starting mid-epoch custom validation at batch {batch_num}:")
                print()
                metrics = self.start_custom_validation()
                self.finalize_custom_validation(total_samples=0, total_batches=0, custom_validation_metrics=metrics)


        average_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        training_time = time.time() - start_epoch_time_at

        if self.total_training_time_seconds is not None:
            self.total_training_time_seconds += training_time

        training_results = EpochTrainingResults(
            epoch=self.epoch,
            average_loss=average_loss,
            num_samples=epoch_samples,
        )
        self.latest_training_results = training_results

        if len(self.all_training_results) == 0 or self.all_training_results[-1].epoch < training_results.epoch:
            self.all_training_results.append(training_results)

        if self.best_training_results is None or training_results.average_loss < self.best_training_results.average_loss:
            self.best_training_results = training_results

    def _run_validation(self) -> ValidationResults:
        running_total_every = self.config.recalculate_running_loss_after_batches

        total_batches = len(self.validation_data_loader)

        if self.config.batch_limit is not None and total_batches >= self.config.batch_limit:
            total_batches = self.config.batch_limit

        custom_validation_metrics = self.start_custom_validation()
        total_loss = 0.0
        total_samples = 0
        running_loss = 0.0
        running_samples = 0
        recent_avg_loss = None

        batch_num = 0

        batch_limit = "" if self.config.batch_limit is None else f", batch_limit={self.config.batch_limit}"
        print(f"Beginning validation (batch_size={self.config.batch_size}{batch_limit})")

        def loader_description():
            recent_loss = f"{recent_avg_loss:#.3g}" if recent_avg_loss is not None else "-N/A-"
            avg_loss = f"{(total_loss / total_samples):#.3g}" if total_samples > 0 else "-N/A-"
            return f"Validation batch {batch_num}/{total_batches} | Loss avg={avg_loss} recent={recent_loss}"

        loader = tqdm(islice(self.validation_data_loader, total_batches), total=total_batches, desc=loader_description())

        for raw_batch in loader:
            batch_num += 1
            loader.set_description(loader_description())

            batch_results = self.process_batch(raw_batch)

            loss = batch_results.total_loss
            running_samples += batch_results.num_samples
            running_loss += loss.item()
            total_samples += batch_results.num_samples
            total_loss += loss.item()

            self.custom_validate_batch(
                custom_validation_metrics,
                batch_results,
                batch_num,
                total_batches,
            )

            if batch_num % running_total_every == 0 or batch_num == total_batches:
                recent_avg_loss = running_loss / running_samples
                running_loss = 0.0
                running_samples = 0
                loader.set_description(loader_description())

        average_loss = total_loss / total_samples if total_samples > 0 else 0.0

        finalized_custom_results = self.finalize_custom_validation(total_samples, total_batches, custom_validation_metrics)

        validation_results = ValidationResults(
            epoch=self.epoch,
            train_comparable_loss=average_loss,
            custom=finalized_custom_results,
        )

        self.latest_validation_results = validation_results
        if len(self.all_validation_results) == 0 or self.all_validation_results[-1].epoch < validation_results.epoch:
            self.all_validation_results.append(validation_results)
        if self.best_validation_results is None or validation_results.objective > self.best_validation_results.objective:
            self.best_validation_results = validation_results

        custom_results_str = str(finalized_custom_results)
        if custom_results_str != "" and custom_results_str != "{}":
            print()
            print(f"Custom validation results: {custom_results_str}")

        return validation_results

    def validate(self) -> ValidationResults:
        self.model.eval()

        with torch.no_grad():
            validation_results = self._run_validation()

        return validation_results

    def save_model(self):
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        training_state = TrainingState(
            epoch=self.epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=scheduler_state,
            model_trainer_class_name=self.__class__.__name__,
            total_training_time_seconds=self.total_training_time_seconds,
            latest_training_results=self.latest_training_results,
            all_training_results=self.all_training_results,
            best_training_results=self.best_training_results,
            latest_validation_results=self.latest_validation_results,
            best_validation_results=self.best_validation_results,
            all_validation_results=self.all_validation_results,
        )
        self.model.save_model_data(
            file_name=self.model.model_name,
            training_config_dict=self.config.to_dict(),
            training_state_dict=training_state.to_dict(),
        )

        if self.latest_validation_results is None:
            if self.config.verbose:
                print("No new validation loss available, skipping comparison with the best model.")
            return

        best_model_name = self.model.model_name + '-best'
        best_validation_objective = None
        best_validation_epoch = None
        if ModelBase.exists(model_name=best_model_name):
            try:
                best_training_state = TrainingState.from_dict(ModelBase.load_only_training_state_dict(model_name=best_model_name))
                best_validation_objective = best_training_state.latest_validation_results.objective
                best_validation_epoch = best_training_state.latest_validation_results.epoch
            except Exception as e:
                print(f"⚠️ Failed to load the best model training state: {e}")

        def format_optional_float(value):
            return f"{value:#.3g}" if value is not None else "N/A"

        latest_validation_objective = self.latest_validation_results.objective

        is_improvement = best_validation_objective is None or latest_validation_objective > best_validation_objective
        if is_improvement:
            if self.config.verbose:
                print(f"The current validation objective {format_optional_float(latest_validation_objective)} is better than the previous best validation objective {format_optional_float(best_validation_objective)} from epoch {best_validation_epoch}, saving as {best_model_name}...")
            self.model.save_model_data(
                file_name=best_model_name,
                training_config_dict=self.config.to_dict(),
                training_state_dict=training_state.to_dict(),
            )
        else:
            if self.config.verbose:
                print(f"The current validation objective {format_optional_float(latest_validation_objective)} is not better than the previous best validation objective {format_optional_float(best_validation_objective)} from epoch {best_validation_epoch}, so not saving as best.")
