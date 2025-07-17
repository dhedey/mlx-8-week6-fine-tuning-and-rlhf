import dataclasses
from collections import OrderedDict
import torch.nn as nn
import torch
import os
import pathlib
import inspect
import re
from typing import Optional, Self

from .utility import select_device, ModuleConfig

class ModelBase(nn.Module):
    registered_types: dict[str, type] = {}
    config_class: type[ModuleConfig] = ModuleConfig

    def __init__(self, model_name: str, config: ModuleConfig):
        super().__init__()
        self.model_name = model_name
        self.config=config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register the class name in the ModelBase.registered_types dictionary
        if cls.__name__ in ModelBase.registered_types:
            raise ValueError(f"Model {cls.__name__} is a duplicate classname. Use a new class name.")
        ModelBase.registered_types[cls.__name__] = cls

        # Register the config class in cls.config_class
        init_signature = inspect.signature(cls.__init__)
        init_params = init_signature.parameters

        if "config" not in init_params:
            raise ValueError(f"Model {cls.__name__} must have a 'config' parameter in its __init__ method.")

        config_param_class = init_params["config"].annotation

        if not issubclass(config_param_class, ModuleConfig):
            raise ValueError(f"Model {cls.__name__} has a 'config' parameter in its __init__ method called {config_param_class}, but this class does not derive from ModuleConfig.")

        cls.config_class = config_param_class

    @property
    def device(self):
        return self.get_device()

    def get_device(self):
        return next(self.parameters()).device
    
    @staticmethod
    def model_path(model_name: str, location: Optional[str] = None) -> str:
        if location is None:
            location = "snapshots"
        assert location == "snapshots" or location == "trained", "ModelBase.model_path only supports 'snapshots' or 'trained' locations"
        return os.path.join(os.path.dirname(__file__), "..", location, f"{model_name}.pt")

    def save_model_data(
            self,
            training_config_dict: dict,
            training_state_dict: dict,
            location: Optional[str] = None,
            file_name: Optional[str] = None,
        ):
        if file_name is None:
            file_name = self.model_name

        model_path = ModelBase.model_path(file_name, location)
        pathlib.Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)

        save_only_grad_weights = training_config_dict.get("save_only_grad_weights", False)

        if save_only_grad_weights:
            model_weights = OrderedDict()
            for name, param in self.named_parameters():
                if param.requires_grad:
                    model_weights[name] = param
        else:
            model_weights = self.state_dict()

        torch.save({
            "model": {
                "class_name": type(self).__name__,
                "model_name": self.model_name,
                "weights": model_weights,
                "config": self.config.to_dict(),
            },
            "training": {
                "config": training_config_dict,
                "state": training_state_dict,
            },
        }, model_path)

        print_path = os.path.relpath(model_path, os.path.join(os.path.dirname(__file__), "..", ".."))

        if save_only_grad_weights:
            print(f"Model (learnable weights only) saved to {print_path}")
        else:
            print(f"Model saved to {print_path}")

    @classmethod
    def exists(
        cls,
        model_name: Optional[str] = None,
        location: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        return os.path.exists(cls.resolve_path(model_name=model_name, location=location, model_path=model_path))
    
    @classmethod
    def resolve_path(
        cls,
        model_name: Optional[str] = None,
        location: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        if model_path is not None:
            return model_path
        if model_name is not None:
            return ModelBase.model_path(model_name, location=location)
        raise ValueError("Either model_name or model_path must be provided to load a model.")

    @classmethod
    def load_only_training_state_dict(
        cls,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> dict:
        model_path = cls.resolve_path(model_name=model_name, model_path=model_path)
        loaded_model_data = torch.load(model_path)
        return loaded_model_data["training"]["state"]

    @classmethod
    def load_advanced(
        cls,
        model_name: Optional[str] = None,
        override_class_name = None,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> tuple[Self, dict, dict]:
        if device is None:
            device = select_device()

        model_path = cls.resolve_path(model_name=model_name, model_path=model_path)

        loaded_model_data = torch.load(model_path, map_location=device)
        print(f"Model data read from {model_path}")

        if model_name is None:
            model_name = loaded_model_data["model"]["model_name"]
    
        loaded_class_name = loaded_model_data["model"]["class_name"]
        actual_class_name = override_class_name if override_class_name is not None else loaded_class_name

        registered_types = ModelBase.registered_types
        if actual_class_name not in registered_types:
            raise ValueError(f"Model class {actual_class_name} is not a known Model. Available classes: {list(registered_types.keys())}")
        model_class: type[Self] = registered_types[actual_class_name]

        if not issubclass(model_class, cls):
            raise ValueError(f"The model {model_name} was attempted to be loaded with {cls.__name__}.load(\"{model_name}\") (loaded class name = {loaded_class_name}, override class name = {override_class_name}), but {model_class} is not a subclass of {cls}.")

        model_weights = loaded_model_data["model"]["weights"]
        model_config = model_class.config_class.from_dict(loaded_model_data["model"]["config"])
        training_state_dict = loaded_model_data["training"]["state"]
        training_config_dict = loaded_model_data["training"]["config"]

        model: ModelBase = model_class(
            model_name=model_name,
            config=model_config,
        )
        if training_config_dict.get("save_only_grad_weights", False):
            model.load_state_dict(model_weights, strict=False)
        else:
            model.load_state_dict(model_weights)

        return model.to(device), training_state_dict, training_config_dict

    @classmethod
    def load_for_evaluation(cls, model_name: Optional[str] = None, model_path: Optional[str] = None, device: Optional[str] = None) -> Self:
        model, _, _ = cls.load_advanced(model_name=model_name, model_path=model_path, device=device)
        model.eval()
        return model

    def print_detailed_parameter_counts(self) -> None:
        print_detailed_parameter_counts(self, f"model ({self.model_name})")

def print_detailed_parameter_counts(nn_module, model_name: Optional[str] = None, module_name: Optional[str] = None, only_learnable: bool = True) -> None:
    @dataclasses.dataclass
    class NodeContent:
        # The number of distinct named parameters which have hit this path
        route_count: int
        # The total number of learnable parameters under this node
        learnable_param_count: int
        # The total number of all parameters under this node
        total_param_count: int
        # normalized_key => (keys_set, node_content) where:
        # - normalized_key can combined multiple keys together, e.g. "fc*"
        # - keys_set is the local matching keys, e.g. "fc1", "fc2" for "fc*"
        children: dict[str, tuple[set[str], Self]]

        @classmethod
        def default(cls) -> Self:
            return cls(route_count=0, learnable_param_count=0, total_param_count=0, children={})

        def add(self, path, requires_grad, param_count):
            self.route_count += 1
            self.total_param_count += param_count
            if requires_grad:
                self.learnable_param_count += param_count

            if len(path) > 0:
                child_key = path.pop(0)
                normalized_key = re.sub(r'\d+', '*', child_key)
                if normalized_key not in self.children:
                    self.children[normalized_key] = {child_key}, NodeContent.default()

                child_keys, child_node = self.children[normalized_key]
                child_keys.add(child_key)
                child_node.add(path, requires_grad, param_count)

        def print(self, root_name: str, only_learnable: bool):
            total_learnable = self.learnable_param_count
            total_any = self.total_param_count

            learnable_just_len = max(len("# Learnable"), len(f"{total_learnable:,}"))
            any_just_len = max(len("# Any"), len(f"{total_any:,}"))

            print(f" Leaf | {"# Learnable".rjust(learnable_just_len)} | %AllLearn | %AllAny | {"# Any".rjust(any_just_len)} | %AllAny | Module")

            self._print_subtree(
                only_learnable=only_learnable,
                prefix=root_name,
                depth=0,
                total_learnable=total_learnable,
                learnable_just_len=learnable_just_len,
                total_any=total_any,
                any_just_len=any_just_len,
            )

        def _print_subtree(
            self,
            only_learnable: bool,
            prefix: str,
            depth: int,
            total_learnable: int,
            learnable_just_len: int,
            total_any: int,
            any_just_len: int,
        ) -> None:
            if depth > 0 and only_learnable and self.learnable_param_count == 0:
                # If this node has no learnable parameters under it, we skip it
                return

            indent = "  " * depth

            leaf_part = ("**" if len(self.children) == 0 else " ").center(6)
            learnable_part = f"{self.learnable_param_count:,}".rjust(learnable_just_len)
            perc_learnable_part = f"{self.learnable_param_count / total_learnable:.2%}".rjust(9) if total_learnable > 0 else "---.--%".rjust(9)
            perc_learnable_all_part = f"{self.learnable_param_count / total_any:.2%}".rjust(7) if total_any > 0 else "---.--%".rjust(7)
            any_weights = f"{self.total_param_count:,}".rjust(any_just_len)
            perc_any_all_weights = f"{self.total_param_count / total_any:.2%}".rjust(7) if total_any > 0 else "---.--%".rjust(7)

            print_prefix = f"{leaf_part}| {learnable_part} | {perc_learnable_part} | {perc_learnable_all_part} | {any_weights} | {perc_any_all_weights} | {indent}"

            match len(self.children):
                case 0:
                    if self.route_count > 1:
                        prefix += f" ({self.route_count} total)"
                    print(f"{print_prefix}{prefix}")
                case 1:
                    # Delegate to single child
                    child_key, (child_keys, child_node) = next(iter(self.children.items()))
                    if len(child_keys) == 1:
                        # Use the un-normalized key, e.g. fc1
                        child_key = next(iter(child_keys))
                    else:
                        # Show the count of different layers, e.g. fc*[count=4]
                        child_key = f"{child_key}[count={len(child_keys)}]"
                    child_node._print_subtree(
                        prefix=f"{prefix}.{child_key}",
                        only_learnable=only_learnable,
                        depth=depth,
                        total_learnable=total_learnable,
                        learnable_just_len=learnable_just_len,
                        total_any=total_any,
                        any_just_len=any_just_len,
                    )
                case _:
                    print(f"{print_prefix}{prefix}")
                    sorted_children = sorted(
                        self.children.items(),
                        key=lambda item: item[1][1].learnable_param_count,
                        reverse=True
                    )
                    for child_key, (child_keys, child_node) in sorted_children:
                        if len(child_keys) == 1:
                            # Use the un-normalized key, e.g. fc1
                            child_key = next(iter(child_keys))
                        else:
                            # Show the count of different layers, e.g. fc*[10]
                            child_key = f"{child_key}[{len(child_keys)}]"
                        child_node._print_subtree(
                            prefix=f".{child_key}",
                            only_learnable=only_learnable,
                            depth=depth + 1,
                            total_learnable=total_learnable,
                            learnable_just_len=learnable_just_len,
                            total_any=total_any,
                            any_just_len=any_just_len,
                        )

    root_node = NodeContent.default()

    for name, parameter in nn_module.named_parameters():
        root_node.add(name.split("."), parameter.requires_grad, parameter.numel())

    model_name = model_name if model_name is not None else nn_module.__class__.__name__
    weights_name = "learnable weights" if only_learnable else "weights"
    print(f"This {model_name} has the following {weights_name}:")
    root_name = module_name if module_name is not None else "root"
    root_node.print(root_name=root_name, only_learnable=only_learnable)
    print()

