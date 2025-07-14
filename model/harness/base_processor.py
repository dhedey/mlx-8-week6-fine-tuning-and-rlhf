import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class TrainingDatasets:
    train: Dataset | Iterable
    validation: Dataset | Iterable

class ProcessorBase(ABC):
    @abstractmethod
    def create_datasets(self) -> TrainingDatasets:
        raise NotImplementedError("This property should be implemented by subclasses.")

    @abstractmethod
    def collate(self, dataset_batch):
        raise NotImplementedError("This property should be implemented by subclasses.")

    def supports_multi_processing(self, device) -> bool:
        """
        WARNING:
        If num_workers > 0 then the whole processor will be copied to another thread when a dataset is loaded.
        This silently zeroes out any tensors in the processor which are stored in cuda, on any thread (!)
        So best to keep any tensors in the processor on the CPU, or to override this method to return false.
        """
        return device.type != 'mps'

    def create_train_dataloader(self, batch_size, num_workers, device, shuffle=True) -> tuple[Dataset | Iterable, DataLoader]:
        dataset = self.create_train_dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers if self.supports_multi_processing(device) else 0,
            pin_memory=device == 'cuda',
            collate_fn=self.collate
        )
        return dataset, dataloader

    def create_validation_dataloader(self, batch_size, num_workers, device, shuffle=False) -> tuple[Dataset | Iterable, DataLoader]:
        dataset = self.create_validation_dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0 if device.type == 'mps' else num_workers,
            pin_memory=device == 'cuda',
            collate_fn=self.collate
        )
        return dataset, dataloader