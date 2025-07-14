import torch.nn as nn
import torch

from model.harness import ModelBase, ModuleConfig, BatchResults, ModelTrainerBase, TrainingConfig, TrainingState, TrainingOverrides, ProcessorBase, TrainingDatasets
from .prepared_datasets import generate_urban_classifier_dataset
from typing import Optional

class UrbanSoundClassifierModel(ModelBase):
    def __init__(self, model_name: str, config: ModuleConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

    def forward(self, collated_batch) -> torch.Tensor:
        """
        Takes MelSpectrograms and returns logits for each class
        """
        raise NotImplementedError("Implement in subclass")

    def total_mels(self) -> int:
        raise NotImplementedError("Implement in subclass")

    def total_time_frames(self) -> int:
        raise NotImplementedError("Implement in subclass")


class ConvolutionalUrbanSoundClassifierModelConfig(ModuleConfig):
    dropout: float = 0.1

class ConvolutionalUrbanSoundClassifierModel(UrbanSoundClassifierModel):
    def __init__(self, model_name: str, config: ConvolutionalUrbanSoundClassifierModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

        num_classes = 10

        self.convolutional_layers = nn.Sequential(*[
            # 2d convolution over time and frequency
            # [Batch, Channels=1, Freq=n_mels, Time=T]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, n_mels, T]
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),  # [B, 32, n_mels//2, T//2]
            nn.Dropout(config.dropout),

            # 2d convolution over time and frequency
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),  # [B, 64, n_mels//4, T//4]
            nn.Dropout(config.dropout),
        ])
        self.classifier = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        ])

    def forward(self, collated_batch) -> torch.Tensor:
        """
        Takes MelSpectrograms and returns logits for each class
        """
        # (batch, channels=1, freq=num_mels, time)
        hidden_state = self.convolutional_layers(collated_batch["spectrograms"])
        return self.classifier(hidden_state)

    def total_mels(self) -> int:
        return 128

    def total_time_frames(self) -> int:
        return 321

class PatchTransformerUrbanSoundClassifierModelConfig(ModuleConfig):
    embed_dim: int = 128
    patch_size_mels: int = 16
    patch_size_time: int = 16
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 256
    dropout: float = 0.1
    num_mels: int = 64
    time_frames: int = 321

class PatchTransformerUrbanSoundClassifierModel(UrbanSoundClassifierModel):
    def __init__(self, model_name: str, config: PatchTransformerUrbanSoundClassifierModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        # Input: batch of spectrograms.shape = [batch, 1, n_mels, time]
        # Input for transformer: [batch, seq_len, feature_dim]
        num_classes = 10

        self.n_mels = config.num_mels
        self.time_frames = config.time_frames
        self.patch_mels = config.patch_size_mels
        self.patch_time = config.patch_size_time
        self.embed_dim = config.embed_dim

        self.num_patches_mel = self.n_mels // self.patch_mels
        self.num_patches_time = self.time_frames // self.patch_time
        self.num_patches = self.num_patches_mel * self.num_patches_time

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=self.embed_dim,
            kernel_size=(self.patch_mels, self.patch_time),
            stride=(self.patch_mels, self.patch_time)
        )  # [B, embed_dim, num_patches_mel, num_patches_time]
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.classifier = nn.Linear(self.embed_dim, num_classes)


    def forward(self, collated_batch) -> torch.Tensor:
        """
        Takes MelSpectrograms and returns logits for each class
        """

        # Patchify [B,1,64,128] -> [B, embed_dim, patch_freq_index=4, patch_time_index=8]
        patches = self.patch_embedding(collated_batch["spectrograms"])

        # Flatten and permute to [batch, seq_len=32, embed_dim]
        patch_embeddings = patches.flatten(2).transpose(1,2)

        B, N, D = patch_embeddings.shape
        cls_token = self.cls_token.expand(B, -1, -1) # [B, 1, D]

        input_state = torch.cat([cls_token, patch_embeddings], dim=1)  # [B, 1 + N, D]
        input_state = input_state + self.pos_embedding[:, :N+1]

        hidden_state = self.encoder(input_state)

        # The [CLS] position should contain aggregated global information from all patches 
        cls_out = hidden_state[:,0]

        return self.classifier(cls_out)

    def total_mels(self) -> int:
        return self.n_mels

    def total_time_frames(self) -> int:
        return self.time_frames

class UrbanSoundClassifierModelTrainingConfig(TrainingConfig):
    validation_fold: int = 1

class UrbanSoundProcessor(ProcessorBase):
    def __init__(self, config: UrbanSoundClassifierModelTrainingConfig, num_mels: int, time_frames: int):
        self.config = config
        self.num_mels = num_mels
        self.time_frames = time_frames

    def create_datasets(self) -> TrainingDatasets:
        validation_fold = self.config.validation_fold
        assert 1 <= validation_fold <= 10

        print(f"Preparing datasets with validation_fold = {validation_fold}...")

        train_dataset, eval_dataset = generate_urban_classifier_dataset(validation_fold, self.num_mels, self.time_frames)

        return TrainingDatasets(
            train=train_dataset,
            validation=eval_dataset,
        )

    def collate(self, dataset_batch):
        spectrograms = torch.tensor([
            item["spectrogram"] for item in dataset_batch
        ], dtype=torch.float)
        labels = torch.tensor([
            item["class_id"] for item in dataset_batch
        ], dtype=torch.long)

        return {
            "spectrograms": spectrograms,
            "labels": labels,
        }

class UrbanSoundClassifierModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: UrbanSoundClassifierModel,
            config: UrbanSoundClassifierModelTrainingConfig,
            overrides: Optional[TrainingOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

    def create_processor(self) -> ProcessorBase:
        return UrbanSoundProcessor(self.config, self.model.total_mels(), self.model.total_time_frames())

    def process_batch(self, collated_batch) -> BatchResults:
        # Move batch to the same device as the model
        collated_batch = {
            key: tensor.to(self.model.device) if isinstance(tensor, torch.Tensor) else tensor
            for key, tensor in collated_batch.items()
        }

        num_samples = len(collated_batch["labels"])
        logits = self.model(collated_batch)      # (Batch, Classes) => Logits
        label_indices = collated_batch["labels"] # (Batch) => Class index
        criterion = nn.CrossEntropyLoss()

        loss = criterion(logits, label_indices)

        return BatchResults(
            total_loss=loss,
            num_samples=num_samples,
            intermediates={
                "logits": logits,
                "label_indices": label_indices,
            }
        )

    def start_custom_validation(self) -> dict:
        return {
            "correct_predictions": 0,
        }

    def custom_validate_batch(self, custom_validation_metrics: dict, batch_results: BatchResults, batch_num: int, total_batches: int):
        best_guesses: torch.Tensor = batch_results.intermediates["logits"].argmax(axis = -1)         # (Batch) => Index
        correct_guesses: torch.Tensor = best_guesses == batch_results.intermediates["label_indices"] # (Batch) => True/False
        custom_validation_metrics["correct_predictions"] += correct_guesses.sum().item()

    def finalize_custom_validation(self, total_samples: int, total_batches: int, custom_validation_metrics: dict) -> dict:
        prediction_accuracy = custom_validation_metrics["correct_predictions"] / total_samples
        return {
            "prediction_accuracy": prediction_accuracy,
            "objective": prediction_accuracy
        }
    
if __name__ == "__main__":
   print("Run default_models instead of this file")