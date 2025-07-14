import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import einops
import math
import pydantic

from model.harness import ModelBase, ModuleConfig, BatchResults, ModelTrainerBase, TrainingConfig, TrainingState, \
    TrainingOverrides, select_device_no_mps, TrainingDatasets, ProcessorBase
from .prepared_datasets import generate_speaker_tagged_dataset
from typing import Optional, Any, Literal
from .prepared_datasets import soundfile_to_whisper_embedding

class LinearSpeakerEmbeddingConfig(ModuleConfig):
    model_kind: Literal["linear"] = "linear"

class LayeredSpeakerEmbeddingConfig(ModuleConfig):
    model_kind: Literal["layered"] = "layered"

class SpeakerEmbeddingTwoTowersConfig(ModuleConfig):
    total_speakers: int # 109 in badayvedat/VCTK
    target_embedding_dimension: int
    whisper_embedding_dimension: int
    inner_model_name: str = "speaker-embedding"
    embedding_model: Optional[LinearSpeakerEmbeddingConfig | LayeredSpeakerEmbeddingConfig] = pydantic.Field(discriminator='model_kind', default=None)

class SpeakerEmbeddingTwoTowers(ModelBase):
    def __init__(self, model_name: str, config: SpeakerEmbeddingTwoTowersConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

        if self.config.embedding_model is None:
            self.config.embedding_model = LinearSpeakerEmbeddingConfig()

        match self.config.embedding_model:
            case LinearSpeakerEmbeddingConfig():
                self.speaker_embedding_model = LinearSpeakerEmbeddingModel(
                    model_name=self.config.inner_model_name,
                    config=LinearSpeakerEmbeddingModelConfig(
                        whisper_embedding_dimension=self.config.whisper_embedding_dimension,
                        target_embedding_dimension=self.config.target_embedding_dimension,
                    )
                )
            case LayeredSpeakerEmbeddingConfig():
                self.speaker_embedding_model = LayeredSpeakerEmbeddingModel(
                    model_name=self.config.inner_model_name,
                    config=LayeredSpeakerEmbeddingModelConfig(
                        whisper_embedding_dimension=self.config.whisper_embedding_dimension,
                        target_embedding_dimension=self.config.target_embedding_dimension,
                    )
                )
            case _:
                raise ValueError(f"Unknown embedding model configuration: {self.config.embedding_model}")

        self.known_speaker_embedding = nn.Embedding(config.total_speakers, config.target_embedding_dimension)

    def forward(self, collated_batch) -> tuple[torch.Tensor, torch.Tensor]:
        mean_embedding = self.speaker_embedding_model.mean_embedding_over_time_interval(
            collated_batch["whisper_embeddings"],
            collated_batch["start_offsets_ms"],
            collated_batch["end_offsets_ms"],
        )
        known_speaker_embeddings = self.known_speaker_embedding(collated_batch["speaker_indices"])

        return mean_embedding, known_speaker_embeddings
    
class SpeakerEmbeddingModel(ModelBase):
    def __init__(self, model_name: str, config: ModuleConfig):
        super().__init__(model_name=model_name, config=config)

    def process(self, soundfiles: list[tuple[np.ndarray, int]]) -> torch.Tensor:
        return torch.cat(
            [soundfile_to_whisper_embedding(soundfile) for soundfile in soundfiles],
            dim = 0 # Batch dimension
        )
    
    def single_mean_embedding_over_time_interval(self, whisper_embedding, length_seconds):
        return self.mean_embedding_over_time_interval(
            whisper_embeddings=whisper_embedding.unsqueeze(0),  # Add batch dimension
            start_offsets_ms=torch.tensor([0], dtype=torch.long),
            end_offsets_ms=torch.tensor([(length_seconds * 1000)//1], dtype=torch.long),
        ).squeeze(0)

    def mean_embedding_over_time_interval(self, whisper_embeddings, start_offsets_ms, end_offsets_ms) -> torch.Tensor:
        frame_rate = 50 # From whisper encodings

        # Speed up model by slicing the inputs
        max_offset = max((end_offset_ms.item() * frame_rate) // 1000 for end_offset_ms in end_offsets_ms)
        model_embeddings = self(whisper_embeddings[:, :max_offset, :])

        batch_size, time_size, embed_size = model_embeddings.shape

        mask = torch.zeros(batch_size, time_size, dtype=torch.long, device=model_embeddings.device)
        
        for row_mask, start_offset_ms, end_offset_ms in zip(mask, start_offsets_ms, end_offsets_ms):
            start_offset = (start_offset_ms * frame_rate) // 1000
            end_offset = (end_offset_ms * frame_rate) // 1000
            row_mask[start_offset:end_offset] = 1

        mask_repeated = einops.repeat(mask, 'batch time -> batch time embed', embed = embed_size)
        # (Batch, Time, Embedding)
        return (mask_repeated * model_embeddings).sum(dim=1) / mask_repeated.sum(dim=1)

class LinearSpeakerEmbeddingModelConfig(ModuleConfig):
    whisper_embedding_dimension: int
    target_embedding_dimension: int

class LinearSpeakerEmbeddingModel(SpeakerEmbeddingModel):
    def __init__(self, model_name: str, config: LinearSpeakerEmbeddingModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        self.fc1 = nn.Linear(config.whisper_embedding_dimension, config.target_embedding_dimension, bias=False)

    def forward(self, whisper_embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc1(whisper_embeddings)

class LayeredSpeakerEmbeddingModelConfig(ModuleConfig):
    whisper_embedding_dimension: int
    target_embedding_dimension: int

class LayeredSpeakerEmbeddingModel(SpeakerEmbeddingModel):
    def __init__(self, model_name: str, config: LayeredSpeakerEmbeddingModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        d_in = config.whisper_embedding_dimension
        d_mid = config.target_embedding_dimension * 2
        d_out = config.target_embedding_dimension
        # MLP: linear -> ReLU -> BN -> dropout -> linear
        self.fc1 = nn.Linear(d_in, d_mid)
        self.act1 = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(d_mid)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(d_mid, d_out)

    def forward(self, whisper_embeddings: torch.Tensor) -> torch.Tensor:
        # whisper_embeddings: [B, T, d_in]
        x = self.fc1(whisper_embeddings)           # [B, T, d_mid]
        x = self.act1(x)
        # BN over channels: reshape to [B*T, d_mid]
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.bn(x)
        x = x.view(B, T, C)
        x = self.drop2(x)
        x = self.fc2(x)                           # [B, T, d_out]
        # L2 normalize per time-step
        x = F.normalize(x, p=2, dim=-1)
        return x

class VctkProcessor(ProcessorBase):
    def create_datasets(self) -> TrainingDatasets:
        import datasets
        datasets.config.IN_MEMORY_MAX_SIZE = 2 * 1024 * 1024 # 2GB
        train_dataset, eval_dataset, _ = generate_speaker_tagged_dataset()
        return TrainingDatasets(
            train=train_dataset,
            validation=eval_dataset,
        )

    def collate(self, dataset_batch) -> dict:
        whisper_embeddings = torch.cat([
            torch.tensor(item["whisper_embedding"], dtype=torch.float) for item in dataset_batch
        ])
        start_offsets_ms = torch.tensor([
            item["start_offset_ms"] for item in dataset_batch
        ], dtype=torch.long)
        end_offsets_ms = torch.tensor([
            item["end_offset_ms"] for item in dataset_batch
        ], dtype=torch.long)
        speaker_indices = torch.tensor([
            item["speaker_index"] for item in dataset_batch
        ], dtype=torch.long)

        return {
            "whisper_embeddings": whisper_embeddings,
            "speaker_indices": speaker_indices,
            "start_offsets_ms": start_offsets_ms,
            "end_offsets_ms": end_offsets_ms,
        }


class SpeakerEmbeddingTrainingConfig(TrainingConfig):
    loss_kind: Literal["triplet", "david"] = "david"

class SpeakerEmbeddingModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: SpeakerEmbeddingTwoTowers,
            config: SpeakerEmbeddingTrainingConfig,
            overrides: Optional[TrainingOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

    def create_processor(self) -> ProcessorBase:
        return VctkProcessor()

    def process_batch(self, collated_batch) -> BatchResults:
        # model_embeddings: [Batch, Time=1500, OurEmbedding=8]
        # known_speaker_embeddings: [Batch, OurEmbedding=8]

        model_device = self.model.get_device()
        collated_batch = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in collated_batch.items()
        }
        batch_size = len(collated_batch["whisper_embeddings"])

        mean_model_embeddings, known_speaker_embeddings = self.model(collated_batch)

        assert mean_model_embeddings.shape == (batch_size, self.model.config.target_embedding_dimension)

        ### ?! The model embeddings are almost the same for almost all audios

        match self.config.loss_kind:
            case "david":
                margin = 0.85
                # Does the model's embedding align with the actual speaker embedding?
                positive_contribution = torch.nn.CosineSimilarity(dim=-1)(mean_model_embeddings, known_speaker_embeddings)

                # Does the model's embedding align with other speakers?
                negative_contributions = []
                for speaker_embedding in self.model.known_speaker_embedding.weight:
                    repeated_speaker_embedding = speaker_embedding.repeat(batch_size, 1)
                    negative_contributions.append(
                        torch.nn.CosineSimilarity(dim=-1)(mean_model_embeddings, repeated_speaker_embedding)
                    )
                negative_contributions = torch.stack(negative_contributions) # (EachSpeaker, Batch)
                negative_contribution = negative_contributions.mean(dim=0)   # (Batch)

                total_loss = torch.max(torch.tensor(0), margin - positive_contribution + negative_contribution).sum(dim=0)
            case "triplet":
                device = self.model.get_device()
                all_indices = torch.arange(batch_size, device=device)
                neg_indices = (all_indices + torch.randint(1, batch_size, (batch_size,), device=device)) % batch_size
                known_embs_neg = known_speaker_embeddings[neg_indices]
                # compute triplet loss: anchor=mean_model_embs, pos=known_embs_pos, neg=known_embs_neg
                triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)
                total_loss = triplet_loss(mean_model_embeddings, known_speaker_embeddings, known_embs_neg)
            case _:
                raise ValueError(f"Unknown loss function {self.config.loss_kind}")

        return BatchResults(
            total_loss=total_loss,
            num_samples=batch_size,
            intermediates={
                "mean_model_embeddings": mean_model_embeddings,
                "speaker_indices": collated_batch["speaker_indices"],
            }
        )

    def start_custom_validation(self) -> dict:
        return {
            "correct_predictions": 0,
        }

    def custom_validate_batch(self, custom_validation_metrics: dict, batch_results: BatchResults, batch_num: int, total_batches: int):
        mean_model_embeddings = batch_results.intermediates["mean_model_embeddings"]
        speaker_indices = batch_results.intermediates["speaker_indices"]

        correct = 0
        for embedding, actual_speaker_index in zip(mean_model_embeddings, speaker_indices):
            repeated_embedding = embedding.expand_as(self.model.known_speaker_embedding.weight)
            guessed_id = torch.nn.CosineSimilarity(dim=-1)(repeated_embedding, self.model.known_speaker_embedding.weight).argmax().item()
            if actual_speaker_index == guessed_id:
                correct += 1

        custom_validation_metrics["correct_predictions"] += correct

    def finalize_custom_validation(self, total_samples: int, total_batches: int, custom_validation_metrics: dict) -> dict:
        prediction_accuracy = custom_validation_metrics["correct_predictions"] / total_samples
        return {
            "prediction_accuracy": prediction_accuracy,
            "objective": prediction_accuracy
        }
    
if __name__ == "__main__":
   print("Run default_models instead of this file")