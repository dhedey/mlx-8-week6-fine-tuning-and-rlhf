import torch.nn as nn
import torch
import einops
import PIL
import transformers
from .encoder import SelfEncoderBlock, SelfEncoderBlockConfig
from ..harness import ModuleConfig


class PatchEmbedder(nn.Module):
    def __init__(
            self,
            image_width: int,
            image_height: int,
            patch_width: int,
            patch_height: int,
            embedding_dimension: int,
    ):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height

        vertical_patches = image_height // patch_height
        horizontal_patches = image_width // patch_width
        total_patches = vertical_patches * horizontal_patches

        self.patch_embedding = nn.Linear(patch_width * patch_height, embedding_dimension)
        self.patch_positional_bias = nn.Parameter(
            torch.zeros([total_patches, embedding_dimension], dtype=torch.float32),
        )

    def forward(self, image):
        flattened_image_patches = einops.rearrange(
            image,
            'batch 1 (h_patches patch_height) (w_patches patch_width) -> batch (h_patches w_patches) (patch_height patch_width)',
            patch_width=self.patch_width,
            patch_height=self.patch_height
        )  # Shape: (batch_size, channel, total_patches, flattened_patch_size)

        embedded_patches = self.patch_embedding(
            flattened_image_patches)  # Shape: (batch_size, encoder_blocks, encoder_embedding_size)

        return embedded_patches + self.patch_positional_bias


class PatchBasedImageEncoderConfig(ModuleConfig):
    image_width: int
    image_height: int
    image_patch_width: int
    image_patch_height: int
    embedding_dimension: int
    encoder_block_count: int
    encoder_block: SelfEncoderBlockConfig

    def __post_init__(self):
        if self.image_width % self.image_patch_width != 0 or self.image_height % self.image_patch_height != 0:
            raise ValueError("Image dimensions must be divisible by block dimensions.")


class PatchBasedImageEncoder(nn.Module):
    def __init__(self, config: PatchBasedImageEncoderConfig):
        super().__init__()
        self.patch_embedder = PatchEmbedder(
            image_width=config.image_width,
            image_height=config.image_height,
            patch_width=config.image_patch_width,
            patch_height=config.image_patch_height,
            embedding_dimension=config.embedding_dimension,
        )

        self.encoder_blocks = nn.ModuleList([
            SelfEncoderBlock(config.encoder_block)
            for _ in range(config.encoder_block_count)
        ])

    def forward(self, image):
        residual_stream = self.patch_embedder(image)  # Shape: (batch_size, patches, embedding_size)

        for encoder_block in self.encoder_blocks:
            residual_stream = encoder_block(residual_stream)

        return residual_stream


class ImageEncoderBase(nn.Module):
    def __init__(self):
        super().__init__()

    def embedding_dimension(self) -> int:
        raise NotImplementedError("Should be implemented in a subclass.")

    def pre_process(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")

    def forward(self, preprocessed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")


class ClipImageEncoderConfig(ModuleConfig):
    model_name: str = "openai/clip-vit-base-patch32"
    tokens_per_image: int
    model_embedding_dimension: int
    freeze_visual_model: bool = True


class ClipImageEncoder(ImageEncoderBase):
    def __init__(self, config: ClipImageEncoderConfig):
        super().__init__()
        self.processor = transformers.CLIPProcessor.from_pretrained(config.model_name, use_fast=True)
        self.model = transformers.CLIPModel.from_pretrained(config.model_name)
        self.model.text_model = None  # Remove unneeded weights

        if config.freeze_visual_model:
            self.model.requires_grad_(False)

        self.config = config
        self.linear_mapping = nn.Linear(
            self.model.config.projection_dim,
            config.tokens_per_image * config.model_embedding_dimension,
        )

    def pre_process(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        # Pre-processing resizes all images to the expected size
        # (batch_size, 3, height = 244, width = 244)
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def forward(self, preprocessed: torch.Tensor) -> torch.Tensor:
        preprocessed = preprocessed.to(self.model.device)

        # (batch_size, model_projection_dim) => (batch_size, image_embedding_dimension)
        image_vector = self.model.get_image_features(pixel_values=preprocessed)
        # (batch_size, model_projection_dim) => (batch_size, tokens_per_image * model_embedding_dimension)
        outputs = self.linear_mapping(image_vector)
        # (batch_size, tokens_per_image, model_embedding_dimension)
        return einops.rearrange(
            outputs,
            "batch (tokens_per_image embedding_dimension) -> batch tokens_per_image embedding_dimension",
            embedding_dimension=self.config.model_embedding_dimension,
        )