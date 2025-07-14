from dataclasses import dataclass
import torch.nn as nn
import torch
import einops
from typing import Any

from .image_encoders import ImageEncoderBase


@dataclass
class Section:
    batch_size: int

@dataclass
class CaptionSection(Section):
    section_token_ids: torch.Tensor
    """The <|section_start|> <|caption|> ...text... <|section_end|> tokens"""

@dataclass
class ImageSection(Section):
    prepared_image: Any

@dataclass
class SectionResult:
    pass

@dataclass
class CaptionSectionResult(SectionResult):
    section_logits: torch.Tensor

@dataclass
class ImageSectionResult(SectionResult):
    pass

@dataclass
class SpecialTokenIds:
    # Let's use the following structure
    # <|section_start|><|image|>   ... <|section_end|>
    # <|section_start|><|caption|> ... <|section_end|>
    section_start: int
    section_end: int
    image: int
    caption: int
    padding: int

@dataclass
class EmbeddedSection:
    section: Section
    start_offset: int
    end_offset: int

class SectionEmbedder:
    def __init__(self):
        self.embeddings: list[torch.Tensor] = []
        self.current_sequence_offset = 0
        self.current_section_start_offset = 0
        self.sections: list[EmbeddedSection] = []

    def start_section(self):
        self.current_section_start_offset = self.current_sequence_offset

    def add(self, embeddings):
        self.embeddings.append(embeddings)
        self.current_sequence_offset += embeddings.shape[1]

    def end_section(self, section):
        self.sections.append(EmbeddedSection(
            section=section,
            start_offset=self.current_section_start_offset,
            end_offset=self.current_sequence_offset,
        ))

@dataclass
class MultiModalModelResult:
    sections: list[SectionResult]
    cache: Any

class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def special_token_ids(self) -> SpecialTokenIds:
        raise NotImplementedError("Implement in super class")

    @property
    def image_encoder(self) -> ImageEncoderBase:
        raise NotImplementedError("Implement in super class")

    def tokenize_no_padding_without_special_chars(self, texts: list[str]) -> list[list[int]]:
        raise NotImplementedError("Implement in super class")

    def token_ids_to_text(self, token_ids: list[int]) -> str:
        raise NotImplementedError("Implement in super class")

    def embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in super class")

    def unembed_to_token_id_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in super class")

    def run_model(self, input_embeds: torch.Tensor, cache: Any = None) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError("Implement in super class")

    def preprocess_images(self, images) -> ImageSection:
        return ImageSection(
            batch_size=len(images),
            prepared_image=self.image_encoder.pre_process(images)
        )

    def preprocess_captions(self, captions: list[str]) -> CaptionSection:
        batch_size = len(captions)

        # We want an end token before padding, but the tokenizer doesn't add it automatically.
        # So instead we get the tokenizer to just return lists and we put it into a tensor manually
        token_id_lists = self.tokenize_no_padding_without_special_chars(captions)

        token_id_lists_max_length = max(len(ids) for ids in token_id_lists)
        section_length = token_id_lists_max_length + 3  # +3 for <|section_start|>, <|caption|>, <|section_end|>
        token_ids_tensor = torch.zeros((batch_size, section_length), dtype=torch.long)
        torch.fill(token_ids_tensor, self.special_token_ids.padding)  # Fill with padding token id
        for i, token_ids in enumerate(token_id_lists):
            token_id_length = len(token_ids)
            token_ids_tensor[i, 0] = self.special_token_ids.section_start
            token_ids_tensor[i, 1] = self.special_token_ids.caption
            for j in range(token_id_length):
                token_ids_tensor[i, j + 2] = token_ids[j]
            token_ids_tensor[i, 2 + token_id_length] = self.special_token_ids.section_end
            # We have already filled with padding ids

        return CaptionSection(
            batch_size=batch_size,
            section_token_ids=token_ids_tensor,
        )

    def embed_image_section(self, embedder: SectionEmbedder, section: ImageSection) -> None:
        batch_size = section.batch_size
        prepared_image = section.prepared_image
        tokens = self.special_token_ids
        embedder.add(self.embed_token_id(tokens.section_start, batch_size))
        embedder.add(self.embed_token_id(tokens.image, batch_size))
        embedder.add(self.image_encoder(prepared_image))
        embedder.add(self.embed_token_id(tokens.section_end, batch_size))

    def embed_caption_section(self, embedder: SectionEmbedder, section: CaptionSection) -> None:
        embedder.add(self.embed_token_ids(section.section_token_ids))

    def embed_token_id(self, token_id: int, batch_size) -> torch.Tensor:
        input = torch.tensor([token_id], dtype=torch.long)
        embedding = self.embed_token_ids(input)
        return einops.repeat(embedding, '1 embedding -> batch_size 1 embedding', batch_size=batch_size)

    def continue_forward(self, next_token_embed, cache) -> tuple[torch.Tensor, Any]:
        """Returns the logits for the next token"""
        final_hidden_state, cache = self.run_model(next_token_embed, cache)
        return self.unembed_to_token_id_logits(final_hidden_state), cache

    def forward(self, sections: list[Section]) -> MultiModalModelResult:
        # Go through each section, embed then, concatenate the embeddings, then run the model,
        # then return the results for each section.

        embedder = SectionEmbedder()
        for section in sections:
            embedder.start_section()
            match section:
                case ImageSection():
                    assert isinstance(section, ImageSection) # Fix PyCharm
                    self.embed_image_section(embedder, section)
                case CaptionSection():
                    assert isinstance(section, CaptionSection) # Fix PyCharm
                    self.embed_caption_section(embedder, section)
            embedder.end_section(section)

        # (Batch, Sequence, Embedding)
        inputs_embeds = torch.cat(embedder.embeddings, dim=-2)  # Concatenate along the sequence dimension
        final_hidden_state, cache = self.run_model(inputs_embeds)

        section_results = []

        for section_data in embedder.sections:
            match section_data.section:
                case ImageSection():
                    section_results.append(ImageSectionResult())
                case CaptionSection():
                    section_final_state = final_hidden_state[:, section_data.start_offset:section_data.end_offset, :]
                    section_results.append(CaptionSectionResult(
                        section_logits=self.unembed_to_token_id_logits(section_final_state)
                    ))
                case _:
                    raise ValueError(f"Unknown section type: {section_data.section}")

        return MultiModalModelResult(
            sections=section_results,
            cache=cache,
        )