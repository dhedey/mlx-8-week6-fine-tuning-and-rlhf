import datasets
import os
import huggingface_hub
import torchaudio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Any
import numpy as np
import wandb

from torchgen.utils import OrderedSet
from transformers import WhisperProcessor, WhisperModel
import torch
import torchaudio
from model.harness import datasets_cache_folder, select_device, select_device_no_mps

def noop_collate(batch):
    return batch

def resample_and_trim_or_pad(soundfile, target_sample_rate, target_num_samples) -> torch.Tensor:
    soundfile_array, sample_rate = soundfile["array"], soundfile["sampling_rate"]

    if soundfile_array.ndim == 1:
        waveform = torch.from_numpy(soundfile_array).to(torch.float32).unsqueeze(0)  # [1, time]
    else:
        waveform = torch.from_numpy(soundfile_array).to(torch.float32).T  # [channels, time]

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate).to('cpu')
        waveform = resampler(waveform)

    # Trim or pad
    num_samples = waveform.shape[-1]
    if num_samples > target_num_samples:
        waveform = waveform[..., :target_num_samples]
    elif num_samples < target_num_samples:
        waveform = F.pad(waveform, (0, target_num_samples - num_samples))

    return waveform.to('cpu')

def urbansound8K_prepare_v2(dataset_item, num_mels: Optional[int] = None, total_time_frames: Optional[int] = None):
    if num_mels is None:
        num_mels = 64
    if total_time_frames is None:
        # SAMPLE_RATE = 16000 corresponds to total_time_periods = 321
        total_time_frames = 321

    FIXED_LENGTH_SECONDS = 4

    target_num_samples = 200 * (total_time_frames - 1) # + 400
    target_sample_rate = target_num_samples // FIXED_LENGTH_SECONDS

    waveform = resample_and_trim_or_pad(
        soundfile=dataset_item["audio"],
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
    )

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_mels=num_mels,
    ).to('cpu')  # Ensure transform is on CPU
    spectrogram = transform(waveform)
    spectrogram = spectrogram.to('cpu')

    expected_shape = (1, num_mels, total_time_frames) # 1 = channel
    assert spectrogram.shape == expected_shape, f"Expected spectrogram to have shape {expected_shape}, but got {spectrogram.shape}"

    # elapsed_seconds = dataset_item["end"] - dataset_item["start"]
    #
    # plt.figure(figsize=(10, 4))
    # plt.imshow(spectrogram.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='magma')
    # plt.colorbar(format='%+2.0f dB')
    # plt.xlabel('Time')
    # plt.ylabel('Mel Frequency Bin')
    # plt.title(f'Mel Spectrogram ({elapsed_seconds}/4 seconds)')
    # plt.tight_layout()
    # plt.show()

    return {
        "spectrogram": spectrogram,
        "class_id": dataset_item["classID"],
        "class": dataset_item["class"],
        "salience": dataset_item["salience"],
    }

def generate_urban_classifier_dataset(validation_fold: int, num_mels: Optional[int] = None, total_time_frames: Optional[int] = None):
    assert 1 <= validation_fold <= 10

    dataset = datasets.load_dataset(
        "danavery/urbansound8K",
        cache_dir=datasets_cache_folder(),
        # It only comes with one split "train" but advises we do cross validation based on the fold
        split="train",
    )

    train_dataset = dataset.filter(lambda x: x["fold"] != validation_fold)
    train_dataset = train_dataset.map(lambda x: urbansound8K_prepare_v2(x, num_mels=num_mels, total_time_frames=total_time_frames), remove_columns=dataset.column_names)
    eval_dataset = dataset.filter(lambda x: x["fold"] == validation_fold)
    eval_dataset = eval_dataset.map(lambda x: urbansound8K_prepare_v2(x, num_mels=num_mels, total_time_frames=total_time_frames), remove_columns=dataset.column_names)

    return train_dataset, eval_dataset

_preloaded_whisper = None

def get_whisper() -> tuple[WhisperProcessor, WhisperModel]:
    global _preloaded_whisper
    if _preloaded_whisper is None:
        print("Loading whisper...")
        device = select_device_no_mps()
        _preloaded_whisper = (WhisperProcessor.from_pretrained("openai/whisper-tiny"), WhisperModel.from_pretrained("openai/whisper-tiny").to(device))
    return _preloaded_whisper

_vctk_speaker_id_mapping = {}

def soundfile_to_whisper_embedding(soundfile: tuple[np.ndarray, int]):
    # MPS is not supported by Whisper - we get:
    # > NotImplementedError: Output channels > 65536 not supported at the MPS device.
    device = select_device_no_mps()

    waveform_np, sample_rate = soundfile
    waveform = torch.from_numpy(waveform_np).to(torch.float).to(device)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    # Whisper expects mono audio
    if waveform.shape[0] > 1:
       waveform = waveform.mean(dim=0, keepdim=True)

    # TODO: Add in randomization of the audio

    # Load model and processor
    processor, model = get_whisper()

    # Prepare input for Whisper
    inputs = processor(waveform.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt")

    # Get encoder hidden states (embeddings)
    with torch.no_grad():
        encoder_outputs = model.encoder(inputs.input_features.to(device))

    # encoder_outputs.last_hidden_state shape: [batch, time, hidden_dim]
    # Note: Outputs 50 embeddings / sec, over 30 seconds for a total of 1500
    whisper_embedding = encoder_outputs.last_hidden_state  # This is the sequence of embeddings

    return whisper_embedding


def prepare_vctk_v2(item):
    global _vctk_speaker_id_mapping
    soundfile = item["flac"]

    whisper_embedding = soundfile_to_whisper_embedding((soundfile["array"], soundfile["sampling_rate"]))
    length_ms = (1000 * soundfile["array"].shape[-1]) // soundfile["sampling_rate"]

    speaker_id = item["speaker_id"]
    if speaker_id not in _vctk_speaker_id_mapping:
        _vctk_speaker_id_mapping[speaker_id] = len(_vctk_speaker_id_mapping)

    speaker_index = _vctk_speaker_id_mapping[speaker_id]

    return {
        "whisper_embedding": whisper_embedding, # [batch, time, hidden_dim]
        "speaker_index": speaker_index,
        "start_offset_ms": 0,
        "end_offset_ms": length_ms,
    }

_counter = 0
def every_100th(item):
    global _counter
    is_included = (_counter % 100 == 0)
    _counter += 1
    return is_included

def every_10th(item):
    global _counter
    is_included = (_counter % 10 == 0)
    _counter += 1
    return is_included

def generate_speaker_tagged_dataset(
    cache_dir: str = "datasets/vctk_processed_v2",
    raw_cache_dir: str = "datasets/vctk_raw",
    artifact_name: str = "vctk-processed-dataset-v2",
    use_alias: str = "latest",
):
    # 0. Check if we're logged in to W&B
    use_wandb = wandb.run is not None

    if not use_wandb:
        print("🔒 Not logged into W&B; skipping artifact download & logging.")

    # Resolve cache_dir relative to this file
    base_path = os.path.dirname(__file__)
    cache_dir = os.path.abspath(os.path.join(base_path, '..', cache_dir))
    raw_cache_dir = os.path.abspath(os.path.join(base_path, '..', raw_cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(raw_cache_dir, exist_ok=True)

    # 1. Try loading existing processed dataset from disk
    try:
        print(f"Loading processed dataset from local cache: {cache_dir}")
        ds = datasets.load_from_disk(cache_dir)
        train, valid = ds["train"], ds["validation"]
        return train, valid, 0
    except Exception as e:
        print(f"Local cache miss: {e}")
    
    # 2. If logged in, try fetching the artifact
    if use_wandb:
        try:
            print("Fetching processed dataset from W&B…")
            artifact_ref = f"{artifact_name}:{use_alias}"
            print(f"Fetching {artifact_ref} into {cache_dir}…")
            art = wandb.use_artifact(artifact_ref)
            download_root = art.download(root=cache_dir)
            print(f"✅ Artifact downloaded into {download_root}")
            ds = datasets.load_from_disk(download_root)
            train, valid = ds["train"], ds["validation"]
            return train, valid, 0
        except Exception as e:
            print(f"⚠️ Artifact fetch failed: {e}")
            print("Falling back to local rebuild…")
    else:
        print("Skipping W&B fetch; rebuilding locally.")
    
    # 2. Download and process if not cached
    print("Downloading and processing VCTK dataset...")
    raw_dataset = datasets.load_dataset("badayvedat/VCTK", cache_dir=raw_cache_dir)
    
    # ... your processing steps here ...
    train = raw_dataset["train"].filter(every_10th).map(prepare_vctk_v2, remove_columns=raw_dataset["train"].column_names)
    eval = raw_dataset["validation"].filter(every_10th).map(prepare_vctk_v2, remove_columns=raw_dataset["validation"].column_names)
    
    # 3. Save processed dataset to disk for future use
    processed = datasets.DatasetDict({"train": train, "validation": eval})
    processed.save_to_disk(cache_dir)
    print(f"Saved processed dataset to cache: {cache_dir}")

    if use_wandb:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description="Speaker-tagged VCTK train/validation split"
        )
        artifact.add_dir(cache_dir, name=os.path.basename(cache_dir))
        wandb.log_artifact(artifact)
        print(f"✅ Logged processed dataset as `{artifact_name}:latest`")

    return train, eval, 0


if __name__ == "__main__":
    dataset = datasets.load_dataset("badayvedat/VCTK", cache_dir=datasets_cache_folder())
    _ignored = dataset["train"].map(prepare_vctk_v2, remove_columns=dataset["train"].column_names)
    # generate_speaker_tagged_dataset()
    # generate_urban_classifier_dataset(1)