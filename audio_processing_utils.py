"""
Audio Processing Utilities for Higgs Audio WebUI
Handles volume normalization, audio enhancement, and multi-speaker audio processing.
"""

import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_audio_volume(
    audio: Union[np.ndarray, torch.Tensor], 
    target_rms: float = 0.1,
    max_gain: float = 10.0,
    sample_rate: int = 16000
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize audio volume to a target RMS level.
    
    Args:
        audio: Audio data as numpy array or torch tensor
        target_rms: Target RMS (Root Mean Square) level for normalization
        max_gain: Maximum gain multiplier to prevent extreme amplification
        sample_rate: Sample rate of the audio
        
    Returns:
        Normalized audio with same type as input
    """
    is_torch = isinstance(audio, torch.Tensor)
    
    # Convert to numpy for processing if needed
    if is_torch:
        audio_np = audio.cpu().numpy()
        original_device = audio.device
    else:
        audio_np = audio.copy()
    
    # Handle stereo/mono audio
    if len(audio_np.shape) > 1:
        # For multi-channel audio, work with the mean
        audio_mono = np.mean(audio_np, axis=0)
    else:
        audio_mono = audio_np
    
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(audio_mono ** 2))
    
    if current_rms == 0:
        logger.warning("Audio has zero RMS - skipping normalization")
        return audio
    
    # Calculate required gain
    gain = target_rms / current_rms
    
    # Limit gain to prevent extreme amplification
    gain = min(gain, max_gain)
    
    # Apply gain
    normalized_audio = audio_np * gain
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized_audio))
    if max_val > 1.0:
        normalized_audio = normalized_audio / max_val * 0.99
    
    logger.info(f"Audio normalization: RMS {current_rms:.3f} → {np.sqrt(np.mean(normalized_audio ** 2)):.3f}, Gain: {gain:.2f}x")
    
    # Convert back to original type
    if is_torch:
        return torch.tensor(normalized_audio, device=original_device, dtype=audio.dtype)
    else:
        return normalized_audio


def normalize_multi_speaker_segments(
    audio: Union[np.ndarray, torch.Tensor],
    speaker_timestamps: list,
    target_rms: float = 0.1,
    sample_rate: int = 16000
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize volume for different speaker segments in multi-speaker audio.
    
    Args:
        audio: Full audio data
        speaker_timestamps: List of (start_time, end_time, speaker_id) tuples
        target_rms: Target RMS level for all segments
        sample_rate: Sample rate of the audio
        
    Returns:
        Audio with normalized speaker segments
    """
    is_torch = isinstance(audio, torch.Tensor)
    
    # Convert to numpy for processing
    if is_torch:
        audio_np = audio.cpu().numpy()
        original_device = audio.device
    else:
        audio_np = audio.copy()
    
    # Process each speaker segment
    for start_time, end_time, speaker_id in speaker_timestamps:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract segment
        segment = audio_np[start_sample:end_sample]
        
        if len(segment) > 0:
            # Normalize segment
            normalized_segment = normalize_audio_volume(
                segment, target_rms=target_rms, sample_rate=sample_rate
            )
            
            # Replace in original audio
            audio_np[start_sample:end_sample] = normalized_segment
    
    # Convert back to original type
    if is_torch:
        return torch.tensor(audio_np, device=original_device, dtype=audio.dtype)
    else:
        return audio_np


def adaptive_volume_normalization(
    audio: Union[np.ndarray, torch.Tensor],
    window_size: float = 2.0,
    overlap: float = 0.5,
    target_rms: float = 0.1,
    sample_rate: int = 16000
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply adaptive volume normalization using sliding windows.
    This is useful for multi-speaker audio where volume changes frequently.
    
    Args:
        audio: Audio data
        window_size: Size of each normalization window in seconds
        overlap: Overlap between windows (0.0 to 1.0)
        target_rms: Target RMS level
        sample_rate: Sample rate of the audio
        
    Returns:
        Audio with adaptive volume normalization
    """
    is_torch = isinstance(audio, torch.Tensor)
    
    # Convert to numpy for processing
    if is_torch:
        audio_np = audio.cpu().numpy()
        original_device = audio.device
    else:
        audio_np = audio.copy()
    
    # Calculate window parameters
    window_samples = int(window_size * sample_rate)
    hop_samples = int(window_samples * (1 - overlap))
    
    # Create output array
    output_audio = np.zeros_like(audio_np)
    weight_sum = np.zeros_like(audio_np)
    
    # Process each window
    for start in range(0, len(audio_np) - window_samples + 1, hop_samples):
        end = start + window_samples
        
        # Extract window
        window = audio_np[start:end]
        
        # Normalize window
        normalized_window = normalize_audio_volume(
            window, target_rms=target_rms, sample_rate=sample_rate
        )
        
        # Apply window function (Hann window for smooth blending)
        hann_window = np.hanning(window_samples)
        normalized_window = normalized_window * hann_window
        
        # Add to output with overlap-add
        output_audio[start:end] += normalized_window
        weight_sum[start:end] += hann_window
    
    # Normalize by weight sum to account for overlaps
    output_audio = np.divide(output_audio, weight_sum, 
                           out=np.zeros_like(output_audio), 
                           where=weight_sum != 0)
    
    # Handle edges that might not be covered
    if weight_sum[0] == 0:
        output_audio[:hop_samples] = normalize_audio_volume(
            audio_np[:hop_samples], target_rms=target_rms
        )
    
    if weight_sum[-1] == 0:
        output_audio[-hop_samples:] = normalize_audio_volume(
            audio_np[-hop_samples:], target_rms=target_rms
        )
    
    logger.info(f"Applied adaptive normalization with {window_size}s windows")
    
    # Convert back to original type
    if is_torch:
        return torch.tensor(output_audio, device=original_device, dtype=audio.dtype)
    else:
        return output_audio


def detect_speaker_segments(
    audio: Union[np.ndarray, torch.Tensor],
    min_segment_length: float = 0.5,
    energy_threshold: float = 0.01,
    sample_rate: int = 16000
) -> list:
    """
    Detect speaker segments based on energy levels.
    This is a simple approach - more advanced methods could use voice activity detection.
    
    Args:
        audio: Audio data
        min_segment_length: Minimum segment length in seconds
        energy_threshold: Energy threshold for speech detection
        sample_rate: Sample rate of the audio
        
    Returns:
        List of (start_time, end_time) tuples for detected speech segments
    """
    # Convert to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio
    
    # Calculate frame-wise energy
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    energy = []
    for i in range(0, len(audio_np) - frame_length + 1, hop_length):
        frame = audio_np[i:i + frame_length]
        frame_energy = np.mean(frame ** 2)
        energy.append(frame_energy)
    
    energy = np.array(energy)
    
    # Detect speech segments
    speech_frames = energy > energy_threshold
    
    # Find segment boundaries
    segments = []
    in_segment = False
    segment_start = 0
    
    for i, is_speech in enumerate(speech_frames):
        time = i * hop_length / sample_rate
        
        if is_speech and not in_segment:
            # Start of new segment
            segment_start = time
            in_segment = True
        elif not is_speech and in_segment:
            # End of segment
            segment_length = time - segment_start
            if segment_length >= min_segment_length:
                segments.append((segment_start, time))
            in_segment = False
    
    # Handle case where audio ends during a segment
    if in_segment:
        final_time = len(audio_np) / sample_rate
        segment_length = final_time - segment_start
        if segment_length >= min_segment_length:
            segments.append((segment_start, final_time))
    
    logger.info(f"Detected {len(segments)} speaker segments")
    return segments


def enhance_multi_speaker_audio(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    normalization_method: str = "adaptive",
    target_rms: float = 0.1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Main function to enhance multi-speaker audio with volume normalization.
    
    Args:
        audio: Input audio data
        sample_rate: Sample rate of the audio
        normalization_method: "simple", "adaptive", or "segment-based"
        target_rms: Target RMS level for normalization
        
    Returns:
        Enhanced audio with normalized volumes
    """
    logger.info(f"Enhancing multi-speaker audio using {normalization_method} normalization")
    
    if normalization_method == "simple":
        return normalize_audio_volume(audio, target_rms=target_rms, sample_rate=sample_rate)
    
    elif normalization_method == "adaptive":
        return adaptive_volume_normalization(
            audio, 
            window_size=2.0, 
            overlap=0.5, 
            target_rms=target_rms, 
            sample_rate=sample_rate
        )
    
    elif normalization_method == "segment-based":
        # Detect speaker segments and normalize each
        segments = detect_speaker_segments(audio, sample_rate=sample_rate)
        
        # Convert segments to the format expected by normalize_multi_speaker_segments
        speaker_timestamps = [(start, end, f"speaker_{i}") for i, (start, end) in enumerate(segments)]
        
        return normalize_multi_speaker_segments(
            audio, 
            speaker_timestamps, 
            target_rms=target_rms, 
            sample_rate=sample_rate
        )
    
    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")


def save_normalized_audio(
    audio: Union[np.ndarray, torch.Tensor],
    output_path: str,
    sample_rate: int = 16000
) -> None:
    """
    Save normalized audio to file.
    
    Args:
        audio: Audio data to save
        output_path: Path to save the audio file
        sample_rate: Sample rate for the output file
    """
    # Convert to torch tensor if needed
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    else:
        audio_tensor = audio.float()
    
    # Ensure audio is 2D (channels, samples)
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Save audio
    torchaudio.save(output_path, audio_tensor, sample_rate)
    logger.info(f"Saved normalized audio to {output_path}") 