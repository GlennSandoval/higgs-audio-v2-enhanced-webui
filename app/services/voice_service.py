"""Voice library service for managing voice assets and transcription."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from app import audio_io, config

WHISPER_AVAILABLE: bool = False
_WHISPER_BACKEND: str | None = None
_whisper_model = None
_whisper_module = None
_faster_whisper_model_cls = None

try:
    from faster_whisper import \
        WhisperModel as _FasterWhisperModel  # type: ignore

    _WHISPER_BACKEND = "faster"
    WHISPER_AVAILABLE = True
    _faster_whisper_model_cls = _FasterWhisperModel
    print("✅ Using faster-whisper for transcription")
except ImportError:
    try:
        import whisper as _openai_whisper  # type: ignore

        _WHISPER_BACKEND = "openai"
        WHISPER_AVAILABLE = True
        _whisper_module = _openai_whisper
        print("✅ Using openai-whisper for transcription")
    except ImportError:
        WHISPER_AVAILABLE = False
        _WHISPER_BACKEND = None
        print("⚠️ Whisper not available - voice samples will use dummy text")


TranscribeFn = Callable[[str], str]


def initialize_whisper() -> None:
    """Ensure the whisper model is loaded if available."""
    global _whisper_model

    if _whisper_model is not None or not WHISPER_AVAILABLE:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if _WHISPER_BACKEND == "faster" and _faster_whisper_model_cls is not None:
            _whisper_model = _faster_whisper_model_cls("large-v3", device=device)
            print("✅ Loaded faster-whisper model")
        elif _WHISPER_BACKEND == "openai" and _whisper_module is not None:
            _whisper_model = _whisper_module.load_model("large")
            print("✅ Loaded openai-whisper model")
    except Exception as exc:
        print(f"⚠️ Failed to load whisper model: {exc}")
        try:
            if _WHISPER_BACKEND == "faster" and _faster_whisper_model_cls is not None:
                _whisper_model = _faster_whisper_model_cls("base", device=device)
            elif _WHISPER_BACKEND == "openai" and _whisper_module is not None:
                _whisper_model = _whisper_module.load_model("base")
            if _whisper_model is not None:
                print("✅ Loaded whisper base model as fallback")
        except Exception as fallback_exc:
            print(f"❌ Failed to load any whisper model: {fallback_exc}")
            _whisper_model = None


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using the configured whisper backend."""
    if not WHISPER_AVAILABLE:
        return config.WHISPER_FALLBACK_TRANSCRIPTION

    initialize_whisper()

    if _whisper_model is None:
        return config.WHISPER_FALLBACK_TRANSCRIPTION

    try:
        # faster-whisper returns (segments_generator, info) tuple
        if _WHISPER_BACKEND == "faster":
            segments, _info = _whisper_model.transcribe(audio_path, language="en")
            transcription = " ".join(segment.text for segment in segments)
        # openai-whisper returns dict with 'text' key
        else:
            result = _whisper_model.transcribe(audio_path)
            transcription = result.get("text", "")

        transcription = transcription.strip()
        if not transcription:
            transcription = config.WHISPER_FALLBACK_TRANSCRIPTION

        print(f"🎤 Transcribed: {transcription[:100]}...")
        return transcription
    except Exception as exc:
        print(f"❌ Transcription failed: {exc}")
        return config.WHISPER_FALLBACK_TRANSCRIPTION


def robust_txt_path_creation(audio_path: str) -> str:
    """
    Return the transcript file path (.txt) for a given audio file path.
    
    Handles common audio extensions (.wav, .mp3, .flac, .m4a, .ogg) case-insensitively.
    If the audio file has a recognized extension, replaces it with the transcript extension.
    Otherwise, appends the transcript extension to the original path.
    """
    audio_extensions = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    path = Path(audio_path)
    lower_name = path.name.lower()
    for ext in audio_extensions:
        if lower_name.endswith(ext):
            return str(path.with_suffix(config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION))
    # If extension is not recognized, append transcript extension to preserve original filename
    return str(path) + config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION


def create_voice_reference_txt(audio_path: str, transcript_sample: str | None = None) -> str:
    """
    Create a transcript file for a saved voice sample.
    
    If transcript_sample is provided, uses it directly; otherwise transcribes the audio file.
    Returns the path to the created transcript file.
    """
    transcript_path = robust_txt_path_creation(audio_path)
    transcript = transcript_sample if transcript_sample is not None else transcribe_audio(audio_path)

    with open(transcript_path, "w", encoding="utf-8") as handle:
        handle.write(transcript)

    print(f"📝 Created voice reference text: {transcript_path}")
    return transcript_path


class VoiceLibrary:
    """Service class for managing voice library assets and metadata."""

    def __init__(
        self,
        transcribe_fn: TranscribeFn | None = None,
        save_temp_audio_fn: Callable[[np.ndarray, int], str] = audio_io.save_temp_audio_robust,
        voice_directory: str = config.VOICE_LIBRARY_DIR,
        voice_prompts_dir: str = config.VOICE_PROMPTS_DIR,
    ) -> None:
        self._transcribe_fn = transcribe_fn or transcribe_audio
        self._save_temp_audio_fn = save_temp_audio_fn
        self._voice_directory = voice_directory
        self._voice_prompts_dir = voice_prompts_dir

    # Listing and configuration helpers -------------------------------------------------
    def list_voice_library_voices(self) -> list[str]:
        """Return the list of saved voices in the library."""
        if not os.path.exists(self._voice_directory):
            return []

        voices: list[str] = []
        for entry in os.listdir(self._voice_directory):
            if entry.endswith(config.VOICE_LIBRARY_AUDIO_EXTENSION):
                voices.append(entry.replace(config.VOICE_LIBRARY_AUDIO_EXTENSION, ""))
        return voices

    def get_voice_config_path(self, voice_name: str) -> str:
        """Return the config path for a voice."""
        return os.path.join(
            self._voice_directory,
            f"{voice_name}{config.VOICE_LIBRARY_CONFIG_SUFFIX}",
        )

    @staticmethod
    def get_default_voice_config() -> dict:
        """Return a default configuration dictionary for a voice."""
        return dict(config.DEFAULT_VOICE_CONFIG)

    def save_voice_config(self, voice_name: str, voice_config_data: dict) -> bool:
        """Persist configuration for a voice."""
        config_path = self.get_voice_config_path(voice_name)
        try:
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(voice_config_data, handle, indent=2)
            return True
        except Exception as exc:
            print(f"Error saving voice config: {exc}")
            return False

    def load_voice_config(self, voice_name: str) -> dict:
        """Load saved configuration for a voice or fall back to defaults."""
        config_path = self.get_voice_config_path(voice_name)
        if os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception as exc:
                print(f"Error loading voice config: {exc}")
        return self.get_default_voice_config()

    # CRUD operations --------------------------------------------------------------------
    def save_voice(self, audio_data: np.ndarray, sample_rate: int, voice_name: str) -> str:
        """Persist a voice sample, transcript, and default config."""
        if not voice_name or not voice_name.strip():
            return "❌ Please enter a voice name"

        sanitized_name = voice_name.strip().replace(" ", "_")
        os.makedirs(self._voice_directory, exist_ok=True)
        voice_path = os.path.join(
            self._voice_directory,
            f"{sanitized_name}{config.VOICE_LIBRARY_AUDIO_EXTENSION}",
        )

        if os.path.exists(voice_path):
            return f"❌ Voice '{sanitized_name}' already exists in library"

        try:
            temp_path = self._save_temp_audio_fn(audio_data, sample_rate)
            shutil.move(temp_path, voice_path)

            if WHISPER_AVAILABLE:
                transcript = self._transcribe_fn(voice_path)
            else:
                transcript = config.WHISPER_FALLBACK_TRANSCRIPTION
            create_voice_reference_txt(voice_path, transcript)

            default_config = self.get_default_voice_config()
            self.save_voice_config(sanitized_name, default_config)
            suffix = "" if WHISPER_AVAILABLE else " ⚠️ Saved with fallback transcript (Whisper unavailable)."
            return f"✅ Voice '{sanitized_name}' saved to library with default settings!{suffix}"
        except Exception as exc:
            return f"❌ Error saving voice: {exc}"

    def delete_voice(self, voice_name: str) -> str:
        """
        Remove a voice and its associated files from the library.
        
        Deletes the audio file, transcript, and configuration file for the specified voice.
        """
        if not voice_name or voice_name == "None":
            return "❌ Please select a voice to delete"

        voice_path = os.path.join(
            self._voice_directory,
            f"{voice_name}{config.VOICE_LIBRARY_AUDIO_EXTENSION}",
        )
        txt_path = os.path.join(
            self._voice_directory,
            f"{voice_name}{config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION}",
        )
        cfg_path = self.get_voice_config_path(voice_name)

        try:
            for path in (voice_path, txt_path, cfg_path):
                if os.path.exists(path):
                    os.remove(path)
            return f"✅ Voice '{voice_name}' deleted from library"
        except Exception as exc:
            return f"❌ Error deleting voice: {exc}"

    # Resolution helpers -----------------------------------------------------------------
    def list_all_available_voices(self) -> list[str]:
        """
        Return combined list of all available voice options.
        
        Includes Smart Voice mode, predefined voices (from voice_prompts/), 
        and user-saved library voices, each with appropriate prefix.
        """
        predefined: list[str] = []
        if os.path.exists(self._voice_prompts_dir):
            for entry in os.listdir(self._voice_prompts_dir):
                if entry.endswith((config.VOICE_LIBRARY_AUDIO_EXTENSION, ".mp3")):
                    predefined.append(entry)

        library = self.list_voice_library_voices()

        combined = [config.SMART_VOICE_LABEL]
        combined.extend(f"{config.PREDEFINED_VOICE_PREFIX}{voice}" for voice in predefined)
        combined.extend(f"{config.LIBRARY_VOICE_PREFIX}{voice}" for voice in library)
        return combined

    def get_voice_path(self, voice_selection: str | None) -> str | None:
        """
        Resolve a voice selector value to a filesystem path.
        
        Returns None for Smart Voice mode, resolves predefined and library voice
        prefixes to their respective directories.
        """
        if not voice_selection or voice_selection == config.SMART_VOICE_LABEL:
            return None

        if voice_selection.startswith(config.PREDEFINED_VOICE_PREFIX):
            voice_name = voice_selection[len(config.PREDEFINED_VOICE_PREFIX) :]
            return os.path.join(self._voice_prompts_dir, voice_name)

        if voice_selection.startswith(config.LIBRARY_VOICE_PREFIX):
            voice_name = voice_selection[len(config.LIBRARY_VOICE_PREFIX) :]
            return os.path.join(
                self._voice_directory,
                f"{voice_name}{config.VOICE_LIBRARY_AUDIO_EXTENSION}",
            )

        return None

    # Transcript utilities ---------------------------------------------------------------
    def create_voice_reference_txt(
        self, audio_path: str, transcript_sample: str | None = None
    ) -> str:
        """
        Create the transcript file for an audio reference.
        
        Delegates to the module-level create_voice_reference_txt function.
        """
        return create_voice_reference_txt(audio_path, transcript_sample)

    def robust_txt_path_creation(self, audio_path: str) -> str:
        """
        Return the transcript path for an audio file.
        
        Delegates to the module-level robust_txt_path_creation function.
        """
        return robust_txt_path_creation(audio_path)


def create_default_voice_library() -> VoiceLibrary:
    """
    Factory function that returns a VoiceLibrary instance with default configuration.
    
    Uses the default transcribe_audio function and default directory paths.
    """
    return VoiceLibrary()
