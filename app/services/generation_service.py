"""Generation orchestration service for the Higgs Audio application."""

from __future__ import annotations

import gc
import hashlib
import os
import re
import time
from collections.abc import Sequence

import numpy as np
import torch
import torchaudio

from app import audio_io, config, startup
from app.audio import (AudioBuffer, enhance_multi_speaker_audio,
                       normalize_audio_volume)
from app.services.cache import GenerationCache
from app.services.voice_service import VoiceLibrary, transcribe_audio
from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
from boson_multimodal.serve.serve_engine import (HiggsAudioResponse,
                                                 HiggsAudioServeEngine)


class GenerationService:
    """Encapsulates audio generation flows and Higgs engine lifecycle management."""

    def __init__(
        self,
        device: torch.device,
        voice_library_service: VoiceLibrary,
    ) -> None:
        self._device = device
        self._voice_library = voice_library_service
        self._serve_engine: HiggsAudioServeEngine | None = None
        audio_cache, token_cache = startup.initialize_caches()
        self._cache = GenerationCache(audio=audio_cache, tokens=token_cache)

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------
    def clear_caches(self) -> None:
        """Clear audio/token caches and free GPU memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Cleared caches and freed memory")

    @staticmethod
    def _seed_if_needed(seed: int) -> None:
        if seed > 0:
            torch.manual_seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _get_cache_key(
        messages: Sequence[Message],
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float | None,
        repetition_penalty: float,
        ras_win_len: int,
        ras_win_max_num_repeat: int,
        do_sample: bool,
    ) -> str:
        cache_payload = (
            f"{messages}_{temperature}_{top_k}_{top_p}_{min_p}_"
            f"{repetition_penalty}_{ras_win_len}_{ras_win_max_num_repeat}_{do_sample}"
        )
        return hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()

    def _ensure_engine(self) -> None:
        if self._serve_engine is not None:
            return

        print("🚀 Initializing Higgs Audio model...")
        self._serve_engine = HiggsAudioServeEngine(
            config.MODEL_ID,
            config.AUDIO_TOKENIZER_ID,
            device=self._device,
            torch_dtype=torch.bfloat16, 
        )
        print("✅ Model initialized successfully")

    def _prepare_system_message(self, scene_description: str = "") -> str:
        system_content = config.DEFAULT_SYSTEM_MESSAGE
        if scene_description and scene_description.strip():
            system_content += (
                f" {config.SCENE_DESC_START_TAG}\n"
                f"{scene_description}\n"
                f"{config.SCENE_DESC_END_TAG}"
            )
        return system_content

    def _generate_with_cache(
        self,
        messages: Sequence[Message],
        max_new_tokens: int,
        temperature: float,
        top_k: int = config.DEFAULT_TOP_K,
        top_p: float = config.DEFAULT_TOP_P,
        min_p: float | None = None,
        repetition_penalty: float = config.DEFAULT_REPETITION_PENALTY,
        ras_win_len: int = config.DEFAULT_RAS_WIN_LEN,
        ras_win_max_num_repeat: int = config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
        do_sample: bool = config.DEFAULT_DO_SAMPLE,
        *,
        use_cache: bool = True,
    ) -> HiggsAudioResponse:
        """Generate audio with optional caching.
        
        Only temperature, top_k, top_p, and ras_win_* parameters are currently used
        by the serve engine. min_p and repetition_penalty are included for cache keys
        and future compatibility but not passed to generation.
        """
        self._ensure_engine()

        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(
                messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                do_sample=do_sample,
            )
            if cache_key in self._cache.audio:
                print("🚀 Using cached audio result")
                return self._cache.audio[cache_key]

        generate_kwargs = {
            "chat_ml_sample": ChatMLSample(messages=list(messages)),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop_strings": list(config.STOP_STRINGS),
            "ras_win_len": ras_win_len,
            "ras_win_max_num_repeat": ras_win_max_num_repeat,
        }

        start = time.time()
        output: HiggsAudioResponse = self._serve_engine.generate(**generate_kwargs)  # type: ignore[arg-type]
        print(f"Generation took: {time.time() - start:.2f}s")

        if use_cache and cache_key:
            self._cache.audio[cache_key] = output
            if len(self._cache.audio) > config.DEFAULT_CACHE_MAX_ENTRIES:
                oldest_key = next(iter(self._cache.audio))
                del self._cache.audio[oldest_key]

        return output

    # ------------------------------------------------------------------
    # Voice library integration
    # ------------------------------------------------------------------
    def apply_voice_config_to_generation(
        self,
        voice_selection: str | None,
        transcript: str,
        scene_description: str = "",
        force_audio_gen: bool = False,
    ) -> HiggsAudioResponse | None:
        if not voice_selection or voice_selection == config.SMART_VOICE_LABEL:
            return None

        voice_name: str | None = None
        if voice_selection.startswith(config.LIBRARY_VOICE_PREFIX):
            voice_name = voice_selection[len(config.LIBRARY_VOICE_PREFIX) :]

        if not voice_name:
            return None

        voice_config = self._voice_library.load_voice_config(voice_name)
        voice_path = self._voice_library.get_voice_path(
            f"{config.LIBRARY_VOICE_PREFIX}{voice_name}"
        )

        if not voice_path or not os.path.exists(voice_path):
            return None

        try:
            system_content = self._prepare_system_message(scene_description)
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content="Please speak this text."),
                Message(role="assistant", content=AudioContent(audio_url=voice_path)),
                Message(role="user", content=transcript),
            ]

            min_p_value = voice_config["min_p"] if voice_config["min_p"] > 0 else None
            output = self._generate_with_cache(
                messages,
                voice_config["max_new_tokens"],
                voice_config["temperature"],
                voice_config["top_k"],
                voice_config["top_p"],
                min_p_value,
                voice_config["repetition_penalty"],
                voice_config["ras_win_len"],
                voice_config["ras_win_max_num_repeat"],
                voice_config["do_sample"],
                use_cache=True,
            )
            return output
        except Exception as exc:
            print(f"Error applying voice config: {exc}")
            return None

    def test_voice_sample(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        test_text: str = config.DEFAULT_TEST_VOICE_PROMPT,
    ) -> tuple[str | None, str]:
        """Test a voice sample by generating audio with provided text.
        
        Creates temporary audio/transcript files, generates audio using the voice,
        and returns the test output path and status message.
        """
        if audio_data is None:
            return None, "❌ Please upload an audio sample first"

        if not test_text.strip():
            test_text = config.DEFAULT_TEST_VOICE_PROMPT

        try:
            self._ensure_engine()
            temp_audio_path = audio_io.save_temp_audio_robust(audio_data, sample_rate)
            temp_txt_path = self._voice_library.create_voice_reference_txt(temp_audio_path)

            system_content = config.DEFAULT_SYSTEM_MESSAGE
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content="Please speak this text."),
                Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)),
                Message(role="user", content=test_text),
            ]

            output = self._generate_with_cache(
                messages,
                config.DEFAULT_MAX_NEW_TOKENS,
                config.DEFAULT_TEMPERATURE,
                use_cache=False,
            )

            test_output_path = config.VOICE_LIBRARY_TEST_OUTPUT_FILENAME
            torchaudio.save(
                test_output_path,
                torch.from_numpy(output.audio)[None, :],
                output.sampling_rate,
            )

            audio_io.robust_file_cleanup([temp_audio_path, temp_txt_path])

            return (
                test_output_path,
                "✅ Voice test completed! Listen to the result above.",
            )
        except Exception as exc:
            return None, f"❌ Error testing voice: {str(exc)}"

    # ------------------------------------------------------------------
    # Generation workflows
    # ------------------------------------------------------------------
    def generate_basic(
        self,
        transcript: str,
        voice_prompt: str | None,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        scene_description: str,
        top_k: int = config.DEFAULT_TOP_K,
        top_p: float = config.DEFAULT_TOP_P,
        min_p: float = config.DEFAULT_MIN_P,
        repetition_penalty: float = config.DEFAULT_REPETITION_PENALTY,
        ras_win_len: int = config.DEFAULT_RAS_WIN_LEN,
        ras_win_max_num_repeat: int = config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
        do_sample: bool = config.DEFAULT_DO_SAMPLE,
        enable_normalization: bool = False,
        target_volume: float = config.DEFAULT_TARGET_VOLUME,
    ) -> str | None:
        self._ensure_engine()
        self._seed_if_needed(seed)

        system_content = self._prepare_system_message(scene_description)
        messages: list[Message]

        if voice_prompt and voice_prompt != config.SMART_VOICE_LABEL:
            ref_audio_path = self._voice_library.get_voice_path(voice_prompt)
            if ref_audio_path and os.path.exists(ref_audio_path):
                txt_path = self._voice_library.robust_txt_path_creation(ref_audio_path)
                if not os.path.exists(txt_path):
                    self._voice_library.create_voice_reference_txt(ref_audio_path)

                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content="Please speak this text."),
                    Message(
                        role="assistant",
                        content=AudioContent(audio_url=ref_audio_path),
                    ),
                    Message(role="user", content=transcript),
                ]
            else:
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=transcript),
                ]
        else:
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content=transcript),
            ]

        min_p_value = min_p if min_p > 0 else None
        output = self._generate_with_cache(
            messages,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            min_p_value,
            repetition_penalty,
            ras_win_len,
            ras_win_max_num_repeat,
            do_sample,
        )

        output_path = audio_io.get_output_path(config.OUTPUT_BASIC_SUBDIR, "basic_audio")
        torchaudio.save(
            output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate
        )

        if enable_normalization:
            print("🔊 Applying volume normalization to basic generation...")
            waveform, sample_rate = torchaudio.load(output_path)
            buffer = AudioBuffer.from_tensor(waveform, sample_rate)
            normalized_audio = normalize_audio_volume(
                buffer, target_rms=target_volume
            )
            normalized_path = audio_io.get_output_path(
                config.OUTPUT_BASIC_SUBDIR, "normalized_basic_audio"
            )

            torchaudio.save(normalized_path, normalized_audio.to_tensor(), sample_rate)

            self.clear_caches()
            return normalized_path

        self.clear_caches()
        return output_path

    def generate_voice_clone(
        self,
        transcript: str,
        uploaded_voice: tuple[int, np.ndarray],
        temperature: float,
        max_new_tokens: int,
        seed: int,
        top_k: int = config.DEFAULT_TOP_K,
        top_p: float = config.DEFAULT_TOP_P,
        min_p: float = config.DEFAULT_MIN_P,
        repetition_penalty: float = config.DEFAULT_REPETITION_PENALTY,
        ras_win_len: int = config.DEFAULT_RAS_WIN_LEN,
        ras_win_max_num_repeat: int = config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
        do_sample: bool = config.DEFAULT_DO_SAMPLE,
    ) -> str:
        self._ensure_engine()

        if not transcript.strip():
            raise ValueError("Please enter text to synthesize")

        if uploaded_voice is None or uploaded_voice[1] is None:
            raise ValueError("Please upload a voice sample for cloning")

        self._seed_if_needed(seed)

        temp_audio_path: str | None = None
        temp_txt_path: str | None = None

        try:
            temp_audio_path = audio_io.save_temp_audio_robust(
                uploaded_voice[1], uploaded_voice[0]
            )
            temp_txt_path = self._voice_library.create_voice_reference_txt(
                temp_audio_path
            )

            system_content = config.DEFAULT_SYSTEM_MESSAGE
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content="Please speak this text."),
                Message(
                    role="assistant", content=AudioContent(audio_url=temp_audio_path)
                ),
                Message(role="user", content=transcript),
            ]

            min_p_value = min_p if min_p > 0 else None
            output = self._generate_with_cache(
                messages,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                min_p_value,
                repetition_penalty,
                ras_win_len,
                ras_win_max_num_repeat,
                do_sample,
                use_cache=False,
            )

            output_path = audio_io.get_output_path(
                config.OUTPUT_VOICE_CLONING_SUBDIR, "cloned_voice"
            )
            torchaudio.save(
                output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate
            )
            self.clear_caches()
            return output_path
        finally:
            audio_io.robust_file_cleanup([temp_audio_path, temp_txt_path])

    def generate_voice_clone_alternative(
        self,
        transcript: str,
        uploaded_voice: tuple[int, np.ndarray],
        temperature: float,
        max_new_tokens: int,
        seed: int,
        top_k: int = config.DEFAULT_TOP_K,
        top_p: float = config.DEFAULT_TOP_P,
        min_p: float = config.DEFAULT_MIN_P,
        repetition_penalty: float = config.DEFAULT_REPETITION_PENALTY,
        ras_win_len: int = config.DEFAULT_RAS_WIN_LEN,
        ras_win_max_num_repeat: int = config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
        do_sample: bool = config.DEFAULT_DO_SAMPLE,
    ) -> str:
        """Alternative voice cloning using <|voice_ref_start|> tag format."""
        self._ensure_engine()

        if not transcript.strip():
            raise ValueError("Please enter text to synthesize")

        if uploaded_voice is None or uploaded_voice[1] is None:
            raise ValueError("Please upload a voice sample for cloning")

        self._seed_if_needed(seed)

        temp_audio_path: str | None = None
        try:
            temp_audio_path = audio_io.enhanced_save_temp_audio_fixed(uploaded_voice)

            system_content = config.DEFAULT_SYSTEM_MESSAGE
            user_content = (
                "<|voice_ref_start|>\n"
                f"{temp_audio_path}\n"
                "<|voice_ref_end|>\n\n"
                f"{transcript}"
            )

            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content=user_content),
            ]

            min_p_value = min_p if min_p > 0 else None
            output = self._generate_with_cache(
                messages,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                min_p_value,
                repetition_penalty,
                ras_win_len,
                ras_win_max_num_repeat,
                do_sample,
                use_cache=False,
            )

            output_path = audio_io.get_output_path(
                config.OUTPUT_VOICE_CLONING_SUBDIR, "cloned_voice_alt"
            )
            torchaudio.save(
                output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate
            )
            self.clear_caches()
            return output_path
        finally:
            audio_io.robust_file_cleanup(temp_audio_path)

    def generate_longform(
        self,
        transcript: str,
        voice_choice: str,
        uploaded_voice: tuple[int, np.ndarray] | None,
        voice_prompt: str | None,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        scene_description: str,
        chunk_size: int,
    ) -> str | None:
        """Generate long-form audio by chunking text and concatenating segments.
        
        Supports three voice modes:
        - Upload Voice: uses provided uploaded_voice for all chunks
        - Predefined Voice: uses library voice from voice_prompt for all chunks  
        - Smart Voice: generates first chunk without reference, then uses it for consistency
        """
        self._ensure_engine()
        self._seed_if_needed(seed)

        chunks = self._smart_chunk_text(transcript, max_chunk_size=chunk_size)

        temp_audio_path: str | None = None
        temp_txt_path: str | None = None
        voice_ref_path: str | None = None
        voice_ref_text: str | None = None
        first_chunk_audio_path: str | None = None
        first_chunk_text: str | None = None

        try:
            if (
                voice_choice == "Upload Voice"
                and uploaded_voice is not None
                and uploaded_voice[1] is not None
            ):
                temp_audio_path = audio_io.enhanced_save_temp_audio_fixed(
                    uploaded_voice
                )
                temp_txt_path = self._voice_library.create_voice_reference_txt(
                    temp_audio_path
                )
                voice_ref_path = temp_audio_path
                if temp_txt_path and os.path.exists(temp_txt_path):
                    with open(temp_txt_path, encoding="utf-8") as handle:
                        voice_ref_text = handle.read().strip()
                else:
                    voice_ref_text = config.WHISPER_FALLBACK_TRANSCRIPTION
            elif (
                voice_choice == "Predefined Voice"
                and voice_prompt != config.SMART_VOICE_LABEL
            ):
                ref_audio_path = self._voice_library.get_voice_path(voice_prompt)
                if ref_audio_path and os.path.exists(ref_audio_path):
                    voice_ref_path = ref_audio_path
                    txt_path = self._voice_library.robust_txt_path_creation(
                        ref_audio_path
                    )
                    if not os.path.exists(txt_path):
                        self._voice_library.create_voice_reference_txt(ref_audio_path)

                    if os.path.exists(txt_path):
                        with open(txt_path, encoding="utf-8") as handle:
                            voice_ref_text = handle.read().strip()
                    else:
                        voice_ref_text = config.VOICE_SAMPLE_FALLBACK_TRANSCRIPT

            system_content = self._prepare_system_message(scene_description)

            full_audio: list[np.ndarray] = []
            sampling_rate = config.DEFAULT_SAMPLE_RATE

            for index, chunk in enumerate(chunks):
                print(f"Processing chunk {index + 1}/{len(chunks)}: {chunk[:50]}...")
                if (
                    voice_choice == "Upload Voice"
                    and voice_ref_path
                    and voice_ref_text
                ):
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=voice_ref_text),
                        Message(
                            role="assistant", content=AudioContent(audio_url=voice_ref_path)
                        ),
                        Message(role="user", content=chunk),
                    ]
                elif (
                    voice_choice == "Predefined Voice"
                    and voice_ref_path
                    and voice_ref_text
                ):
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=voice_ref_text),
                        Message(
                            role="assistant", content=AudioContent(audio_url=voice_ref_path)
                        ),
                        Message(role="user", content=chunk),
                    ]
                elif voice_choice == "Smart Voice":
                    if index == 0:
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=chunk),
                        ]
                    else:
                        if first_chunk_audio_path and first_chunk_text:
                            messages = [
                                Message(role="system", content=system_content),
                                Message(role="user", content=first_chunk_text),
                                Message(
                                    role="assistant",
                                    content=AudioContent(audio_url=first_chunk_audio_path),
                                ),
                                Message(role="user", content=chunk),
                            ]
                        else:
                            messages = [
                                Message(role="system", content=system_content),
                                Message(role="user", content=chunk),
                            ]
                else:
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=chunk),
                    ]

                output = self._generate_with_cache(
                    messages, max_new_tokens, temperature, use_cache=True
                )

                if voice_choice == "Smart Voice" and index == 0:
                    first_chunk_audio_path = f"first_chunk_audio_{seed}_{hash(transcript[:20])}.wav"
                    torchaudio.save(
                        first_chunk_audio_path,
                        torch.from_numpy(output.audio)[None, :],
                        output.sampling_rate,
                    )
                    first_chunk_text = chunk

                full_audio.append(output.audio)
                sampling_rate = output.sampling_rate

            if full_audio:
                full_audio_array = np.concatenate(full_audio, axis=0)
                output_path = audio_io.get_output_path(
                    config.OUTPUT_LONGFORM_SUBDIR, "longform_audio"
                )
                torchaudio.save(
                    output_path,
                    torch.from_numpy(full_audio_array)[None, :],
                    sampling_rate,
                )
                self.clear_caches()
                return output_path

            self.clear_caches()
            return None
        finally:
            audio_io.robust_file_cleanup(
                [temp_audio_path, temp_txt_path, first_chunk_audio_path]
            )

    def generate_multi_speaker(
        self,
        transcript: str,
        voice_method: str,
        uploaded_voices: list[tuple[int, np.ndarray] | None],
        predefined_voices: Sequence[str | None],
        temperature: float,
        max_new_tokens: int,
        seed: int,
        scene_description: str,
        auto_format: bool,
        speaker_pause_duration: float = config.DEFAULT_SPEAKER_PAUSE_SECONDS,
    ) -> str:
        """Generate multi-speaker audio from transcript with [SPEAKER0], [SPEAKER1] tags.
        
        Voice methods:
        - Upload Voices: uses uploaded_voices list indexed by speaker number
        - Predefined Voices: uses predefined_voices list from voice library
        - Smart Voice: generates first line per speaker, reuses for consistency
        
        Optionally inserts pauses between different speakers based on speaker_pause_duration.
        """
        self._ensure_engine()
        self._seed_if_needed(seed)

        if auto_format:
            transcript = self._auto_format_multi_speaker(transcript)

        speakers = self._parse_multi_speaker_text(transcript)
        if not speakers:
            raise ValueError(
                "No speakers found in transcript. Use [SPEAKER0], [SPEAKER1] format or enable auto-format."
            )

        print(f"🎭 Found speakers: {list(speakers.keys())}")

        voice_refs: dict[str, str] = {}
        temp_files: list[str] = []
        speaker_first_refs: dict[str, tuple[str, str]] = {}
        uploaded_voice_refs: dict[str, tuple[str, str]] = {}
        speaker_audio_refs: dict[str, str] = {}

        try:
            if voice_method == "Upload Voices":
                for index, audio in enumerate(uploaded_voices or []):
                    if audio is not None and audio[1] is not None:
                        speaker_key = f"SPEAKER{index}"
                        print(f"🎤 Processing uploaded voice for {speaker_key}...")
                        temp_path = audio_io.enhanced_save_temp_audio_fixed(audio)
                        temp_txt_path = self._voice_library.create_voice_reference_txt(
                            temp_path
                        )
                        if os.path.exists(temp_txt_path):
                            with open(temp_txt_path, encoding="utf-8") as handle:
                                transcription = handle.read().strip()
                        else:
                            transcription = config.WHISPER_FALLBACK_TRANSCRIPTION
                        uploaded_voice_refs[speaker_key] = (temp_path, transcription)
                        temp_files.extend([temp_path, temp_txt_path])
                        print(
                            f"✅ Setup voice reference for {speaker_key}: {temp_path}"
                        )
                        print(
                            f"📝 {speaker_key} transcription: '{transcription[:50]}...'"
                        )
                print(
                    f"🎭 Upload Voices setup complete. Voice refs: {list(uploaded_voice_refs.keys())}"
                )
            elif voice_method == "Predefined Voices":
                for index, voice_name in enumerate(predefined_voices or []):
                    if voice_name and voice_name != config.SMART_VOICE_LABEL:
                        speaker_key = f"SPEAKER{index}"
                        ref_audio_path = self._voice_library.get_voice_path(voice_name)
                        if ref_audio_path and os.path.exists(ref_audio_path):
                            voice_refs[speaker_key] = ref_audio_path
                            print(
                                f"📁 Setup voice reference for {speaker_key}: {ref_audio_path}"
                            )
                            txt_path = self._voice_library.robust_txt_path_creation(
                                ref_audio_path
                            )
                            if not os.path.exists(txt_path):
                                self._voice_library.create_voice_reference_txt(
                                    ref_audio_path
                                )
                            if os.path.exists(txt_path):
                                with open(
                                    txt_path, encoding="utf-8"
                                ) as handle:
                                    text_content = handle.read().strip()
                            else:
                                text_content = config.VOICE_SAMPLE_FALLBACK_TRANSCRIPT
                            speaker_audio_refs[speaker_key] = ref_audio_path
                            speaker_first_refs[speaker_key] = (
                                ref_audio_path,
                                text_content,
                            )

            system_content = self._prepare_system_message(scene_description)
            full_audio: list[np.ndarray] = []
            sampling_rate = config.DEFAULT_SAMPLE_RATE
            lines = transcript.splitlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                speaker_match = re.match(r"\[SPEAKER(\d+)\]\s*(.+)", line)
                if not speaker_match:
                    continue

                speaker_id = f"SPEAKER{speaker_match.group(1)}"
                text_content = speaker_match.group(2).strip()
                if not text_content:
                    continue

                print(f"🎭 Generating for {speaker_id}: {text_content[:50]}...")

                if (
                    voice_method == "Upload Voices"
                    and speaker_id in uploaded_voice_refs
                ):
                    ref_audio_path, ref_text = uploaded_voice_refs[speaker_id]
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=ref_text),
                        Message(
                            role="assistant", content=AudioContent(audio_url=ref_audio_path)
                        ),
                        Message(role="user", content=text_content),
                    ]
                elif voice_method == "Predefined Voices" and speaker_id in voice_refs:
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content="Please speak this text."),
                        Message(
                            role="assistant",
                            content=AudioContent(audio_url=voice_refs[speaker_id]),
                        ),
                        Message(role="user", content=text_content),
                    ]
                elif voice_method == "Smart Voice":
                    if speaker_id in speaker_first_refs:
                        first_audio_path, first_text = speaker_first_refs[speaker_id]
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=first_text),
                            Message(
                                role="assistant",
                                content=AudioContent(audio_url=first_audio_path),
                            ),
                            Message(role="user", content=text_content),
                        ]
                    else:
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=text_content),
                        ]
                else:
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=text_content),
                    ]

                print(f"📝 Generating audio for: '{text_content}'")
                output = self._generate_with_cache(
                    messages, max_new_tokens, temperature, use_cache=False
                )

                if voice_method == "Smart Voice" and speaker_id not in speaker_first_refs:
                    speaker_audio_path = (
                        f"temp_speaker_{speaker_id}_{seed}_{int(time.time())}.wav"
                    )
                    torchaudio.save(
                        speaker_audio_path,
                        torch.from_numpy(output.audio)[None, :],
                        output.sampling_rate,
                    )
                    text_path = (
                        f"temp_speaker_{speaker_id}_{seed}_{int(time.time())}.txt"
                    )
                    try:
                        transcribed_text = transcribe_audio(speaker_audio_path)
                    except Exception:
                        transcribed_text = text_content
                    with open(text_path, "w", encoding="utf-8") as handle:
                        handle.write(transcribed_text)
                    speaker_first_refs[speaker_id] = (
                        speaker_audio_path,
                        text_content,
                    )
                    temp_files.extend([speaker_audio_path, text_path])

                if output.audio is not None and len(output.audio) > 0:
                    full_audio.append(output.audio)
                    sampling_rate = output.sampling_rate
                    print(
                        f"✅ Added audio segment (length: {len(output.audio)} samples)"
                    )
                else:
                    print(
                        f"⚠️ Empty or invalid audio output for: '{text_content}'"
                    )

                if len(full_audio) > 1 and speaker_pause_duration > 0:
                    # Add pause between different speakers
                    prev_line_index = lines.index(line) - 1
                    if prev_line_index >= 0:
                        prev_line = lines[prev_line_index].strip()
                        prev_match = re.match(r"\[SPEAKER(\d+)\]", prev_line or "")
                        if prev_match and prev_match.group(1) != speaker_match.group(1):
                            pause_samples = int(speaker_pause_duration * sampling_rate)
                            pause_audio = np.zeros(pause_samples, dtype=np.float32)
                            full_audio.append(pause_audio)
                            print(
                                f"🔇 Added {speaker_pause_duration}s pause between speakers"
                            )

            if full_audio:
                full_audio_array = np.concatenate(full_audio, axis=0)
                output_path = audio_io.get_output_path(
                    config.OUTPUT_MULTI_SPEAKER_SUBDIR, "multi_speaker_audio"
                )
                torchaudio.save(
                    output_path,
                    torch.from_numpy(full_audio_array)[None, :],
                    sampling_rate,
                )
                print(
                    f"🎉 Multi-speaker audio generated successfully: {output_path}"
                )
                self.clear_caches()
                return output_path

            self.clear_caches()
            raise ValueError("No audio was generated. Check your transcript format and voice samples.")
        except Exception as exc:
            print(f"❌ Error in multi-speaker generation: {exc}")
            raise
        finally:
            audio_io.robust_file_cleanup(temp_files)

    def generate_dynamic_multi_speaker(
        self,
        text: str,
        voice_method: str,
        speaker_mapping: dict[str, str],
        temperature: float,
        max_new_tokens: int,
        seed: int,
        scene_description: str,
        enable_normalization: bool,
        normalization_method: str,
        target_volume: float,
        speaker_pause_duration: float,
        *voice_components,
    ) -> str | None:
        """Generate multi-speaker audio with dynamic speaker name mapping.
        
        Converts custom speaker names (e.g., [Alice], [Bob]) to SPEAKER0, SPEAKER1
        format using speaker_mapping, then delegates to generate_multi_speaker.
        
        Voice components are extracted based on voice_method:
        - Smart Voice: no voice assets needed
        - Upload Voices: first 10 components are uploaded audio tuples
        - Voice Library: components 10-20 (or last 10) are library voice names
        
        Optionally applies volume normalization after generation.
        """
        if not text or not speaker_mapping:
            return None

        try:
            self._ensure_engine()
            self._seed_if_needed(seed)

            converted_text = self._convert_to_speaker_format(text, speaker_mapping)
            print(f"🎭 Converted text: {converted_text[:200]}...")

            output_audio_path: str | None = None

            if voice_method == "Smart Voice":
                output_audio_path = self.generate_multi_speaker(
                    converted_text,
                    "Smart Voice",
                    [],
                    [],
                    temperature,
                    max_new_tokens,
                    seed,
                    scene_description,
                    False,
                    speaker_pause_duration,
                )
            elif voice_method == "Upload Voices":
                uploaded_voices: list[tuple[int, np.ndarray] | None] = []
                num_speakers = len(speaker_mapping)
                for index in range(min(num_speakers, 10)):
                    if index < len(voice_components) and voice_components[index] is not None:
                        uploaded_voices.append(voice_components[index])
                    else:
                        uploaded_voices.append(None)

                print(f"🎭 Using uploaded voices for {num_speakers} speakers")
                output_audio_path = self.generate_multi_speaker(
                    converted_text,
                    "Upload Voices",
                    uploaded_voices,
                    [],
                    temperature,
                    max_new_tokens,
                    seed,
                    scene_description,
                    False,
                    speaker_pause_duration,
                )
            elif voice_method == "Voice Library":
                # Extract library voice selections from voice_components
                if len(voice_components) >= 20:
                    library_selections = voice_components[10:20]
                else:
                    library_selections = voice_components[-10:]

                predefined_voices: list[str | None] = []
                num_speakers = len(speaker_mapping)
                for index in range(max(3, num_speakers)):
                    if (
                        index < len(library_selections)
                        and library_selections[index]
                        and library_selections[index] != config.SMART_VOICE_LABEL
                    ):
                        predefined_voices.append(library_selections[index])
                    else:
                        predefined_voices.append(config.SMART_VOICE_LABEL)

                print(f"🎭 Using library voices: {predefined_voices[:num_speakers]}")
                output_audio_path = self.generate_multi_speaker(
                    converted_text,
                    "Predefined Voices",
                    [],
                    predefined_voices,
                    temperature,
                    max_new_tokens,
                    seed,
                    scene_description,
                    False,
                    speaker_pause_duration,
                )

            if output_audio_path and enable_normalization:
                print(f"🔊 Applying {normalization_method} volume normalization...")
                waveform, sample_rate = torchaudio.load(output_audio_path)
                buffer = AudioBuffer.from_tensor(waveform, sample_rate)
                normalized_audio = enhance_multi_speaker_audio(
                    buffer,
                    normalization_method=normalization_method,
                    target_rms=target_volume,
                )
                normalized_path = audio_io.get_output_path(
                    config.OUTPUT_MULTI_SPEAKER_SUBDIR,
                    "normalized_multi_speaker_audio",
                )
                torchaudio.save(
                    normalized_path,
                    normalized_audio.to_tensor(),
                    sample_rate,
                )

                print(f"🎵 Saved normalized audio to: {normalized_path}")
                return normalized_path

            return output_audio_path
        except Exception as exc:
            print(f"❌ Error generating dynamic multi-speaker audio: {exc}")
            import traceback

            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Text processing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_format_multi_speaker(text: str) -> str:
        """Convert free-form dialogue to [SPEAKER0], [SPEAKER1] format.
        
        Assumes alternating speakers when detecting quoted text or lines with colons.
        Returns text unchanged if already formatted with [SPEAKER tags.
        """
        if "[SPEAKER" in text:
            return text

        lines = text.split("\n")
        formatted_lines: list[str] = []
        current_speaker = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('"') or line.startswith("'") or ":" in line:
                if formatted_lines:
                    current_speaker = 1 - current_speaker
                formatted_lines.append(f"[SPEAKER{current_speaker}] {line}")
            else:
                formatted_lines.append(f"[SPEAKER{current_speaker}] {line}")

        return "\n".join(formatted_lines)

    @staticmethod
    def _parse_multi_speaker_text(text: str) -> dict[str, list[str]]:
        """Extract speaker IDs and their text segments from formatted transcript.
        
        Returns dict mapping SPEAKER0, SPEAKER1, etc. to lists of their text content.
        """
        speaker_pattern = r"\[SPEAKER(\d+)\]\s*([^[]*?)(?=\[SPEAKER\d+\]|$)"
        matches = re.findall(speaker_pattern, text, re.DOTALL)

        speakers: dict[str, list[str]] = {}
        for speaker_id, content in matches:
            speaker_key = f"SPEAKER{speaker_id}"
            speakers.setdefault(speaker_key, []).append(content.strip())
        return speakers

    @staticmethod
    def _convert_to_speaker_format(
        text: str, speaker_mapping: dict[str, str]
    ) -> str:
        """Convert custom speaker names to [SPEAKER0], [SPEAKER1] format.
        
        Uses speaker_mapping to replace [CustomName] tags with [SPEAKER#] tags.
        """
        if not text or not speaker_mapping:
            return text

        converted_text = text
        for speaker_name, speaker_id in speaker_mapping.items():
            pattern = rf"\[{re.escape(speaker_name)}\]\s*[:.]?\s*"
            replacement = f"[{speaker_id}] "
            converted_text = re.sub(
                pattern, replacement, converted_text, flags=re.MULTILINE
            )
        return converted_text

    @staticmethod
    def _smart_chunk_text(text: str, max_chunk_size: int = 200) -> list[str]:
        """Split text into chunks respecting paragraph and sentence boundaries.
        
        Prioritizes:
        1. Paragraph boundaries (double newlines)
        2. Sentence boundaries (. ! ?)
        3. Word boundaries if needed to fit max_chunk_size
        """
        paragraphs = text.split("\n\n")
        chunks: list[str] = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) <= max_chunk_size:
                chunks.append(paragraph)
                continue

            sentences = []
            sentence_parts = re.split(r"([.!?]+)", paragraph)

            for index in range(0, len(sentence_parts), 2):
                current_sentence = sentence_parts[index].strip()
                if index + 1 < len(sentence_parts):
                    current_sentence += sentence_parts[index + 1]
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())

            current_chunk = ""
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk = f"{current_chunk} {sentence}".strip()

            if current_chunk:
                chunks.append(current_chunk.strip())

        return chunks


def create_generation_service(
    device: torch.device,
    voice_library_service: VoiceLibrary,
) -> GenerationService:
    """Factory for the default GenerationService."""
    return GenerationService(device=device, voice_library_service=voice_library_service)
