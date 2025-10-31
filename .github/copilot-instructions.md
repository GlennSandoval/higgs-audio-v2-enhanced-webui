## Architecture
- `higgs_audio_gradio.py` is the only entrypoint; it wires the Gradio UI to prompt parsing, voice handling, audio enhancement, and the serving engine.
- Core model access stays behind `boson_multimodal/serve/serve_engine.py` (`HiggsAudioServeEngine`); instantiate this singleton instead of touching `HiggsAudioModel` directly.
- Chat prompts must be built with `ChatMLSample`, `Message`, and `AudioContent` (`boson_multimodal/data_types.py`); the system message defaults to `"Generate audio following instruction."` and scene text is wrapped in `<|scene_desc_start|>` tags.
- Generated outputs arrive as `HiggsAudioResponse`, carrying a 24 kHz numpy waveform plus text/audio token metadata for downstream tooling.

## Workflows
- Use `uv sync` to create `.venv` and install deps, then `uv run python verify_setup.py` to sanity-check imports.
- Launch the UI with `uv run python higgs_audio_gradio.py`; the script sets `HF_HUB_ENABLE_HF_TRANSFER=1` to speed Hugging Face downloads.
- Large model weights come from `bosonai/higgs-audio-v2-generation-3B-base`; expect multi-GB pulls and GPU memory >16 GB for smooth runs.
- Keep `ffmpeg` on PATH (`brew install ffmpeg` on macOS) so `pydub` and `torchaudio` can transcode uploads without crashes.

## Voice Assets & Library
- Voices live under `voice_library/` as triplets: `<name>.wav`, auto-generated `<name>.txt` transcript, and `<name>_config.json` with saved sampling params.
- Use `save_voice_to_library` / `create_voice_reference_txt` patterns when adding assets so Whisper transcription stays in sync with audio.
- Reference library voices via `get_voice_path` and reuse their configs with `load_voice_config` to preserve temperature/top_* settings.

## Generation Flow
- `optimized_generate_audio` handles caching and delegates to `HiggsAudioServeEngine.generate`; only `temperature`, `top_k`, `top_p`, and `ras_win_*` are honored today (`min_p` and `repetition_penalty` are TODOs).
- Multi-speaker prompts expect `[SPEAKER0]` style tags; `auto_format_multi_speaker` and `parse_multi_speaker_text` normalize free-form dialogue.
- Uploaded or predefined voices are injected by pairing a `Message` carrying `AudioContent(audio_url=...)` with the target text; Smart Voice mode auto-saves the first segment per speaker for later consistency.
- Scene descriptions are optional but must stay inside `<|scene_desc_start|>` / `<|scene_desc_end|>` so the tokenizer admits them.

## Audio Post-Processing
- Enable volume controls via `enhance_multi_speaker_audio` and `normalize_audio_volume` in `audio_processing_utils.py`; they expect float32 arrays and will downmix intelligently.
- Multi-speaker gap timing relies on `speaker_pause_duration` in `generate_multi_speaker`; adjust that instead of manual silence insertion.
- Call `clear_caches()` after long-form or batch runs to release `_audio_cache`, `_token_cache`, and CUDA memory.

## Environment & Dependencies
- Whisper integration prefers `faster-whisper`; the loader falls back to `openai-whisper` or dummy text, so guard new transcription logic behind `WHISPER_AVAILABLE`.
- Audio tokenization uses `bosonai/higgs-audio-v2-tokenizer` via `load_higgs_audio_tokenizer`, which pulls configs with `huggingface_hub.snapshot_download`—ensure the HF auth token exists if the repo is gated.
- Quantization and codec utilities live under `boson_multimodal/audio_processing/descriptaudiocodec`; reuse them rather than reimplementing encoders/decoders.

## Conventions & Gotchas
- Waveforms are always 24 kHz mono float32 when saved; use `save_temp_audio_robust` / `enhanced_save_temp_audio_fixed` to enforce this when persisting uploads.
- Never persist raw user prompts or generated transcripts—`save_transcript_if_enabled` and related functions are intentionally disabled.
- `optimized_generate_audio` cache keys include temperature/top_* params; bypass caching (`use_cache=False`) when experimenting with new generation flags.
- Avoid mixing direct numpy/tensor writes with `torchaudio.save`; stick to helpers so stereo/mono reshaping and normalization stays consistent.
