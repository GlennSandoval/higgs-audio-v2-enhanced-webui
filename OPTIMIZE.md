# Voice Generation Optimization Checklist

## 1. Metal Acceleration
Enable Metal acceleration by installing the latest PyTorch (nightly if needed), exporting `PYTORCH_ENABLE_MPS_FALLBACK=1`, and verifying `higgs_audio_gradio.py` runs on the local GPU/ANE with manageable batch sizes.

## 2. Cache Reuse Discipline
Maximize cache reuse by keeping `optimized_generate_audio` on `use_cache=True`, avoiding unnecessary tweaks to `temperature`, `top_k`, and `top_p`, and reserving `clear_caches()` for major configuration changes.

## 3. Voice Asset Preload
Preload voice assets by loading configs with `app.services.voice_service.load_voice_config` during startup and staging reusable `AudioSegment` objects to skip redundant Whisper runs.

## 4. Input Preprocessing
Streamline preprocessing by enforcing 24 kHz mono inputs through `save_temp_audio_robust` or `enhanced_save_temp_audio_fixed`, and validating Whisper availability using `uv run python verify_setup.py`.

## 5. Decoding Balance
Balance decoding parameters by preferring modest increases to `top_k`/`top_p` instead of lowering `temperature` to maintain richness while reducing decoding iterations, and by adjusting `speaker_pause_duration` for dialogue pacing.

## 6. UI Parallelization
Parallelize UI workloads by keeping Gradio callbacks non-blocking, letting background threads handle normalization (`normalize_audio_volume`) and cache writes, and avoiding large blocking numpy copies in the main UI thread.

## Implementation Checklist
- [ ] 1. Metal Acceleration
- [ ] 2. Cache Reuse Discipline
- [ ] 3. Voice Asset Preload
- [ ] 4. Input Preprocessing
- [ ] 5. Decoding Balance
- [ ] 6. UI Parallelization
