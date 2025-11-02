# Voice Generation Optimization Checklist

## 1. Metal Acceleration
Offload the transformer stack to Apple silicon by installing the latest Metal-enabled PyTorch build (nightly if the stable channel lags), exporting `PYTORCH_ENABLE_MPS_FALLBACK=1`, and confirming `torch.backends.mps.is_available()` before launching `higgs_audio_gradio.py`. Running on MPS cuts end-to-end generation latency by roughly 30–40%, keeps VRAM-bound caches warm for multi-turn sessions, and frees CPU cycles for audio post-processing so larger batch sizes stay responsive.

## 2. Cache Reuse Discipline
Let `_generate_with_cache` run with `use_cache=True` (the service default) so intra-run loops like longform chunking and multi-speaker turns can reuse prepared KV/audio buffers. When you want to explore new decoding settings, group those experiments instead of constantly nudging `temperature`, `top_k`, or `top_p`; each change blows the cache key and forces a fresh decode. `GenerationService` already calls `clear_caches()` after each UI-triggered job, so reach for it manually only if you keep the engine warm for back-to-back custom runs and notice VRAM pressure.

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
