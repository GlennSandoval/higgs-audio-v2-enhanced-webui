"""Central configuration and default values for the Higgs Audio Gradio app."""

import os

HF_HUB_ENABLE_HF_TRANSFER = "1"
PYTORCH_ENABLE_MPS_FALLBACK = "1"

MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_ID = "bosonai/higgs-audio-v2-tokenizer"

DEFAULT_SAMPLE_RATE = 24_000

OUTPUT_BASE_DIR = "output"
OUTPUT_BASIC_SUBDIR = "basic_generation"
OUTPUT_VOICE_CLONING_SUBDIR = "voice_cloning"
OUTPUT_LONGFORM_SUBDIR = "longform_generation"
OUTPUT_MULTI_SPEAKER_SUBDIR = "multi_speaker"
VOICE_LIBRARY_DIR = "voice_library"

OUTPUT_DIRECTORIES = (
    os.path.join(OUTPUT_BASE_DIR, OUTPUT_BASIC_SUBDIR),
    os.path.join(OUTPUT_BASE_DIR, OUTPUT_VOICE_CLONING_SUBDIR),
    os.path.join(OUTPUT_BASE_DIR, OUTPUT_LONGFORM_SUBDIR),
    os.path.join(OUTPUT_BASE_DIR, OUTPUT_MULTI_SPEAKER_SUBDIR),
    VOICE_LIBRARY_DIR,
)

VOICE_LIBRARY_AUDIO_EXTENSION = ".wav"
VOICE_LIBRARY_TRANSCRIPT_EXTENSION = ".txt"
VOICE_LIBRARY_CONFIG_SUFFIX = "_config.json"
VOICE_LIBRARY_TEST_OUTPUT_FILENAME = "voice_test_output.wav"

VOICE_PROMPTS_DIR = "examples/voice_prompts"

SMART_VOICE_LABEL = "None (Smart Voice)"
PREDEFINED_VOICE_PREFIX = "📁 "
LIBRARY_VOICE_PREFIX = "👤 "

DEFAULT_SYSTEM_MESSAGE = "Generate audio following instruction."
SCENE_DESC_START_TAG = "<|scene_desc_start|>"
SCENE_DESC_END_TAG = "<|scene_desc_end|>"

WHISPER_FALLBACK_TRANSCRIPTION = "This is a voice sample for cloning."
VOICE_SAMPLE_FALLBACK_TRANSCRIPT = "This is a voice sample."

DEFAULT_TEST_VOICE_PROMPT = "Hello, this is a test of my voice. How does it sound?"

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_NEW_TOKENS = 1_024
DEFAULT_SEED = 12_345
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_RAS_WIN_LEN = 7
DEFAULT_RAS_WIN_MAX_NUM_REPEAT = 2
DEFAULT_DO_SAMPLE = True
DEFAULT_TARGET_VOLUME = 0.15
DEFAULT_SPEAKER_PAUSE_SECONDS = 0.3

DEFAULT_CACHE_MAX_ENTRIES = 50

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7_860

STOP_STRINGS = ("<|end_of_text|>", "<|eot_id|>")

DEFAULT_VOICE_CONFIG = {
    "temperature": DEFAULT_TEMPERATURE,
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "seed": DEFAULT_SEED,
    "top_k": DEFAULT_TOP_K,
    "top_p": DEFAULT_TOP_P,
    "min_p": DEFAULT_MIN_P,
    "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    "ras_win_len": DEFAULT_RAS_WIN_LEN,
    "ras_win_max_num_repeat": DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
    "do_sample": DEFAULT_DO_SAMPLE,
    "description": "",
    "tags": [],
}
