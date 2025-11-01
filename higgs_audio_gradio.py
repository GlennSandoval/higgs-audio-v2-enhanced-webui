import argparse
import os
import re
import sys

import gradio as gr
import torchaudio

from app import config, generation, startup, voice_library as voice_lib

startup.configure_environment()

device = startup.select_device()

startup.ensure_output_directories()

voice_library_service = voice_lib.create_default_voice_library()
generation_service = generation.create_generation_service(device, voice_library_service)
WHISPER_AVAILABLE = voice_lib.WHISPER_AVAILABLE

# Voice library management
def get_voice_library_voices():
    """Proxy for listing voices in the library."""
    return voice_library_service.list_voice_library_voices()


def get_voice_config_path(voice_name):
    """Get the config file path for a voice."""
    return voice_library_service.get_voice_config_path(voice_name)


def get_default_voice_config():
    """Return the default voice configuration."""
    return voice_library_service.get_default_voice_config()


def save_voice_config(voice_name, voice_config_data):
    """Persist configuration data for a voice."""
    return voice_library_service.save_voice_config(voice_name, voice_config_data)


def load_voice_config(voice_name):
    """Load configuration data for a voice."""
    return voice_library_service.load_voice_config(voice_name)


def save_voice_to_library(audio_data, sample_rate, voice_name):
    """Save a new voice to the library."""
    return voice_library_service.save_voice(audio_data, sample_rate, voice_name)


def delete_voice_from_library(voice_name):
    """Delete a voice from the library."""
    return voice_library_service.delete_voice(voice_name)


def get_all_available_voices():
    """Return the full list of selectable voices."""
    return voice_library_service.list_all_available_voices()


def get_voice_path(voice_selection):
    """Resolve a voice selection to a filesystem path."""
    return voice_library_service.get_voice_path(voice_selection)

def apply_voice_config_to_generation(
    voice_selection, transcript, scene_description="", force_audio_gen=False
):
    """Delegate voice-configured generation to the generation service."""
    return generation_service.apply_voice_config_to_generation(
        voice_selection,
        transcript,
        scene_description=scene_description,
        force_audio_gen=force_audio_gen,
    )

# Available voice prompts - this needs to be refreshed dynamically
def get_current_available_voices():
    """Get current available voices (refreshed each time)"""
    return get_all_available_voices()

available_voices = get_current_available_voices()




def detect_dynamic_speakers(text):
    """Detect any speaker names in brackets and return list of unique speakers"""
    if not text or not text.strip():
        return []
    
    # Pattern to match any text in brackets at the start of lines
    # Supports formats like [Alice], [Bob], [Character Name], etc.
    speaker_pattern = r'^\s*\[([^\]]+)\]\s*[:.]?\s*(.+?)(?=^\s*\[[^\]]+\]|$)'
    
    # Find all matches across multiple lines
    matches = re.findall(speaker_pattern, text, re.MULTILINE | re.DOTALL)
    
    # Extract unique speaker names
    speakers = []
    seen_speakers = set()
    
    for speaker_name, content in matches:
        speaker_name = speaker_name.strip()
        if speaker_name and speaker_name not in seen_speakers:
            speakers.append(speaker_name)
            seen_speakers.add(speaker_name)
    
    return speakers

# VOICE LIBRARY FUNCTIONS

def generate_basic(
    transcript,
    voice_prompt,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    top_k=config.DEFAULT_TOP_K,
    top_p=config.DEFAULT_TOP_P,
    min_p=config.DEFAULT_MIN_P,
    repetition_penalty=config.DEFAULT_REPETITION_PENALTY,
    ras_win_len=config.DEFAULT_RAS_WIN_LEN,
    ras_win_max_num_repeat=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
    do_sample=config.DEFAULT_DO_SAMPLE,
    enable_normalization=False,
    target_volume=config.DEFAULT_TARGET_VOLUME
):
    return generation_service.generate_basic(
        transcript,
        voice_prompt,
        temperature,
        max_new_tokens,
        seed,
        scene_description,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        ras_win_len,
        ras_win_max_num_repeat,
        do_sample,
        enable_normalization,
        target_volume,
    )

def generate_voice_clone(
    transcript,
    uploaded_voice,
    temperature,
    max_new_tokens,
    seed,
    top_k=config.DEFAULT_TOP_K,
    top_p=config.DEFAULT_TOP_P,
    min_p=config.DEFAULT_MIN_P,
    repetition_penalty=config.DEFAULT_REPETITION_PENALTY,
    ras_win_len=config.DEFAULT_RAS_WIN_LEN,
    ras_win_max_num_repeat=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
    do_sample=config.DEFAULT_DO_SAMPLE
):
    return generation_service.generate_voice_clone(
        transcript,
        uploaded_voice,
        temperature,
        max_new_tokens,
        seed,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        ras_win_len,
        ras_win_max_num_repeat,
        do_sample,
    )

def generate_voice_clone_alternative(
    transcript,
    uploaded_voice,
    temperature,
    max_new_tokens,
    seed,
    top_k=config.DEFAULT_TOP_K,
    top_p=config.DEFAULT_TOP_P,
    min_p=config.DEFAULT_MIN_P,
    repetition_penalty=config.DEFAULT_REPETITION_PENALTY,
    ras_win_len=config.DEFAULT_RAS_WIN_LEN,
    ras_win_max_num_repeat=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
    do_sample=config.DEFAULT_DO_SAMPLE
):
    """Alternative voice cloning method using voice_ref format"""
    return generation_service.generate_voice_clone_alternative(
        transcript,
        uploaded_voice,
        temperature,
        max_new_tokens,
        seed,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        ras_win_len,
        ras_win_max_num_repeat,
        do_sample,
    )

def generate_longform(
    transcript,
    voice_choice,
    uploaded_voice,
    voice_prompt,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    chunk_size
):
    return generation_service.generate_longform(
        transcript,
        voice_choice,
        uploaded_voice,
        voice_prompt,
        temperature,
        max_new_tokens,
        seed,
        scene_description,
        chunk_size,
    )

# IMPROVED HANDLER FUNCTION FOR MULTI-SPEAKER
def generate_multi_speaker(
    transcript,
    voice_method,
    uploaded_voices,
    predefined_voices,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    auto_format,
    speaker_pause_duration=config.DEFAULT_SPEAKER_PAUSE_SECONDS
):
    return generation_service.generate_multi_speaker(
        transcript,
        voice_method,
        uploaded_voices,
        predefined_voices,
        temperature,
        max_new_tokens,
        seed,
        scene_description,
        auto_format,
        speaker_pause_duration,
    )

def refresh_voice_list():
    updated_voices = get_all_available_voices()
    return gr.update(choices=updated_voices)

def refresh_voice_list_multi():
    """Refresh voice list for multi-speaker (returns 3 updates)"""
    updated_voices = get_all_available_voices()
    return [gr.update(choices=updated_voices), gr.update(choices=updated_voices), gr.update(choices=updated_voices)]

def refresh_library_list():
    library_voices = ["None"] + get_voice_library_voices()
    return gr.update(choices=library_voices)

# Check audio processing capabilities at startup
startup.check_audio_dependencies()

# Gradio interface
with gr.Blocks(title="Higgs Audio v2 Generator") as demo:
    gr.HTML('<h1 style="text-align:center; margin-bottom:0.2em;"><a href="https://github.com/Saganaki22/higgs-audio-WebUI" target="_blank" style="text-decoration:none; color:inherit;">🎵 Higgs Audio v2 WebUI</a></h1>')
    gr.HTML('<div style="text-align:center; font-size:1.2em; margin-bottom:1.5em;">Generate high-quality speech from text with voice cloning, longform generation, multi speaker generation, voice library, smart batching</div>')
    with gr.Tabs():
        # Tab 1: Basic Generation with Predefined Voices
        with gr.Tab("Basic Generation"):
            with gr.Row():
                with gr.Column():
                    basic_transcript = gr.TextArea(
                        label="Transcript",
                        placeholder="Enter text to synthesize...",
                        value="The sun rises in the east and sets in the west.",
                        lines=5
                    )
                    
                    with gr.Accordion("Voice Settings", open=True):
                        basic_voice_prompt = gr.Dropdown(
                            choices=available_voices,
                            value=config.SMART_VOICE_LABEL,
                            label="Predefined Voice Prompts"
                        )
                        basic_refresh_voices = gr.Button("Refresh Voice List")
                        
                        basic_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the recording environment...",
                            value=""
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                basic_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=config.DEFAULT_TEMPERATURE,
                                    step=0.05,
                                    label="Temperature",
                                    info="Controls randomness in generation (lower = more consistent)"
                                )
                                basic_max_new_tokens = gr.Slider(
                                    minimum=128,
                                    maximum=2048,
                                    value=config.DEFAULT_MAX_NEW_TOKENS,
                                    step=128,
                                    label="Max New Tokens"
                                )
                                basic_seed = gr.Number(
                                    label="Seed (0 for random)",
                                    value=config.DEFAULT_SEED,
                                    precision=0
                                )
                            
                            with gr.Column():
                                basic_top_k = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=config.DEFAULT_TOP_K,
                                    step=1,
                                    label="Top-K",
                                    info="Limits vocabulary to top K most likely tokens"
                                )
                                basic_top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=config.DEFAULT_TOP_P,
                                    step=0.05,
                                    label="Top-P (Nucleus Sampling)",
                                    info="Cumulative probability threshold for token selection"
                                )
                                basic_min_p = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.2,
                                    value=config.DEFAULT_MIN_P,
                                    step=0.01,
                                    label="Min-P",
                                    info="Minimum probability threshold (0 = disabled)"
                                )
                        
                        with gr.Accordion("Advanced Sampling", open=False):
                            with gr.Row():
                                with gr.Column():
                                    basic_repetition_penalty = gr.Slider(
                                        minimum=0.8,
                                        maximum=1.2,
                                        value=config.DEFAULT_REPETITION_PENALTY,
                                        step=0.05,
                                        label="Repetition Penalty",
                                        info="Penalty for repeating tokens (1.0 = no penalty)"
                                    )
                                    basic_do_sample = gr.Checkbox(
                                        label="Enable Sampling",
                                        value=config.DEFAULT_DO_SAMPLE,
                                        info="Use sampling vs greedy decoding"
                                    )
                                
                                with gr.Column():
                                    basic_ras_win_len = gr.Slider(
                                        minimum=0,
                                        maximum=20,
                                        value=config.DEFAULT_RAS_WIN_LEN,
                                        step=1,
                                        label="RAS Window Length",
                                        info="Repetition detection window (0 = disabled)"
                                    )
                                    basic_ras_win_max_num_repeat = gr.Slider(
                                        minimum=0,
                                        maximum=5,
                                        value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
                                        step=1,
                                        label="RAS Max Repeats",
                                        info="Maximum repeats allowed in RAS window"
                                    )
                    
                    with gr.Accordion("🔊 Volume Normalization", open=False):
                        gr.Markdown("*Optional: Normalize audio volume for consistent playback*")
                        with gr.Row():
                            basic_enable_normalization = gr.Checkbox(
                                label="Enable Volume Normalization", 
                                value=False,
                                info="Normalize audio volume level"
                            )
                            basic_target_volume = gr.Slider(
                                0.05, 0.3, 
                                value=0.15, 
                                step=0.01, 
                                label="Target Volume",
                                info="RMS level (0.15 = moderate)"
                            )
                    
                    basic_generate_btn = gr.Button("Generate Audio", variant="primary")
                
                with gr.Column():
                    basic_output_audio = gr.Audio(label="Generated Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #ff9800;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Basic Generation:</b><br>
                        • For best results, use clear, natural sentences.<br>
                        • You can select a predefined voice or use Smart Voice for random high-quality voices.<br>
                        • Scene description can help set the environment (e.g., "in a quiet room").<br>
                        • Adjust temperature for more/less expressive speech.<br>
                        • Try different seeds for voice variety.
                    </div>
                    ''')
        
        # Tab 2: Voice Cloning (YOUR voice only)
        with gr.Tab("Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    vc_transcript = gr.TextArea(
                        label="Transcript",
                        placeholder="Enter text to synthesize with your voice...",
                        value="Hello, this is my cloned voice speaking!",
                        lines=5
                    )
                    
                    with gr.Accordion("Voice Cloning", open=True):
                        gr.Markdown("### Upload Your Voice Sample")
                        vc_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                        if WHISPER_AVAILABLE:
                            gr.Markdown("*Record 10-30 seconds of clear speech for best results. Audio will be auto-transcribed!* ✨")
                        else:
                            gr.Markdown("*Record 10-30 seconds of clear speech for best results. Install whisper for auto-transcription: `pip install faster-whisper`*")
                        
                        # Add a toggle to switch between methods
                        vc_method = gr.Radio(
                            choices=["Official Method", "Alternative Method"],
                            value="Official Method",
                            label="Voice Cloning Method"
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                vc_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=config.DEFAULT_TEMPERATURE,
                                    step=0.05,
                                    label="Temperature",
                                    info="Controls randomness in generation (lower = more consistent)"
                                )
                                vc_max_new_tokens = gr.Slider(
                                    minimum=128,
                                    maximum=2048,
                                    value=config.DEFAULT_MAX_NEW_TOKENS,
                                    step=128,
                                    label="Max New Tokens"
                                )
                                vc_seed = gr.Number(
                                    label="Seed (0 for random)",
                                    value=config.DEFAULT_SEED,
                                    precision=0
                                )
                            
                            with gr.Column():
                                vc_top_k = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=config.DEFAULT_TOP_K,
                                    step=1,
                                    label="Top-K",
                                    info="Limits vocabulary to top K most likely tokens"
                                )
                                vc_top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=config.DEFAULT_TOP_P,
                                    step=0.05,
                                    label="Top-P (Nucleus Sampling)",
                                    info="Cumulative probability threshold for token selection"
                                )
                                vc_min_p = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.2,
                                    value=config.DEFAULT_MIN_P,
                                    step=0.01,
                                    label="Min-P",
                                    info="Minimum probability threshold (0 = disabled)"
                                )
                        
                        with gr.Accordion("Advanced Sampling", open=False):
                            with gr.Row():
                                with gr.Column():
                                    vc_repetition_penalty = gr.Slider(
                                        minimum=0.8,
                                        maximum=1.2,
                                        value=config.DEFAULT_REPETITION_PENALTY,
                                        step=0.05,
                                        label="Repetition Penalty",
                                        info="Penalty for repeating tokens (1.0 = no penalty)"
                                    )
                                    vc_do_sample = gr.Checkbox(
                                        label="Enable Sampling",
                                        value=config.DEFAULT_DO_SAMPLE,
                                        info="Use sampling vs greedy decoding"
                                    )
                                
                                with gr.Column():
                                    vc_ras_win_len = gr.Slider(
                                        minimum=0,
                                        maximum=20,
                                        value=config.DEFAULT_RAS_WIN_LEN,
                                        step=1,
                                        label="RAS Window Length",
                                        info="Repetition detection window (0 = disabled)"
                                    )
                                    vc_ras_win_max_num_repeat = gr.Slider(
                                        minimum=1,
                                        maximum=5,
                                        value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
                                        step=1,
                                        label="RAS Max Repeats",
                                        info="Maximum allowed repetitions in window"
                                    )
                    
                    vc_generate_btn = gr.Button("Clone My Voice & Generate", variant="primary")
                
                with gr.Column():
                    vc_output_audio = gr.Audio(label="Cloned Voice Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #4caf50;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Voice Cloning:</b><br>
                        • Upload a clear 10-30 second sample of your voice, speaking naturally.<br>
                        • The sample will be auto-transcribed for best cloning results.<br>
                        • Use the "Official Method" for most cases; try "Alternative Method" if you want to experiment.<br>
                        • Longer, more expressive samples improve cloning quality.<br>
                        • Use the same seed to reproduce results.
                    </div>
                    ''')
        
        # Tab 3: Long-form Generation
        with gr.Tab("Long-form Generation"):
            with gr.Row():
                with gr.Column():
                    lf_transcript = gr.TextArea(
                        label="Long Transcript",
                        placeholder="Enter long text to synthesize...",
                        value="Artificial intelligence is transforming our world. It helps solve complex problems in healthcare, climate, and education. Machine learning algorithms can process vast amounts of data to find patterns humans might miss. As we develop these technologies, we must consider their ethical implications. The future of AI holds both incredible promise and significant challenges.",
                        lines=10
                    )
                    
                    with gr.Accordion("Voice Options", open=True):
                        lf_voice_choice = gr.Radio(
                            choices=["Smart Voice", "Upload Voice", "Predefined Voice"],
                            value="Smart Voice",
                            label="Voice Selection Method"
                        )
                        
                        with gr.Group(visible=False) as lf_upload_group:
                            lf_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                            if WHISPER_AVAILABLE:
                                gr.Markdown("*Audio will be auto-transcribed for voice cloning!* ✨")
                            else:
                                gr.Markdown("*Install whisper for auto-transcription: `pip install faster-whisper`*")
                        
                        with gr.Group(visible=False) as lf_predefined_group:
                            lf_voice_prompt = gr.Dropdown(
                                choices=available_voices,
                                value=config.SMART_VOICE_LABEL,
                                label="Predefined Voice Prompts"
                            )
                            lf_refresh_voices = gr.Button("Refresh Voice List")
                        
                        lf_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the recording environment...",
                            value=""
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        lf_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=config.DEFAULT_TEMPERATURE,
                            step=0.05,
                            label="Temperature"
                        )
                        lf_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=config.DEFAULT_MAX_NEW_TOKENS,
                            step=128,
                            label="Max New Tokens per Chunk"
                        )
                        lf_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=config.DEFAULT_SEED,
                            precision=0
                        )
                        lf_chunk_size = gr.Slider(
                            minimum=100,
                            maximum=500,
                            value=200,
                            step=50,
                            label="Characters per Chunk"
                        )
                    
                    lf_generate_btn = gr.Button("Generate Long-form Audio", variant="primary")
                
                with gr.Column():
                    lf_output_audio = gr.Audio(label="Generated Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #2196f3;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Long-form Generation:</b><br>
                        • Paste or write long text (stories, articles, etc.) for continuous speech.<br>
                        • Choose Smart Voice, upload your own, or select a predefined voice.<br>
                        • Adjust chunk size for smoother transitions (smaller = more natural, larger = faster).<br>
                        • Scene description can set the mood or environment.<br>
                        • Use consistent voice references for best results in long texts.
                    </div>
                    ''')
            
            # Visibility logic for voice options
            def update_voice_options(choice):
                return {
                    lf_upload_group: gr.update(visible=choice == "Upload Voice"),
                    lf_predefined_group: gr.update(visible=choice == "Predefined Voice")
                }
            
            lf_voice_choice.change(
                fn=update_voice_options,
                inputs=lf_voice_choice,
                outputs=[lf_upload_group, lf_predefined_group]
            )
        
        # Tab 4: Dynamic Multi-Speaker Generation
        with gr.Tab("Multi-Speaker Generation"):
            with gr.Row():
                with gr.Column():
                    ms_transcript = gr.TextArea(
                        label="Multi-Speaker Transcript",
                        placeholder="Enter dialogue with any speaker names in brackets:\n[Alice] Hello there!\n[Bob] How are you?\n[Charlie] Great to see you both!",
                        value="[Alice] Hello there, how are you doing today?\n[Bob] I'm doing great, thank you for asking! How about yourself?\n[Alice] I'm fantastic! It's such a beautiful day outside.\n[Bob] Yes, it really is. Perfect weather for a walk in the park.",
                        lines=8
                    )
                    
                    # Process Text Button
                    ms_process_btn = gr.Button("🔍 Process Text & Detect Speakers", variant="secondary", size="lg")
                    
                    # Speaker Detection Results
                    ms_speaker_info = gr.Markdown("*Click 'Process Text' to detect speakers in your dialogue*")
                    
                    with gr.Accordion("Voice Configuration", open=True):
                        ms_voice_method = gr.Radio(
                            choices=["Smart Voice", "Upload Voices", "Voice Library"],
                            value="Smart Voice",
                            label="Voice Method"
                        )
                        
                        # Dynamic speaker assignment area (initially hidden)
                        with gr.Column(visible=False) as ms_speaker_assignment:
                            gr.Markdown("### 🎭 Speaker Voice Assignment")
                            ms_assignment_content = gr.Markdown("*Speaker assignments will appear here*")
                        
                        # Smart Voice info
                        with gr.Group() as ms_smart_voice_group:
                            gr.Markdown("### Smart Voice Mode")
                            gr.Markdown("*AI will automatically assign distinct voices to each detected speaker*")
                        
                        # Upload voices area (initially hidden)  
                        with gr.Group(visible=False) as ms_upload_group:
                            gr.Markdown("### Upload Voice Samples")
                            gr.Markdown("*Upload a voice sample for each speaker. Files will be auto-transcribed.*")
                            
                            # Create upload slots for up to 10 speakers
                            ms_upload_slots = []
                            for i in range(10):
                                with gr.Row(visible=False) as upload_row:
                                    speaker_audio = gr.Audio(
                                        label=f"Speaker {i} Voice Sample",
                                        type="numpy",
                                        scale=3
                                    )
                                    ms_upload_slots.append((upload_row, speaker_audio))
                        
                        # Voice library selection area (initially hidden)
                        with gr.Group(visible=False) as ms_library_group:
                            gr.Markdown("### Select Voices from Library")
                            gr.Markdown("*Choose a saved voice for each speaker*")
                            
                            # Create dropdown slots for up to 10 speakers
                            ms_library_slots = []
                            for i in range(10):
                                speaker_dropdown = gr.Dropdown(
                                    choices=get_current_available_voices(),
                                    value=config.SMART_VOICE_LABEL,
                                    label=f"Speaker {i} Voice",
                                    visible=False
                                )
                                ms_library_slots.append(speaker_dropdown)
                            
                            ms_refresh_library_voices = gr.Button("Refresh Voice Library", visible=False)
                        
                        ms_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the conversation setting...",
                            value="A friendly conversation between people in a quiet room."
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        ms_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=config.DEFAULT_TEMPERATURE,
                            step=0.05,
                            label="Temperature"
                        )
                        ms_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=config.DEFAULT_MAX_NEW_TOKENS,
                            step=128,
                            label="Max New Tokens per Segment"
                        )
                        ms_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=config.DEFAULT_SEED,
                            precision=0
                        )
                    
                    with gr.Accordion("🔊 Volume Normalization", open=True):
                        gr.Markdown("*Fix volume inconsistencies between speakers - recommended for all multi-speaker audio*")
                        with gr.Row():
                            with gr.Column(scale=3):
                                ms_enable_normalization = gr.Checkbox(
                                    label="Enable Volume Normalization", 
                                    value=True,
                                    info="Automatically balance speaker volumes"
                                )
                            with gr.Column(scale=3):
                                ms_normalization_method = gr.Dropdown(
                                    choices=["adaptive", "simple", "segment-based"],
                                    value="adaptive",
                                    label="Normalization Method",
                                    info="Adaptive = sliding windows, Simple = whole audio, Segment = detect speakers"
                                )
                            with gr.Column(scale=2):
                                ms_target_volume = gr.Slider(
                                    0.05, 0.3,
                                    value=config.DEFAULT_TARGET_VOLUME,
                                    step=0.01,
                                    label="Target Volume",
                                    info="RMS level (0.15 = moderate)"
                                )
                    
                    with gr.Accordion("⏸️ Speaker Timing", open=False):
                        gr.Markdown("*Control timing and pauses between different speakers*")
                        ms_speaker_pause = gr.Slider(
                            0.0, 2.0,
                            value=config.DEFAULT_SPEAKER_PAUSE_SECONDS,
                            step=0.1,
                            label="Pause Between Speakers (seconds)",
                            info="Duration of silence when speakers change (0.0 = no pause, 0.3 = default)"
                        )
                    
                    ms_generate_btn = gr.Button("Generate Multi-Speaker Audio", variant="primary", interactive=False)
                
                with gr.Column():
                    ms_output_audio = gr.Audio(label="Generated Multi-Speaker Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #e91e63;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 New Dynamic Multi-Speaker System:</b><br>
                        • Use ANY speaker names in brackets: [Alice], [Bob], [Character Name]<br>
                        • Click "Process Text" to automatically detect all speakers<br>
                        • Supports unlimited number of speakers<br>
                        • Smart Voice: AI assigns distinct voices automatically<br>
                        • Upload Voices: Clone real voices for each speaker<br>
                        • Voice Library: Use your saved voices for each speaker<br>
                        • Works with your existing scripts - no need to change character names!
                    </div>
                    ''')
            
            # State variables to store detected speakers and assignments
            ms_detected_speakers = gr.State([])
            ms_speaker_mapping = gr.State({})
        
        # Tab 5: Voice Library Management
        with gr.Tab("Voice Library"):
            gr.HTML("<h2 style='text-align: center;'>🎵 Voice Library Management</h2>")
            gr.HTML("<p style='text-align: center;'>Save voices with custom generation parameters for perfect reuse!</p>")
            
            with gr.Row():
                # Left Column: Add New Voice
                with gr.Column(scale=1):
                    gr.Markdown("## 🎤 Add New Voice")
                    
                    # Step 1: Upload
                    gr.Markdown("### Step 1: Upload Voice Sample")
                    vl_new_voice_audio = gr.Audio(label="Upload Voice Sample", type="filepath")
                    
                    # Step 2: Configure Parameters
                    gr.Markdown("### Step 2: Configure Generation Parameters")
                    with gr.Accordion("Generation Parameters", open=True):
                        with gr.Row():
                            with gr.Column():
                                vl_temperature = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=config.DEFAULT_TEMPERATURE, step=0.05,
                                    label="Temperature", info="Controls randomness"
                                )
                                vl_max_new_tokens = gr.Slider(
                                    minimum=128, maximum=2048, value=config.DEFAULT_MAX_NEW_TOKENS, step=128,
                                    label="Max New Tokens"
                                )
                                vl_seed = gr.Number(
                                    label="Seed (0 for random)", value=config.DEFAULT_SEED, precision=0
                                )
                            
                            with gr.Column():
                                vl_top_k = gr.Slider(
                                    minimum=1, maximum=100, value=config.DEFAULT_TOP_K, step=1,
                                    label="Top-K", info="Vocabulary limit"
                                )
                                vl_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=config.DEFAULT_TOP_P, step=0.05,
                                    label="Top-P", info="Nucleus sampling"
                                )
                                vl_min_p = gr.Slider(
                                    minimum=0.0, maximum=0.2, value=config.DEFAULT_MIN_P, step=0.01,
                                    label="Min-P", info="Min probability (0 = disabled)"
                                )
                        
                        with gr.Accordion("Advanced Parameters", open=False):
                            with gr.Row():
                                with gr.Column():
                                    vl_repetition_penalty = gr.Slider(
                                        minimum=0.8, maximum=1.2, value=config.DEFAULT_REPETITION_PENALTY, step=0.05,
                                        label="Repetition Penalty", info="Prevent repetition"
                                    )
                                    vl_do_sample = gr.Checkbox(
                                        label="Enable Sampling", value=config.DEFAULT_DO_SAMPLE, info="Use sampling vs greedy"
                                    )
                                
                                with gr.Column():
                                    vl_ras_win_len = gr.Slider(
                                        minimum=0, maximum=20, value=config.DEFAULT_RAS_WIN_LEN, step=1,
                                        label="RAS Window Length", info="Repetition window"
                                    )
                                    vl_ras_win_max_num_repeat = gr.Slider(
                                        minimum=1, maximum=5, value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT, step=1,
                                        label="RAS Max Repeats", info="Max allowed repeats"
                                    )
                    
                    # Step 3: Test & Save
                    gr.Markdown("### Step 3: Test & Save")
                    vl_test_text = gr.Textbox(
                        label="Test Text", 
                        placeholder="Enter text to test your voice with these settings...",
                        value="This is a test of my voice with custom parameters.",
                        lines=3
                    )
                    
                    with gr.Row():
                        vl_test_btn = gr.Button("🎵 Test Voice", variant="primary")
                        vl_clear_test_btn = gr.Button("🔄 Clear Test", variant="secondary")
                    
                    vl_new_voice_name = gr.Textbox(
                        label="Voice Name", 
                        placeholder="Enter a unique name for this voice..."
                    )
                    vl_voice_description = gr.Textbox(
                        label="Description (Optional)", 
                        placeholder="Describe this voice or when to use it...",
                        lines=2
                    )
                    
                    vl_save_btn = gr.Button("💾 Save Voice to Library", variant="stop", size="lg")
                    
                    if WHISPER_AVAILABLE:
                        gr.HTML("<p><em>✨ Voice will be auto-transcribed when saved!</em></p>")
                
                # Right Column: Manage Existing Voices
                with gr.Column(scale=1):
                    gr.Markdown("## 🗂️ Manage Voice Library")
                    
                    # Voice Selection & Info
                    vl_voice_selector = gr.Dropdown(
                        label="Select Voice to Manage",
                        choices=["None"] + get_voice_library_voices(),
                        value="None"
                    )
                    
                    # Voice Info Display
                    vl_voice_info = gr.Markdown("*Select a voice to view details*")
                    
                    # Voice Parameters (for editing existing voices)
                    with gr.Accordion("Edit Voice Parameters", open=False) as vl_edit_accordion:
                        gr.Markdown("*Modify generation parameters for the selected voice*")
                        with gr.Row():
                            with gr.Column():
                                vl_edit_temperature = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=config.DEFAULT_TEMPERATURE, step=0.05,
                                    label="Temperature"
                                )
                                vl_edit_max_new_tokens = gr.Slider(
                                    minimum=128, maximum=2048, value=config.DEFAULT_MAX_NEW_TOKENS, step=128,
                                    label="Max New Tokens"
                                )
                                vl_edit_seed = gr.Number(
                                    label="Seed", value=config.DEFAULT_SEED, precision=0
                                )
                            
                            with gr.Column():
                                vl_edit_top_k = gr.Slider(
                                    minimum=1, maximum=100, value=config.DEFAULT_TOP_K, step=1,
                                    label="Top-K"
                                )
                                vl_edit_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=config.DEFAULT_TOP_P, step=0.05,
                                    label="Top-P"
                                )
                                vl_edit_min_p = gr.Slider(
                                    minimum=0.0, maximum=0.2, value=config.DEFAULT_MIN_P, step=0.01,
                                    label="Min-P"
                                )
                        
                        with gr.Row():
                            with gr.Column():
                                vl_edit_repetition_penalty = gr.Slider(
                                    minimum=0.8, maximum=1.2, value=config.DEFAULT_REPETITION_PENALTY, step=0.05,
                                    label="Repetition Penalty"
                                )
                                vl_edit_do_sample = gr.Checkbox(
                                    label="Enable Sampling", value=config.DEFAULT_DO_SAMPLE
                                )
                            
                            with gr.Column():
                                vl_edit_ras_win_len = gr.Slider(
                                    minimum=0, maximum=20, value=config.DEFAULT_RAS_WIN_LEN, step=1,
                                    label="RAS Window Length"
                                )
                                vl_edit_ras_win_max_num_repeat = gr.Slider(
                                    minimum=1, maximum=5, value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT, step=1,
                                    label="RAS Max Repeats"
                                )
                        
                        vl_edit_description = gr.Textbox(
                            label="Description", 
                            placeholder="Describe this voice...",
                            lines=2
                        )
                    
                    # Management Buttons
                    with gr.Row():
                        vl_refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
                        vl_save_changes_btn = gr.Button("💾 Save Changes", variant="primary")
                        vl_delete_btn = gr.Button("🗑️ Delete Voice", variant="stop")
                    
                    # Test Results & Status
                    vl_test_audio = gr.Audio(label="🎧 Voice Test Result", type="filepath", show_download_button=True)
                    vl_test_status = gr.Textbox(label="Test Status", interactive=False)
                    vl_save_status = gr.Textbox(label="Save Status", interactive=False)
                    vl_delete_status = gr.Textbox(label="Management Status", interactive=False)

    # Function to handle voice cloning method selection
    def handle_voice_clone_generation(
        transcript, uploaded_voice, temperature, max_new_tokens, seed, method,
        top_k, top_p, min_p, repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample
    ):
        if method == "Official Method":
            return generate_voice_clone(transcript, uploaded_voice, temperature, max_new_tokens, seed,
                                      top_k, top_p, min_p, repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample)
        else:
            return generate_voice_clone_alternative(transcript, uploaded_voice, temperature, max_new_tokens, seed,
                                                  top_k, top_p, min_p, repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample)
    
    # Voice Library Event Handlers
    def handle_test_voice_with_params(audio_data, test_text, temperature, max_new_tokens, seed,
                                    top_k, top_p, min_p, repetition_penalty, ras_win_len, 
                                    ras_win_max_num_repeat, do_sample):
        """Test voice with custom generation parameters - isolated test that doesn't interfere with voice library"""
        if audio_data is None:
            return None, "❌ Please upload an audio sample first"
        
        if not test_text.strip():
            test_text = "This is a test of my voice with custom parameters."
        
        try:
            waveform, sample_rate = torchaudio.load(audio_data)
            audio_array = waveform.numpy()
            if audio_array.ndim > 1:
                audio_array = audio_array[0]
            uploaded_voice = (sample_rate, audio_array)
            output_path = generation_service.generate_voice_clone(
                test_text,
                uploaded_voice,
                temperature,
                max_new_tokens,
                seed,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                ras_win_len,
                ras_win_max_num_repeat,
                do_sample,
            )
            return output_path, "✅ Voice test completed successfully!"
        except Exception as e:
            return None, f"❌ Error testing voice: {str(e)}"
    
    def handle_save_voice_with_config(audio_data, voice_name, description, temperature, max_new_tokens, seed,
                                     top_k, top_p, min_p, repetition_penalty, ras_win_len, 
                                     ras_win_max_num_repeat, do_sample):
        """Save voice to library with custom configuration"""
        if audio_data is None:
            return "❌ Please upload an audio sample first"
        
        if not voice_name or not voice_name.strip():
            return "❌ Please enter a voice name"
        
        try:
            # Load audio file and convert to array format
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_data)
            audio_array = waveform.numpy()[0]  # Convert to numpy array
            
            # Save the audio file first
            status = save_voice_to_library(audio_array, sample_rate, voice_name.strip())
            
            # If audio saved successfully, save custom config
            if status.startswith("✅"):
                voice_config_data = {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "top_k": top_k,
                    "top_p": top_p,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "ras_win_len": ras_win_len,
                    "ras_win_max_num_repeat": ras_win_max_num_repeat,
                    "do_sample": do_sample,
                    "description": description.strip() if description else "",
                    "tags": []
                }
                
                if save_voice_config(voice_name.strip(), voice_config_data):
                    return f"✅ Voice '{voice_name}' saved with custom parameters!"
                else:
                    return f"⚠️ Voice saved but failed to save parameters"
            else:
                return status
                
        except Exception as e:
            return f"❌ Error saving voice: {str(e)}"
    
    def handle_voice_selection(voice_name):
        """Handle voice selection and load its configuration"""
        if not voice_name or voice_name == "None":
            return (
                "*Select a voice to view details*",
                config.DEFAULT_TEMPERATURE,
                config.DEFAULT_MAX_NEW_TOKENS,
                config.DEFAULT_SEED,
                config.DEFAULT_TOP_K,
                config.DEFAULT_TOP_P,
                config.DEFAULT_MIN_P,
                config.DEFAULT_REPETITION_PENALTY,
                config.DEFAULT_RAS_WIN_LEN,
                config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
                config.DEFAULT_DO_SAMPLE,
                "",
            )
        
        # Load voice configuration
        voice_config = load_voice_config(voice_name)
        
        # Get voice info
        voice_path = os.path.join(
            config.VOICE_LIBRARY_DIR,
            f"{voice_name}{config.VOICE_LIBRARY_AUDIO_EXTENSION}"
        )
        txt_path = os.path.join(
            config.VOICE_LIBRARY_DIR,
            f"{voice_name}{config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION}"
        )
        
        info_text = f"## 🎤 {voice_name}\n\n"
        
        # Add description if available
        if voice_config.get("description"):
            info_text += f"**Description:** {voice_config['description']}\n\n"
        
        # Add transcript preview
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                    if len(transcript) > 200:
                        transcript = transcript[:200] + "..."
                    info_text += f"**Sample Text:** *{transcript}*\n\n"
            except:
                pass
        
        # Add parameter summary
        info_text += "**Current Parameters:**\n"
        info_text += f"- Temperature: {voice_config['temperature']}\n"
        info_text += f"- Max Tokens: {voice_config['max_new_tokens']}\n"
        info_text += f"- Top-K: {voice_config['top_k']}, Top-P: {voice_config['top_p']}\n"
        info_text += f"- RAS Window: {voice_config['ras_win_len']}\n"

        return (
            info_text,
            voice_config['temperature'],
            voice_config['max_new_tokens'],
            voice_config['seed'],
            voice_config['top_k'],
            voice_config['top_p'],
            voice_config['min_p'],
            voice_config['repetition_penalty'],
            voice_config['ras_win_len'],
            voice_config['ras_win_max_num_repeat'],
            voice_config['do_sample'],
            voice_config.get('description', ''),
        )
    
    def handle_save_voice_changes(voice_name, description, temperature, max_new_tokens, seed,
                                 top_k, top_p, min_p, repetition_penalty, ras_win_len, 
                                 ras_win_max_num_repeat, do_sample):
        """Save changes to an existing voice's configuration"""
        if not voice_name or voice_name == "None":
            return "❌ Please select a voice first"
        
        try:
            voice_config_data = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "ras_win_len": ras_win_len,
                "ras_win_max_num_repeat": ras_win_max_num_repeat,
                "do_sample": do_sample,
                "description": description.strip() if description else "",
                "tags": []
            }
            
            if save_voice_config(voice_name, voice_config_data):
                return f"✅ Updated parameters for '{voice_name}'"
            else:
                return f"❌ Failed to save changes for '{voice_name}'"
                
        except Exception as e:
            return f"❌ Error saving changes: {str(e)}"
    
    def handle_delete_voice(voice_name):
        """Delete a voice and its associated files"""
        if not voice_name or voice_name == "None":
            return "❌ Please select a voice first"
        
        try:
            voice_path = os.path.join(
                config.VOICE_LIBRARY_DIR,
                f"{voice_name}{config.VOICE_LIBRARY_AUDIO_EXTENSION}"
            )
            txt_path = os.path.join(
                config.VOICE_LIBRARY_DIR,
                f"{voice_name}{config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION}"
            )
            config_path = get_voice_config_path(voice_name)
            
            files_deleted = []
            
            # Delete audio file
            if os.path.exists(voice_path):
                os.remove(voice_path)
                files_deleted.append("audio")
            
            # Delete transcript file
            if os.path.exists(txt_path):
                os.remove(txt_path)
                files_deleted.append("transcript")
            
            # Delete config file
            if os.path.exists(config_path):
                os.remove(config_path)
                files_deleted.append("config")
            
            if files_deleted:
                return f"✅ Deleted '{voice_name}' ({', '.join(files_deleted)})"
            else:
                return f"❌ Voice '{voice_name}' not found"
                
        except Exception as e:
            return f"❌ Error deleting voice: {str(e)}"
    
    def handle_clear_test():
        return None, "Test cleared. Upload voice and try again."
    
    def process_multi_speaker_text(text):
        """Process multi-speaker text and create dynamic interface"""
        if not text or not text.strip():
            return (
                "*No text provided*",
                [],
                {},
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                "*No text to process*"
            )
        
        # Detect speakers
        speakers = detect_dynamic_speakers(text)
        
        if not speakers:
            return (
                "*❌ No speakers detected. Use format: [Speaker Name] dialogue*",
                [],
                {},
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                "*No speakers detected*"
            )
        
        # Create speaker mapping (Speaker Name -> SPEAKER0, SPEAKER1, etc.)
        speaker_mapping = {}
        for i, speaker in enumerate(speakers):
            speaker_mapping[speaker] = f"SPEAKER{i}"
        
        # Create info display
        info_text = f"**🎭 Detected {len(speakers)} speakers:**\n\n"
        for i, speaker in enumerate(speakers):
            info_text += f"• **{speaker}** → SPEAKER{i}\n"
        
        info_text += f"\n*Now select voice assignment method below and assign voices to each speaker*"
        
        return (
            info_text,
            speakers,
            speaker_mapping,
            gr.update(interactive=True),
            gr.update(visible=False),  # upload group
            gr.update(visible=False),  # library group  
            gr.update(visible=True),   # smart voice group
            gr.update(visible=True),   # speaker assignment
            create_speaker_assignment_interface(speakers, "Smart Voice")  # assignment content
        )
    
    def update_voice_method_visibility(voice_method, detected_speakers):
        """Update visibility of voice assignment sections based on method"""
        # Get updates for upload and library slots
        upload_updates = update_upload_slots(detected_speakers if voice_method == "Upload Voices" else [])
        library_updates = update_library_slots(detected_speakers if voice_method == "Voice Library" else [])
        
        # Base outputs
        outputs = [
            gr.update(visible=voice_method == "Smart Voice"),    # smart voice group
            gr.update(visible=voice_method == "Upload Voices"),  # upload group
            gr.update(visible=voice_method == "Voice Library"),  # library group
            create_speaker_assignment_interface(detected_speakers, voice_method),  # assignment content
        ]
        
        # Add upload slot updates (10 row visibility + 10 audio components)
        outputs.extend(upload_updates)
        
        # Add library slot updates (10 dropdown components)
        outputs.extend(library_updates)
        
        # Add refresh button
        outputs.append(gr.update(visible=(voice_method == "Voice Library" and len(detected_speakers) > 0)))
        
        return outputs
    
    def update_upload_slots(speakers):
        """Update upload slot visibility and labels based on detected speakers"""
        updates = []
        
        # Update row visibility (10 rows)
        for i in range(10):
            if i < len(speakers):
                updates.append(gr.update(visible=True))  # Row visible
            else:
                updates.append(gr.update(visible=False))  # Row hidden
        
        # Update audio component labels (10 audio components)
        for i in range(10):
            if i < len(speakers):
                speaker_name = speakers[i]
                updates.append(gr.update(label=f"{speaker_name} Voice Sample"))
            else:
                updates.append(gr.update(label=f"Speaker {i} Voice Sample"))
        
        return updates
    
    def update_library_slots(speakers):
        """Update library dropdown visibility and labels based on detected speakers"""
        updates = []
        
        # Update dropdown visibility and labels (10 dropdowns)
        for i in range(10):
            if i < len(speakers):
                speaker_name = speakers[i]
                updates.append(gr.update(
                    visible=True,
                    label=f"{speaker_name} Voice",
                    value=config.SMART_VOICE_LABEL
                ))
            else:
                updates.append(gr.update(visible=False))
        
        return updates
    
    def create_speaker_assignment_interface(speakers, voice_method):
        """Create dynamic speaker assignment interface based on voice method"""
        if not speakers:
            return gr.update()
        
        if voice_method == "Smart Voice":
            # For Smart Voice, show info that AI will assign automatically
            content = ""
            for speaker in speakers:
                content += f"**{speaker}**: AI will automatically assign a distinct voice\n\n"
            content += "*No manual assignment needed - the AI will ensure each speaker has a unique voice*"
            return gr.update(value=content)
        
        elif voice_method == "Voice Library":
            # For Voice Library, show voice options for each speaker
            available_voices = get_current_available_voices()
            content = "**🎭 Voice Library Assignment:**\n\n"
            content += "*Use the dropdowns below to assign specific voices from your library to each character*\n\n"
            
            for i, speaker in enumerate(speakers):
                content += f"**{speaker}** → Use dropdown to select voice\n"
            
            content += f"\n**Available voices:** {', '.join(available_voices)}\n\n"
            content += (
                f"*💡 Select '{config.SMART_VOICE_LABEL}' to let AI pick a voice for that character*\n"
            )
            content += "*💡 Click 'Refresh Voice Library' if you've added new voices*"
            return gr.update(value=content)
        
        elif voice_method == "Upload Voices":
            # For Upload Voices, show upload areas for each speaker  
            content = "**Upload Voice Samples:**\n\n"
            for speaker in speakers:
                content += f"**{speaker}**: Upload a voice sample to clone this speaker's voice\n\n"
            content += "*💡 Upload functionality coming soon! For now, use Smart Voice mode.*"
            return gr.update(value=content)
        
        return gr.update()
    

    
    def generate_dynamic_multi_speaker(
        text,
        voice_method,
        speaker_mapping,
        temperature,
        max_new_tokens,
        seed,
        scene_description,
        enable_normalization,
        normalization_method,
        target_volume,
        speaker_pause_duration,
        *voice_components,
    ):
        return generation_service.generate_dynamic_multi_speaker(
            text,
            voice_method,
            speaker_mapping,
            temperature,
            max_new_tokens,
            seed,
            scene_description,
            enable_normalization,
            normalization_method,
            target_volume,
            speaker_pause_duration,
            *voice_components,
        )

    def auto_populate_voice_name(audio_data):
        """Automatically populate voice name from uploaded audio filename"""
        if audio_data is None:
            return ""
        
        try:
            # Check if this is a file path (when audio is uploaded as file)
            if isinstance(audio_data, str) and audio_data.strip():
                # Extract filename without extension
                import os
                filename = os.path.basename(audio_data)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Clean up the name (remove special characters, keep alphanumeric, hyphens, underscores)
                clean_name = "".join(c for c in name_without_ext if c.isalnum() or c in "-_ ").strip()
                # Replace spaces with underscores and remove multiple consecutive underscores
                clean_name = "_".join(clean_name.split())
                clean_name = "_".join(filter(None, clean_name.split("_")))  # Remove empty parts
                
                # Limit length and ensure it's not empty
                if clean_name and len(clean_name) > 0:
                    return clean_name[:50]
                else:
                    return "uploaded_voice"
            
            # For other cases (microphone recordings), generate timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"voice_{timestamp}"
            
        except Exception as e:
            # Fallback if anything goes wrong
            print(f"Error auto-populating voice name: {e}")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"voice_{timestamp}"
    
    def handle_refresh_library():
        new_choices = ["None"] + get_voice_library_voices()
        return gr.update(choices=new_choices, value="None")

    # Event handling for Basic Generation
    basic_generate_btn.click(
        fn=generate_basic,
        inputs=[basic_transcript, basic_voice_prompt, basic_temperature, basic_max_new_tokens, basic_seed, basic_scene_description, 
               basic_top_k, basic_top_p, basic_min_p, basic_repetition_penalty, basic_ras_win_len, basic_ras_win_max_num_repeat, basic_do_sample,
               basic_enable_normalization, basic_target_volume],
        outputs=basic_output_audio
    )
    
    basic_refresh_voices.click(
        fn=refresh_voice_list,
        outputs=basic_voice_prompt
    )
    
    # Event handling for Voice Cloning with method selection
    vc_generate_btn.click(
        fn=handle_voice_clone_generation,
        inputs=[vc_transcript, vc_uploaded_voice, vc_temperature, vc_max_new_tokens, vc_seed, vc_method,
               vc_top_k, vc_top_p, vc_min_p, vc_repetition_penalty, vc_ras_win_len, vc_ras_win_max_num_repeat, vc_do_sample],
        outputs=vc_output_audio
    )
    
    # Event handling for Long-form Generation
    lf_generate_btn.click(
        fn=generate_longform,
        inputs=[lf_transcript, lf_voice_choice, lf_uploaded_voice, lf_voice_prompt, lf_temperature, lf_max_new_tokens, lf_seed, lf_scene_description, lf_chunk_size],
        outputs=lf_output_audio
    )
    
    lf_refresh_voices.click(
        fn=refresh_voice_list,
        outputs=lf_voice_prompt
    )
    
    # Event handling for Dynamic Multi-Speaker Generation
    ms_process_btn.click(
        fn=process_multi_speaker_text,
        inputs=[ms_transcript],
        outputs=[
            ms_speaker_info, ms_detected_speakers, ms_speaker_mapping, ms_generate_btn,
            ms_upload_group, ms_library_group, ms_smart_voice_group, ms_speaker_assignment, ms_assignment_content
        ]
    )
    
    # Prepare outputs for voice method change handler
    voice_method_outputs = [
        ms_smart_voice_group, ms_upload_group, ms_library_group, ms_assignment_content
    ]
    
    # Add upload slot components (10 rows + 10 audio components)
    for upload_row, upload_audio in ms_upload_slots:
        voice_method_outputs.append(upload_row)  # Row visibility
    for upload_row, upload_audio in ms_upload_slots:
        voice_method_outputs.append(upload_audio)  # Audio component updates
    
    # Add library slot components (10 dropdowns)
    voice_method_outputs.extend(ms_library_slots)
    
    # Add refresh button
    voice_method_outputs.append(ms_refresh_library_voices)
    
    ms_voice_method.change(
        fn=update_voice_method_visibility,
        inputs=[ms_voice_method, ms_detected_speakers],
        outputs=voice_method_outputs
    )
    
    # Prepare inputs for generation button (includes all voice components)
    generation_inputs = [
        ms_transcript, ms_voice_method, ms_speaker_mapping, ms_temperature, ms_max_new_tokens, ms_seed, ms_scene_description,
        ms_enable_normalization, ms_normalization_method, ms_target_volume, ms_speaker_pause
    ]
    
    # Add all upload audio components
    for upload_row, upload_audio in ms_upload_slots:
        generation_inputs.append(upload_audio)
    
    # Add all library dropdown components  
    generation_inputs.extend(ms_library_slots)
    
    ms_generate_btn.click(
        fn=generate_dynamic_multi_speaker,
        inputs=generation_inputs,
        outputs=[ms_output_audio]
    )
    
    # Refresh voice library dropdown choices
    def refresh_multi_speaker_voices():
        """Refresh voice choices for multi-speaker dropdowns"""
        choices = get_current_available_voices()
        return [gr.update(choices=choices) for _ in range(10)]
    
    ms_refresh_library_voices.click(
        fn=refresh_multi_speaker_voices,
        outputs=ms_library_slots
    )
    
    # Event handling for Voice Library
    # Auto-populate voice name when audio is uploaded
    vl_new_voice_audio.upload(
        fn=auto_populate_voice_name,
        inputs=[vl_new_voice_audio],
        outputs=[vl_new_voice_name]
    )
    
    vl_test_btn.click(
        fn=handle_test_voice_with_params,
        inputs=[vl_new_voice_audio, vl_test_text, vl_temperature, vl_max_new_tokens, vl_seed,
               vl_top_k, vl_top_p, vl_min_p, vl_repetition_penalty, vl_ras_win_len, 
               vl_ras_win_max_num_repeat, vl_do_sample],
        outputs=[vl_test_audio, vl_test_status]
    )
    
    vl_clear_test_btn.click(
        fn=handle_clear_test,
        outputs=[vl_test_audio, vl_test_status]
    )
    
    vl_save_btn.click(
        fn=handle_save_voice_with_config,
        inputs=[vl_new_voice_audio, vl_new_voice_name, vl_voice_description, vl_temperature, vl_max_new_tokens, vl_seed,
               vl_top_k, vl_top_p, vl_min_p, vl_repetition_penalty, vl_ras_win_len, 
               vl_ras_win_max_num_repeat, vl_do_sample],
        outputs=[vl_save_status]
    )
    
    # Voice selector change handler
    vl_voice_selector.change(
        fn=handle_voice_selection,
        inputs=[vl_voice_selector],
        outputs=[vl_voice_info, vl_edit_temperature, vl_edit_max_new_tokens, vl_edit_seed,
                vl_edit_top_k, vl_edit_top_p, vl_edit_min_p, vl_edit_repetition_penalty,
                vl_edit_ras_win_len, vl_edit_ras_win_max_num_repeat, vl_edit_do_sample, vl_edit_description]
    )
    
    # Save changes to existing voice
    vl_save_changes_btn.click(
        fn=handle_save_voice_changes,
        inputs=[vl_voice_selector, vl_edit_description, vl_edit_temperature, vl_edit_max_new_tokens, vl_edit_seed,
               vl_edit_top_k, vl_edit_top_p, vl_edit_min_p, vl_edit_repetition_penalty,
               vl_edit_ras_win_len, vl_edit_ras_win_max_num_repeat, vl_edit_do_sample],
        outputs=[vl_delete_status]
    )
    
    vl_delete_btn.click(
        fn=handle_delete_voice,
        inputs=[vl_voice_selector],
        outputs=[vl_delete_status]
    )
    
    vl_refresh_btn.click(
        fn=handle_refresh_library,
        outputs=[vl_voice_selector]
    )

    # --- Place the GitHub link at the bottom of the app ---
    gr.HTML("""
    <div style='width:100%;text-align:center;margin-top:2em;margin-bottom:1em;'>
        <a href='https://github.com/Saganaki22/higgs-audio-WebUI' target='_blank' style='color:#fff;font-size:1.1em;text-decoration:underline;'>Github</a>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Higgs Audio v2 Generator WebUI")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link via Hugging Face")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server host address")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    
    print("🚀 Starting Higgs Audio v2 Generator...")
    print("✨ Features: Voice Cloning, Multi-Speaker, Caching, Auto-Transcription, Enhanced Audio Processing")
    
    if args.share:
        print("🌐 Creating public shareable link via Hugging Face...")
        print("⚠️  Warning: Your interface will be publicly accessible to anyone with the link!")
    
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True
    )
