"""Gradio tab definition for managing the voice library."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import gradio as gr
import torchaudio

from app import config
from app.services import GenerationService, VoiceLibrary


@dataclass
class VoiceLibraryTabElements:
    """Components used across voice library callbacks."""

    new_voice_audio: gr.Audio
    new_voice_name: gr.Textbox
    voice_description: gr.Textbox
    temperature: gr.Slider
    max_new_tokens: gr.Slider
    seed: gr.Number
    top_k: gr.Slider
    top_p: gr.Slider
    min_p: gr.Slider
    repetition_penalty: gr.Slider
    ras_win_len: gr.Slider
    ras_win_max_num_repeat: gr.Slider
    do_sample: gr.Checkbox
    test_text: gr.Textbox
    test_button: gr.Button
    clear_test_button: gr.Button
    save_button: gr.Button
    test_audio: gr.Audio
    test_status: gr.Textbox
    save_status: gr.Textbox
    voice_selector: gr.Dropdown
    voice_info: gr.Markdown
    edit_temperature: gr.Slider
    edit_max_new_tokens: gr.Slider
    edit_seed: gr.Number
    edit_top_k: gr.Slider
    edit_top_p: gr.Slider
    edit_min_p: gr.Slider
    edit_repetition_penalty: gr.Slider
    edit_ras_win_len: gr.Slider
    edit_ras_win_max_num_repeat: gr.Slider
    edit_do_sample: gr.Checkbox
    edit_description: gr.Textbox
    refresh_button: gr.Button
    save_changes_button: gr.Button
    delete_button: gr.Button
    delete_status: gr.Textbox


def build_tab(
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
    whisper_available: bool,
) -> VoiceLibraryTabElements:
    """Render the voice library management tab."""

    with gr.Tab("Voice Library"):
        gr.HTML("<h2 style='text-align: center;'>🎵 Voice Library Management</h2>")
        gr.HTML(
            "<p style='text-align: center;'>Save voices with custom generation parameters for perfect reuse!</p>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎤 Add New Voice")
                gr.Markdown("### Step 1: Upload Voice Sample")
                new_voice_audio = gr.Audio(
                    label="Upload Voice Sample", type="filepath"
                )

                gr.Markdown("### Step 2: Configure Generation Parameters")
                with gr.Accordion("Generation Parameters", open=True):
                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=config.DEFAULT_TEMPERATURE,
                                step=0.05,
                                label="Temperature",
                                info="Controls randomness",
                            )
                            max_new_tokens = gr.Slider(
                                minimum=128,
                                maximum=2048,
                                value=config.DEFAULT_MAX_NEW_TOKENS,
                                step=128,
                                label="Max New Tokens",
                            )
                            seed = gr.Number(
                                label="Seed (0 for random)",
                                value=config.DEFAULT_SEED,
                                precision=0,
                            )

                        with gr.Column():
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=config.DEFAULT_TOP_K,
                                step=1,
                                label="Top-K",
                                info="Vocabulary limit",
                            )
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=config.DEFAULT_TOP_P,
                                step=0.05,
                                label="Top-P",
                                info="Nucleus sampling",
                            )
                            min_p = gr.Slider(
                                minimum=0.0,
                                maximum=0.2,
                                value=config.DEFAULT_MIN_P,
                                step=0.01,
                                label="Min-P",
                                info="Min probability (0 = disabled)",
                            )

                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                repetition_penalty = gr.Slider(
                                    minimum=0.8,
                                    maximum=1.2,
                                    value=config.DEFAULT_REPETITION_PENALTY,
                                    step=0.05,
                                    label="Repetition Penalty",
                                    info="Prevent repetition",
                                )
                                do_sample = gr.Checkbox(
                                    label="Enable Sampling",
                                    value=config.DEFAULT_DO_SAMPLE,
                                    info="Use sampling vs greedy",
                                )

                            with gr.Column():
                                ras_win_len = gr.Slider(
                                    minimum=0,
                                    maximum=20,
                                    value=config.DEFAULT_RAS_WIN_LEN,
                                    step=1,
                                    label="RAS Window Length",
                                    info="Repetition window",
                                )
                                ras_win_max_num_repeat = gr.Slider(
                                    minimum=1,
                                    maximum=5,
                                    value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
                                    step=1,
                                    label="RAS Max Repeats",
                                    info="Max allowed repeats",
                                )

                gr.Markdown("### Step 3: Test & Save")
                test_text = gr.Textbox(
                    label="Test Text",
                    placeholder="Enter text to test your voice with these settings...",
                    value="This is a test of my voice with custom parameters.",
                    lines=3,
                )

                with gr.Row():
                    test_button = gr.Button("🎵 Test Voice", variant="primary")
                    clear_test_button = gr.Button(
                        "🔄 Clear Test", variant="secondary"
                    )

                new_voice_name = gr.Textbox(
                    label="Voice Name",
                    placeholder="Enter a unique name for this voice...",
                )
                voice_description = gr.Textbox(
                    label="Description (Optional)",
                    placeholder="Describe this voice or when to use it...",
                    lines=2,
                )

                save_button = gr.Button(
                    "💾 Save Voice to Library", variant="stop", size="lg"
                )

                if whisper_available:
                    gr.HTML("<p><em>✨ Voice will be auto-transcribed when saved!</em></p>")

            with gr.Column(scale=1):
                gr.Markdown("## 🗂️ Manage Voice Library")
                voice_selector = gr.Dropdown(
                    label="Select Voice to Manage",
                    choices=["None"] + voice_library_service.list_voice_library_voices(),
                    value="None",
                )
                voice_info = gr.Markdown("*Select a voice to view details*")

                with gr.Accordion("Edit Voice Parameters", open=False):
                    gr.Markdown(
                        "*Modify generation parameters for the selected voice*"
                    )
                    with gr.Row():
                        with gr.Column():
                            edit_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=config.DEFAULT_TEMPERATURE,
                                step=0.05,
                                label="Temperature",
                            )
                            edit_max_new_tokens = gr.Slider(
                                minimum=128,
                                maximum=2048,
                                value=config.DEFAULT_MAX_NEW_TOKENS,
                                step=128,
                                label="Max New Tokens",
                            )
                            edit_seed = gr.Number(
                                label="Seed",
                                value=config.DEFAULT_SEED,
                                precision=0,
                            )

                        with gr.Column():
                            edit_top_k = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=config.DEFAULT_TOP_K,
                                step=1,
                                label="Top-K",
                            )
                            edit_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=config.DEFAULT_TOP_P,
                                step=0.05,
                                label="Top-P",
                            )
                            edit_min_p = gr.Slider(
                                minimum=0.0,
                                maximum=0.2,
                                value=config.DEFAULT_MIN_P,
                                step=0.01,
                                label="Min-P",
                            )

                    with gr.Row():
                        with gr.Column():
                            edit_repetition_penalty = gr.Slider(
                                minimum=0.8,
                                maximum=1.2,
                                value=config.DEFAULT_REPETITION_PENALTY,
                                step=0.05,
                                label="Repetition Penalty",
                            )
                            edit_do_sample = gr.Checkbox(
                                label="Enable Sampling",
                                value=config.DEFAULT_DO_SAMPLE,
                            )

                        with gr.Column():
                            edit_ras_win_len = gr.Slider(
                                minimum=0,
                                maximum=20,
                                value=config.DEFAULT_RAS_WIN_LEN,
                                step=1,
                                label="RAS Window Length",
                            )
                            edit_ras_win_max_num_repeat = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
                                step=1,
                                label="RAS Max Repeats",
                            )

                    edit_description = gr.Textbox(
                        label="Description",
                        placeholder="Describe this voice...",
                        lines=2,
                    )

                with gr.Row():
                    refresh_button = gr.Button("🔄 Refresh List", variant="secondary")
                    save_changes_button = gr.Button(
                        "💾 Save Changes", variant="primary"
                    )
                    delete_button = gr.Button("🗑️ Delete Voice", variant="stop")

                test_audio = gr.Audio(
                    label="🎧 Voice Test Result",
                    type="filepath",
                    show_download_button=True,
                )
                test_status = gr.Textbox(label="Test Status", interactive=False)
                save_status = gr.Textbox(label="Save Status", interactive=False)
                delete_status = gr.Textbox(
                    label="Management Status", interactive=False
                )

    return VoiceLibraryTabElements(
        new_voice_audio=new_voice_audio,
        new_voice_name=new_voice_name,
        voice_description=voice_description,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        do_sample=do_sample,
        test_text=test_text,
        test_button=test_button,
        clear_test_button=clear_test_button,
        save_button=save_button,
        test_audio=test_audio,
        test_status=test_status,
        save_status=save_status,
        voice_selector=voice_selector,
        voice_info=voice_info,
        edit_temperature=edit_temperature,
        edit_max_new_tokens=edit_max_new_tokens,
        edit_seed=edit_seed,
        edit_top_k=edit_top_k,
        edit_top_p=edit_top_p,
        edit_min_p=edit_min_p,
        edit_repetition_penalty=edit_repetition_penalty,
        edit_ras_win_len=edit_ras_win_len,
        edit_ras_win_max_num_repeat=edit_ras_win_max_num_repeat,
        edit_do_sample=edit_do_sample,
        edit_description=edit_description,
        refresh_button=refresh_button,
        save_changes_button=save_changes_button,
        delete_button=delete_button,
        delete_status=delete_status,
    )


def register_callbacks(
    elements: VoiceLibraryTabElements,
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
) -> None:
    """Attach callbacks for voice library management."""

    def auto_populate_voice_name(audio_data):
        if audio_data is None:
            return ""

        if isinstance(audio_data, str) and audio_data.strip():
            filename = os.path.basename(audio_data)
            name_without_ext, _ = os.path.splitext(filename)
            clean_name = "".join(
                char for char in name_without_ext if char.isalnum() or char in "-_ "
            ).strip()
            clean_name = "_".join(clean_name.split())
            clean_name = "_".join(part for part in clean_name.split("_") if part)
            return clean_name[:50] if clean_name else "uploaded_voice"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"voice_{timestamp}"

    def handle_test_voice_with_params(
        audio_data,
        test_text: str,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        ras_win_len: int,
        ras_win_max_num_repeat: int,
        do_sample: bool,
    ):
        if audio_data is None:
            return None, "❌ Please upload an audio sample first"

        test_text = test_text.strip() or "This is a test of my voice with custom parameters."

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
        except Exception as exc:
            return None, f"❌ Error testing voice: {exc}"

    def handle_clear_test():
        return None, "Test cleared. Upload voice and try again."

    def handle_save_voice_with_config(
        audio_data,
        voice_name: str,
        description: str,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        ras_win_len: int,
        ras_win_max_num_repeat: int,
        do_sample: bool,
    ):
        if audio_data is None:
            return "❌ Please upload an audio sample first"

        if not voice_name or not voice_name.strip():
            return "❌ Please enter a voice name"

        try:
            waveform, sample_rate = torchaudio.load(audio_data)
            audio_array = waveform.numpy()
            if audio_array.ndim > 1:
                audio_array = audio_array[0]

            sanitized_name = voice_name.strip().replace(" ", "_")
            status = voice_library_service.save_voice(
                audio_array, sample_rate, voice_name.strip()
            )

            if not status.startswith("✅"):
                return status

            voice_config = {
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
                "tags": [],
            }

            if voice_library_service.save_voice_config(sanitized_name, voice_config):
                return f"✅ Voice '{sanitized_name}' saved with custom parameters!"

            return "⚠️ Voice saved but failed to save parameters"
        except Exception as exc:
            return f"❌ Error saving voice: {exc}"

    def handle_voice_selection(voice_name: str):
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

        voice_config = voice_library_service.load_voice_config(voice_name)
        selection_value = f"{config.LIBRARY_VOICE_PREFIX}{voice_name}"
        voice_path = voice_library_service.get_voice_path(selection_value)
        transcript_path = (
            voice_library_service.robust_txt_path_creation(voice_path)
            if voice_path
            else None
        )

        info_lines: list[str] = [f"## 🎤 {voice_name}\n\n"]

        description = voice_config.get("description")
        if description:
            info_lines.append(f"**Description:** {description}\n\n")

        if transcript_path and os.path.exists(transcript_path):
            try:
                with open(transcript_path, encoding="utf-8") as handle:
                    transcript = handle.read().strip()
                if len(transcript) > 200:
                    transcript = transcript[:200] + "..."
                info_lines.append(f"**Sample Text:** *{transcript}*\n\n")
            except Exception:
                pass

        info_lines.append("**Current Parameters:**\n")
        info_lines.append(f"- Temperature: {voice_config['temperature']}\n")
        info_lines.append(f"- Max Tokens: {voice_config['max_new_tokens']}\n")
        info_lines.append(
            f"- Top-K: {voice_config['top_k']}, Top-P: {voice_config['top_p']}\n"
        )
        info_lines.append(f"- RAS Window: {voice_config['ras_win_len']}\n")

        return (
            "".join(info_lines),
            voice_config["temperature"],
            voice_config["max_new_tokens"],
            voice_config["seed"],
            voice_config["top_k"],
            voice_config["top_p"],
            voice_config["min_p"],
            voice_config["repetition_penalty"],
            voice_config["ras_win_len"],
            voice_config["ras_win_max_num_repeat"],
            voice_config["do_sample"],
            voice_config.get("description", ""),
        )

    def handle_save_voice_changes(
        voice_name: str,
        description: str,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        ras_win_len: int,
        ras_win_max_num_repeat: int,
        do_sample: bool,
    ):
        if not voice_name or voice_name == "None":
            return "❌ Please select a voice first"

        voice_config = {
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
            "tags": [],
        }

        if voice_library_service.save_voice_config(voice_name, voice_config):
            return f"✅ Updated parameters for '{voice_name}'"

        return f"❌ Failed to save changes for '{voice_name}'"

    def handle_delete_voice(voice_name: str):
        result = voice_library_service.delete_voice(voice_name)
        return result

    def handle_refresh_library():
        updated_choices = ["None"] + voice_library_service.list_voice_library_voices()
        return gr.update(choices=updated_choices, value="None")

    elements.new_voice_audio.upload(
        fn=auto_populate_voice_name,
        inputs=elements.new_voice_audio,
        outputs=elements.new_voice_name,
    )

    elements.test_button.click(
        fn=handle_test_voice_with_params,
        inputs=[
            elements.new_voice_audio,
            elements.test_text,
            elements.temperature,
            elements.max_new_tokens,
            elements.seed,
            elements.top_k,
            elements.top_p,
            elements.min_p,
            elements.repetition_penalty,
            elements.ras_win_len,
            elements.ras_win_max_num_repeat,
            elements.do_sample,
        ],
        outputs=[elements.test_audio, elements.test_status],
    )

    elements.clear_test_button.click(
        fn=handle_clear_test,
        outputs=[elements.test_audio, elements.test_status],
    )

    elements.save_button.click(
        fn=handle_save_voice_with_config,
        inputs=[
            elements.new_voice_audio,
            elements.new_voice_name,
            elements.voice_description,
            elements.temperature,
            elements.max_new_tokens,
            elements.seed,
            elements.top_k,
            elements.top_p,
            elements.min_p,
            elements.repetition_penalty,
            elements.ras_win_len,
            elements.ras_win_max_num_repeat,
            elements.do_sample,
        ],
        outputs=[elements.save_status],
    )

    elements.voice_selector.change(
        fn=handle_voice_selection,
        inputs=elements.voice_selector,
        outputs=[
            elements.voice_info,
            elements.edit_temperature,
            elements.edit_max_new_tokens,
            elements.edit_seed,
            elements.edit_top_k,
            elements.edit_top_p,
            elements.edit_min_p,
            elements.edit_repetition_penalty,
            elements.edit_ras_win_len,
            elements.edit_ras_win_max_num_repeat,
            elements.edit_do_sample,
            elements.edit_description,
        ],
    )

    elements.save_changes_button.click(
        fn=handle_save_voice_changes,
        inputs=[
            elements.voice_selector,
            elements.edit_description,
            elements.edit_temperature,
            elements.edit_max_new_tokens,
            elements.edit_seed,
            elements.edit_top_k,
            elements.edit_top_p,
            elements.edit_min_p,
            elements.edit_repetition_penalty,
            elements.edit_ras_win_len,
            elements.edit_ras_win_max_num_repeat,
            elements.edit_do_sample,
        ],
        outputs=[elements.delete_status],
    )

    elements.delete_button.click(
        fn=handle_delete_voice,
        inputs=elements.voice_selector,
        outputs=[elements.delete_status],
    )

    elements.refresh_button.click(
        fn=handle_refresh_library,
        outputs=[elements.voice_selector],
    )
