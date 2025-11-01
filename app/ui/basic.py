"""Gradio tab definition for basic audio generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

from app import config
from app.services import GenerationService, VoiceLibrary


@dataclass
class BasicTabElements:
    """Gradio components required for the basic generation tab."""

    inputs: list[gr.components.Component]
    voice_prompt: gr.Dropdown
    refresh_button: gr.Button
    generate_button: gr.Button
    output_audio: gr.Audio


def build_tab(
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
) -> BasicTabElements:
    """Render the basic generation tab and return the relevant components."""

    def _current_voice_choices() -> list[str]:
        return voice_library_service.list_all_available_voices()

    with gr.Tab("Basic Generation"):
        with gr.Row():
            with gr.Column():
                transcript = gr.TextArea(
                    label="Transcript",
                    placeholder="Enter text to synthesize...",
                    value="The sun rises in the east and sets in the west.",
                    lines=5,
                )

                with gr.Accordion("Voice Settings", open=True):
                    voice_prompt = gr.Dropdown(
                        choices=_current_voice_choices(),
                        value=config.SMART_VOICE_LABEL,
                        label="Predefined Voice Prompts",
                    )
                    refresh_voices = gr.Button("Refresh Voice List")

                    scene_description = gr.TextArea(
                        label="Scene Description",
                        placeholder="Describe the recording environment...",
                        value="",
                    )

                with gr.Accordion("Generation Parameters", open=False):
                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=config.DEFAULT_TEMPERATURE,
                                step=0.05,
                                label="Temperature",
                                info="Controls randomness in generation (lower = more consistent)",
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
                                info="Limits vocabulary to top K most likely tokens",
                            )
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=config.DEFAULT_TOP_P,
                                step=0.05,
                                label="Top-P (Nucleus Sampling)",
                                info="Cumulative probability threshold for token selection",
                            )
                            min_p = gr.Slider(
                                minimum=0.0,
                                maximum=0.2,
                                value=config.DEFAULT_MIN_P,
                                step=0.01,
                                label="Min-P",
                                info="Minimum probability threshold (0 = disabled)",
                            )

                    with gr.Accordion("Advanced Sampling", open=False):
                        with gr.Row():
                            with gr.Column():
                                repetition_penalty = gr.Slider(
                                    minimum=0.8,
                                    maximum=1.2,
                                    value=config.DEFAULT_REPETITION_PENALTY,
                                    step=0.05,
                                    label="Repetition Penalty",
                                    info="Penalty for repeating tokens (1.0 = no penalty)",
                                )
                                do_sample = gr.Checkbox(
                                    label="Enable Sampling",
                                    value=config.DEFAULT_DO_SAMPLE,
                                    info="Use sampling vs greedy decoding",
                                )

                            with gr.Column():
                                ras_win_len = gr.Slider(
                                    minimum=0,
                                    maximum=20,
                                    value=config.DEFAULT_RAS_WIN_LEN,
                                    step=1,
                                    label="RAS Window Length",
                                    info="Repetition detection window (0 = disabled)",
                                )
                                ras_win_max_num_repeat = gr.Slider(
                                    minimum=0,
                                    maximum=5,
                                    value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
                                    step=1,
                                    label="RAS Max Repeats",
                                    info="Maximum repeats allowed in RAS window",
                                )

                with gr.Accordion("🔊 Volume Normalization", open=False):
                    gr.Markdown(
                        "*Optional: Normalize audio volume for consistent playback*"
                    )
                    with gr.Row():
                        enable_normalization = gr.Checkbox(
                            label="Enable Volume Normalization",
                            value=False,
                            info="Normalize audio volume level",
                        )
                        target_volume = gr.Slider(
                            0.05,
                            0.3,
                            value=0.15,
                            step=0.01,
                            label="Target Volume",
                            info="RMS level (0.15 = moderate)",
                        )

                generate_button = gr.Button("Generate Audio", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    show_download_button=True,
                )
                gr.HTML(
                    """
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #ff9800;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Basic Generation:</b><br>
                        • For best results, use clear, natural sentences.<br>
                        • You can select a predefined voice or use Smart Voice for random high-quality voices.<br>
                        • Scene description can help set the environment (e.g., "in a quiet room").<br>
                        • Adjust temperature for more/less expressive speech.<br>
                        • Try different seeds for voice variety.
                    </div>
                    """
                )

    inputs: list[gr.components.Component] = [
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
    ]

    return BasicTabElements(
        inputs=inputs,
        voice_prompt=voice_prompt,
        refresh_button=refresh_voices,
        generate_button=generate_button,
        output_audio=output_audio,
    )


def register_callbacks(
    elements: BasicTabElements,
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
) -> None:
    """Wire Gradio callbacks for the basic tab."""

    def _refresh_voice_list() -> dict[str, Any]:
        return gr.update(choices=voice_library_service.list_all_available_voices())

    elements.generate_button.click(
        fn=generation_service.generate_basic,
        inputs=elements.inputs,
        outputs=elements.output_audio,
    )

    elements.refresh_button.click(
        fn=_refresh_voice_list,
        outputs=elements.voice_prompt,
    )
