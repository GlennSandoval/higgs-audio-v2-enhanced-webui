"""Gradio tab definition for the voice cloning workflow."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from app import config
from app.generation import GenerationService


@dataclass
class VoiceCloningTabElements:
    """Components and IO wiring for the voice cloning tab."""

    inputs: list[gr.components.Component]
    generate_button: gr.Button
    output_audio: gr.Audio


def build_tab(
    *,
    generation_service: GenerationService,
    whisper_available: bool,
) -> VoiceCloningTabElements:
    """Render the voice cloning tab."""

    with gr.Tab("Voice Cloning"):
        with gr.Row():
            with gr.Column():
                transcript = gr.TextArea(
                    label="Transcript",
                    placeholder="Enter text to synthesize with your voice...",
                    value="Hello, this is my cloned voice speaking!",
                    lines=5,
                )

                with gr.Accordion("Voice Cloning", open=True):
                    gr.Markdown("### Upload Your Voice Sample")
                    uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                    if whisper_available:
                        gr.Markdown(
                            "*Record 10-30 seconds of clear speech for best results. "
                            "Audio will be auto-transcribed!* ✨"
                        )
                    else:
                        gr.Markdown(
                            "*Record 10-30 seconds of clear speech for best results. "
                            "Install whisper for auto-transcription: `pip install faster-whisper`*"
                        )

                    method = gr.Radio(
                        choices=["Official Method", "Alternative Method"],
                        value="Official Method",
                        label="Voice Cloning Method",
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
                                    minimum=1,
                                    maximum=5,
                                    value=config.DEFAULT_RAS_WIN_MAX_NUM_REPEAT,
                                    step=1,
                                    label="RAS Max Repeats",
                                    info="Maximum allowed repetitions in window",
                                )

                generate_button = gr.Button(
                    "Clone My Voice & Generate", variant="primary"
                )

            with gr.Column():
                output_audio = gr.Audio(
                    label="Cloned Voice Audio",
                    type="filepath",
                    show_download_button=True,
                )
                gr.HTML(
                    """
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #4caf50;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Voice Cloning:</b><br>
                        • Upload a clear 10-30 second sample of your voice, speaking naturally.<br>
                        • The sample will be auto-transcribed for best cloning results.<br>
                        • Use the "Official Method" for most cases; try "Alternative Method" if you want to experiment.<br>
                        • Longer, more expressive samples improve cloning quality.<br>
                        • Use the same seed to reproduce results.
                    </div>
                    """
                )

    inputs: list[gr.components.Component] = [
        transcript,
        uploaded_voice,
        temperature,
        max_new_tokens,
        seed,
        method,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        ras_win_len,
        ras_win_max_num_repeat,
        do_sample,
    ]

    return VoiceCloningTabElements(
        inputs=inputs,
        generate_button=generate_button,
        output_audio=output_audio,
    )


def register_callbacks(
    elements: VoiceCloningTabElements,
    *,
    generation_service: GenerationService,
) -> None:
    """Attach callback handlers for the voice cloning tab."""

    def _handle_voice_clone_generation(
        transcript: str,
        uploaded_voice,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        method: str,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        ras_win_len: int,
        ras_win_max_num_repeat: int,
        do_sample: bool,
    ):
        if method == "Official Method":
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

    elements.generate_button.click(
        fn=_handle_voice_clone_generation,
        inputs=elements.inputs,
        outputs=elements.output_audio,
    )
