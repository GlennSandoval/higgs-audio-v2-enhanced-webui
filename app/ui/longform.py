"""Gradio tab definition for long-form audio generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

from app import config
from app.generation import GenerationService
from app.voice_library import VoiceLibrary


@dataclass
class LongformTabElements:
    """Components needed to wire the long-form generation tab."""

    inputs: list[gr.components.Component]
    voice_prompt: gr.Dropdown
    refresh_button: gr.Button
    generate_button: gr.Button
    output_audio: gr.Audio
    voice_choice: gr.Radio
    upload_group: gr.Group
    predefined_group: gr.Group


def build_tab(
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
    whisper_available: bool,
) -> LongformTabElements:
    """Render the long-form generation tab."""

    def _current_voice_choices() -> list[str]:
        return voice_library_service.list_all_available_voices()

    with gr.Tab("Long-form Generation"):
        with gr.Row():
            with gr.Column():
                transcript = gr.TextArea(
                    label="Long Transcript",
                    placeholder="Enter long text to synthesize...",
                    value=(
                        "Artificial intelligence is transforming our world. It helps solve "
                        "complex problems in healthcare, climate, and education. Machine "
                        "learning algorithms can process vast amounts of data to find patterns "
                        "humans might miss. As we develop these technologies, we must consider "
                        "their ethical implications. The future of AI holds both incredible "
                        "promise and significant challenges."
                    ),
                    lines=10,
                )

                with gr.Accordion("Voice Options", open=True):
                    voice_choice = gr.Radio(
                        choices=["Smart Voice", "Upload Voice", "Predefined Voice"],
                        value="Smart Voice",
                        label="Voice Selection Method",
                    )

                    with gr.Group(visible=False) as upload_group:
                        uploaded_voice = gr.Audio(
                            label="Upload Voice Sample",
                            type="numpy",
                        )
                        if whisper_available:
                            gr.Markdown(
                                "*Audio will be auto-transcribed for voice cloning!* ✨"
                            )
                        else:
                            gr.Markdown(
                                "*Install whisper for auto-transcription: "
                                "`pip install faster-whisper`*"
                            )

                    with gr.Group(visible=False) as predefined_group:
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
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=config.DEFAULT_TEMPERATURE,
                        step=0.05,
                        label="Temperature",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=128,
                        maximum=2048,
                        value=config.DEFAULT_MAX_NEW_TOKENS,
                        step=128,
                        label="Max New Tokens per Chunk",
                    )
                    seed = gr.Number(
                        label="Seed (0 for random)",
                        value=config.DEFAULT_SEED,
                        precision=0,
                    )
                    chunk_size = gr.Slider(
                        minimum=100,
                        maximum=500,
                        value=200,
                        step=50,
                        label="Characters per Chunk",
                    )

                generate_button = gr.Button(
                    "Generate Long-form Audio", variant="primary"
                )

            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    show_download_button=True,
                )
                gr.HTML(
                    """
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #2196f3;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Long-form Generation:</b><br>
                        • Paste or write long text (stories, articles, etc.) for continuous speech.<br>
                        • Choose Smart Voice, upload your own, or select a predefined voice.<br>
                        • Adjust chunk size for smoother transitions (smaller = more natural, larger = faster).<br>
                        • Scene description can set the mood or environment.<br>
                        • Use consistent voice references for best results in long texts.
                    </div>
                    """
                )

    inputs: list[gr.components.Component] = [
        transcript,
        voice_choice,
        uploaded_voice,
        voice_prompt,
        temperature,
        max_new_tokens,
        seed,
        scene_description,
        chunk_size,
    ]

    return LongformTabElements(
        inputs=inputs,
        voice_prompt=voice_prompt,
        refresh_button=refresh_voices,
        generate_button=generate_button,
        output_audio=output_audio,
        voice_choice=voice_choice,
        upload_group=upload_group,
        predefined_group=predefined_group,
    )


def register_callbacks(
    elements: LongformTabElements,
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
) -> None:
    """Attach callbacks for the long-form generation tab."""

    def _refresh_voice_list() -> dict[str, Any]:
        return gr.update(choices=voice_library_service.list_all_available_voices())

    def _update_voice_options(choice: str):
        return {
            elements.upload_group: gr.update(visible=choice == "Upload Voice"),
            elements.predefined_group: gr.update(visible=choice == "Predefined Voice"),
        }

    elements.generate_button.click(
        fn=generation_service.generate_longform,
        inputs=elements.inputs,
        outputs=elements.output_audio,
    )

    elements.refresh_button.click(
        fn=_refresh_voice_list,
        outputs=elements.voice_prompt,
    )

    elements.voice_choice.change(
        fn=_update_voice_options,
        inputs=elements.voice_choice,
        outputs=[elements.upload_group, elements.predefined_group],
    )
