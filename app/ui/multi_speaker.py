"""Gradio tab definition for dynamic multi-speaker generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gradio as gr

from app import config
from app.generation import GenerationService
from app.voice_library import VoiceLibrary


@dataclass
class MultiSpeakerTabElements:
    """Components and state for the multi-speaker tab."""

    transcript: gr.TextArea
    process_button: gr.Button
    speaker_info: gr.Markdown
    detected_speakers_state: gr.State
    speaker_mapping_state: gr.State
    generate_button: gr.Button
    voice_method: gr.Radio
    smart_voice_group: gr.Group
    upload_group: gr.Group
    library_group: gr.Group
    speaker_assignment: gr.Column
    assignment_content: gr.Markdown
    upload_slots: list[tuple[gr.Row, gr.Audio]]
    library_slots: list[gr.Dropdown]
    refresh_library_button: gr.Button
    scene_description: gr.TextArea
    temperature: gr.Slider
    max_new_tokens: gr.Slider
    seed: gr.Number
    enable_normalization: gr.Checkbox
    normalization_method: gr.Dropdown
    target_volume: gr.Slider
    speaker_pause: gr.Slider
    output_audio: gr.Audio
    generation_inputs: list[gr.components.Component]


def build_tab(
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
) -> MultiSpeakerTabElements:
    """Render the multi-speaker tab."""

    def _current_voice_choices() -> list[str]:
        return voice_library_service.list_all_available_voices()

    with gr.Tab("Multi-Speaker Generation"):
        with gr.Row():
            with gr.Column():
                transcript = gr.TextArea(
                    label="Multi-Speaker Transcript",
                    placeholder=(
                        "Enter dialogue with any speaker names in brackets:\n"
                        "[Alice] Hello there!\n[Bob] How are you?\n[Charlie] Great to see you both!"
                    ),
                    value=(
                        "[Alice] Hello there, how are you doing today?\n"
                        "[Bob] I'm doing great, thank you for asking! How about yourself?\n"
                        "[Alice] I'm fantastic! It's such a beautiful day outside.\n"
                        "[Bob] Yes, it really is. Perfect weather for a walk in the park."
                    ),
                    lines=8,
                )

                process_button = gr.Button(
                    "🔍 Process Text & Detect Speakers",
                    variant="secondary",
                    size="lg",
                )

                speaker_info = gr.Markdown(
                    "*Click 'Process Text' to detect speakers in your dialogue*"
                )

                with gr.Accordion("Voice Configuration", open=True):
                    voice_method = gr.Radio(
                        choices=["Smart Voice", "Upload Voices", "Voice Library"],
                        value="Smart Voice",
                        label="Voice Method",
                    )

                    with gr.Column(visible=False) as speaker_assignment:
                        gr.Markdown("### 🎭 Speaker Voice Assignment")
                        assignment_content = gr.Markdown(
                            "*Speaker assignments will appear here*"
                        )

                    with gr.Group() as smart_voice_group:
                        gr.Markdown("### Smart Voice Mode")
                        gr.Markdown(
                            "*AI will automatically assign distinct voices to each detected speaker*"
                        )

                    with gr.Group(visible=False) as upload_group:
                        gr.Markdown("### Upload Voice Samples")
                        gr.Markdown(
                            "*Upload a voice sample for each speaker. Files will be auto-transcribed.*"
                        )
                        upload_slots: list[tuple[gr.Row, gr.Audio]] = []
                        for index in range(10):
                            with gr.Row(visible=False) as upload_row:
                                speaker_audio = gr.Audio(
                                    label=f"Speaker {index} Voice Sample",
                                    type="numpy",
                                    scale=3,
                                )
                                upload_slots.append((upload_row, speaker_audio))

                    with gr.Group(visible=False) as library_group:
                        gr.Markdown("### Select Voices from Library")
                        gr.Markdown("*Choose a saved voice for each speaker*")
                        library_slots: list[gr.Dropdown] = []
                        for index in range(10):
                            speaker_dropdown = gr.Dropdown(
                                choices=_current_voice_choices(),
                                value=config.SMART_VOICE_LABEL,
                                label=f"Speaker {index} Voice",
                                visible=False,
                            )
                            library_slots.append(speaker_dropdown)

                        refresh_library_button = gr.Button(
                            "Refresh Voice Library", visible=False
                        )

                    scene_description = gr.TextArea(
                        label="Scene Description",
                        placeholder="Describe the conversation setting...",
                        value="A friendly conversation between people in a quiet room.",
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
                        label="Max New Tokens per Segment",
                    )
                    seed = gr.Number(
                        label="Seed (0 for random)",
                        value=config.DEFAULT_SEED,
                        precision=0,
                    )

                with gr.Accordion("🔊 Volume Normalization", open=True):
                    gr.Markdown(
                        "*Fix volume inconsistencies between speakers - recommended for all multi-speaker audio*"
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            enable_normalization = gr.Checkbox(
                                label="Enable Volume Normalization",
                                value=True,
                                info="Automatically balance speaker volumes",
                            )
                        with gr.Column(scale=3):
                            normalization_method = gr.Dropdown(
                                choices=["adaptive", "simple", "segment-based"],
                                value="adaptive",
                                label="Normalization Method",
                                info=(
                                    "Adaptive = sliding windows, Simple = whole audio, "
                                    "Segment = detect speakers"
                                ),
                            )
                        with gr.Column(scale=2):
                            target_volume = gr.Slider(
                                0.05,
                                0.3,
                                value=config.DEFAULT_TARGET_VOLUME,
                                step=0.01,
                                label="Target Volume",
                                info="RMS level (0.15 = moderate)",
                            )

                with gr.Accordion("⏸️ Speaker Timing", open=False):
                    gr.Markdown(
                        "*Control timing and pauses between different speakers*"
                    )
                    speaker_pause = gr.Slider(
                        0.0,
                        2.0,
                        value=config.DEFAULT_SPEAKER_PAUSE_SECONDS,
                        step=0.1,
                        label="Pause Between Speakers (seconds)",
                        info=(
                            "Duration of silence when speakers change "
                            "(0.0 = no pause, 0.3 = default)"
                        ),
                    )

                generate_button = gr.Button(
                    "Generate Multi-Speaker Audio",
                    variant="primary",
                    interactive=False,
                )

            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Multi-Speaker Audio",
                    type="filepath",
                    show_download_button=True,
                )
                gr.HTML(
                    """
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
                    """
                )

        detected_speakers_state = gr.State([])
        speaker_mapping_state = gr.State({})

    generation_inputs: list[gr.components.Component] = [
        transcript,
        voice_method,
        speaker_mapping_state,
        temperature,
        max_new_tokens,
        seed,
        scene_description,
        enable_normalization,
        normalization_method,
        target_volume,
        speaker_pause,
    ]

    for _, audio_component in upload_slots:
        generation_inputs.append(audio_component)
    generation_inputs.extend(library_slots)

    return MultiSpeakerTabElements(
        transcript=transcript,
        process_button=process_button,
        speaker_info=speaker_info,
        detected_speakers_state=detected_speakers_state,
        speaker_mapping_state=speaker_mapping_state,
        generate_button=generate_button,
        voice_method=voice_method,
        smart_voice_group=smart_voice_group,
        upload_group=upload_group,
        library_group=library_group,
        speaker_assignment=speaker_assignment,
        assignment_content=assignment_content,
        upload_slots=upload_slots,
        library_slots=library_slots,
        refresh_library_button=refresh_library_button,
        scene_description=scene_description,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        enable_normalization=enable_normalization,
        normalization_method=normalization_method,
        target_volume=target_volume,
        speaker_pause=speaker_pause,
        output_audio=output_audio,
        generation_inputs=generation_inputs,
    )


def register_callbacks(
    elements: MultiSpeakerTabElements,
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
) -> None:
    """Wire the callbacks for the multi-speaker tab."""

    def detect_dynamic_speakers(text: str) -> list[str]:
        """Detect speaker names in bracketed dialogue."""
        if not text or not text.strip():
            return []

        pattern = r"^\s*\[([^\]]+)\]\s*[:.]?\s*(.+?)(?=^\s*\[[^\]]+\]|$)"
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)

        speakers: list[str] = []
        seen = set()
        for speaker_name, _ in matches:
            clean_name = speaker_name.strip()
            if clean_name and clean_name not in seen:
                speakers.append(clean_name)
                seen.add(clean_name)

        return speakers

    def create_speaker_assignment_interface(
        speakers: Sequence[str], voice_method: str
    ) -> Any:
        if not speakers:
            return gr.update()

        if voice_method == "Smart Voice":
            content = ""
            for speaker in speakers:
                content += (
                    f"**{speaker}**: AI will automatically assign a distinct voice\n\n"
                )
            content += (
                "*No manual assignment needed - the AI will ensure each speaker has a unique voice*"
            )
            return gr.update(value=content)

        if voice_method == "Voice Library":
            available = voice_library_service.list_all_available_voices()
            content = "**🎭 Voice Library Assignment:**\n\n"
            content += "*Use the dropdowns below to assign specific voices from your library to each character*\n\n"
            for speaker in speakers:
                content += f"**{speaker}** → Use dropdown to select voice\n"
            content += f"\n**Available voices:** {', '.join(available)}\n\n"
            content += (
                f"*💡 Select '{config.SMART_VOICE_LABEL}' to let AI pick a voice for that character*\n"
            )
            content += "*💡 Click 'Refresh Voice Library' if you've added new voices*"
            return gr.update(value=content)

        if voice_method == "Upload Voices":
            content = "**Upload Voice Samples:**\n\n"
            for speaker in speakers:
                content += (
                    f"**{speaker}**: Upload a voice sample to clone this speaker's voice\n\n"
                )
            content += (
                "*💡 Upload functionality coming soon! For now, use Smart Voice mode.*"
            )
            return gr.update(value=content)

        return gr.update()

    def update_upload_slots(speakers: Sequence[str]) -> list[Any]:
        updates: list[Any] = []
        for index in range(10):
            visible = index < len(speakers)
            updates.append(gr.update(visible=visible))

        for index in range(10):
            if index < len(speakers):
                label = f"{speakers[index]} Voice Sample"
            else:
                label = f"Speaker {index} Voice Sample"
            updates.append(gr.update(label=label, visible=index < len(speakers)))
        return updates

    def update_library_slots(speakers: Sequence[str]) -> list[Any]:
        updates: list[Any] = []
        for index in range(10):
            if index < len(speakers):
                updates.append(
                    gr.update(
                        visible=True,
                        label=f"{speakers[index]} Voice",
                        value=config.SMART_VOICE_LABEL,
                    )
                )
            else:
                updates.append(gr.update(visible=False))
        return updates

    def process_multi_speaker_text(text: str):
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
                "*No text to process*",
            )

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
                "*No speakers detected*",
            )

        speaker_mapping: dict[str, str] = {}
        for index, speaker in enumerate(speakers):
            speaker_mapping[speaker] = f"SPEAKER{index}"

        info_lines = [
            f"**🎭 Detected {len(speakers)} speakers:**\n",
        ]
        for index, speaker in enumerate(speakers):
            info_lines.append(f"• **{speaker}** → SPEAKER{index}\n")
        info_lines.append(
            "\n*Now select voice assignment method below and assign voices to each speaker*"
        )
        info_text = "".join(info_lines)

        return (
            info_text,
            speakers,
            speaker_mapping,
            gr.update(interactive=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            create_speaker_assignment_interface(speakers, "Smart Voice"),
        )

    def update_voice_method_visibility(
        voice_method: str, detected_speakers: Sequence[str]
    ) -> list[Any]:
        upload_updates = update_upload_slots(
            detected_speakers if voice_method == "Upload Voices" else []
        )
        library_updates = update_library_slots(
            detected_speakers if voice_method == "Voice Library" else []
        )

        outputs: list[Any] = [
            gr.update(visible=voice_method == "Smart Voice"),
            gr.update(visible=voice_method == "Upload Voices"),
            gr.update(visible=voice_method == "Voice Library"),
            create_speaker_assignment_interface(detected_speakers, voice_method),
        ]

        outputs.extend(upload_updates)
        outputs.extend(library_updates)
        outputs.append(
            gr.update(
                visible=voice_method == "Voice Library"
                and len(detected_speakers) > 0
            )
        )
        return outputs

    def generate_dynamic_multi_speaker(
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

    def refresh_multi_speaker_voices():
        choices = voice_library_service.list_all_available_voices()
        return [gr.update(choices=choices) for _ in elements.library_slots]

    elements.process_button.click(
        fn=process_multi_speaker_text,
        inputs=elements.transcript,
        outputs=[
            elements.speaker_info,
            elements.detected_speakers_state,
            elements.speaker_mapping_state,
            elements.generate_button,
            elements.upload_group,
            elements.library_group,
            elements.smart_voice_group,
            elements.speaker_assignment,
            elements.assignment_content,
        ],
    )

    voice_method_outputs: list[gr.components.Component] = [
        elements.smart_voice_group,
        elements.upload_group,
        elements.library_group,
        elements.assignment_content,
    ]
    for row, audio in elements.upload_slots:
        voice_method_outputs.append(row)
    for row, audio in elements.upload_slots:
        voice_method_outputs.append(audio)
    voice_method_outputs.extend(elements.library_slots)
    voice_method_outputs.append(elements.refresh_library_button)

    elements.voice_method.change(
        fn=update_voice_method_visibility,
        inputs=[elements.voice_method, elements.detected_speakers_state],
        outputs=voice_method_outputs,
    )

    elements.generate_button.click(
        fn=generate_dynamic_multi_speaker,
        inputs=elements.generation_inputs,
        outputs=[elements.output_audio],
    )

    elements.refresh_library_button.click(
        fn=refresh_multi_speaker_voices,
        outputs=elements.library_slots,
    )
