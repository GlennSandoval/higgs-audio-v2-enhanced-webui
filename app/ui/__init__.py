"""UI composition utilities for the Higgs Audio Gradio application."""

from __future__ import annotations

import gradio as gr

from app.audio import describe_missing_dependencies
from app.services import GenerationService, VoiceLibrary
from app.ui import basic, longform, multi_speaker, voice_cloning
from app.ui import voice_library as library_tab


def build_demo(
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
    whisper_available: bool,
    dependency_report: dict[str, bool],
) -> gr.Blocks:
    """Construct the Gradio Blocks demo using modular tab builders."""

    with gr.Blocks(title="Higgs Audio v2 Generator") as demo:
        gr.HTML(
            '<h1 style="text-align:center; margin-bottom:0.2em;"><a href="https://github.com/Saganaki22/higgs-audio-WebUI" target="_blank" style="text-decoration:none; color:inherit;">🎵 Higgs Audio v2 WebUI</a></h1>'
        )
        gr.HTML(
            '<div style="text-align:center; font-size:1.2em; margin-bottom:1.5em;">Generate high-quality speech from text with voice cloning, longform generation, multi speaker generation, voice library, smart batching</div>'
        )

        environment_notices: list[str] = []
        if not whisper_available:
            environment_notices.append(
                "Whisper auto-transcription is disabled. Install `faster-whisper` (or `openai-whisper`) to enable automatic transcripts."
            )

        environment_notices.extend(describe_missing_dependencies(dependency_report))

        if environment_notices:
            notices_html = "".join(f"<li>{notice}</li>" for notice in environment_notices)
            gr.HTML(
                "<div style='background:#2f3136;border-radius:8px;padding:0.85em 1.2em;margin-bottom:1.25em;border-left:5px solid #ffb347;'>"
                "<strong>⚠️ Environment Notices</strong><ul style='margin:0.5em 0 0.25em 1.25em;'>"
                f"{notices_html}</ul></div>"
            )

        with gr.Tabs():
            basic_elements = basic.build_tab(
                generation_service=generation_service,
                voice_library_service=voice_library_service,
            )
            voice_cloning_elements = voice_cloning.build_tab(
                generation_service=generation_service,
                whisper_available=whisper_available,
            )
            longform_elements = longform.build_tab(
                generation_service=generation_service,
                voice_library_service=voice_library_service,
                whisper_available=whisper_available,
            )
            multi_speaker_elements = multi_speaker.build_tab(
                generation_service=generation_service,
                voice_library_service=voice_library_service,
            )
            voice_library_elements = library_tab.build_tab(
                generation_service=generation_service,
                voice_library_service=voice_library_service,
                whisper_available=whisper_available,
            )

        gr.HTML(
            """
            <div style='width:100%;text-align:center;margin-top:2em;margin-bottom:1em;'>
                <a href='https://github.com/Saganaki22/higgs-audio-WebUI' target='_blank' style='color:#fff;font-size:1.1em;text-decoration:underline;'>Github</a>
            </div>
            """
        )

        basic.register_callbacks(
            basic_elements,
            generation_service=generation_service,
            voice_library_service=voice_library_service,
        )
        voice_cloning.register_callbacks(
            voice_cloning_elements,
            generation_service=generation_service,
        )
        longform.register_callbacks(
            longform_elements,
            generation_service=generation_service,
            voice_library_service=voice_library_service,
        )
        multi_speaker.register_callbacks(
            multi_speaker_elements,
            generation_service=generation_service,
            voice_library_service=voice_library_service,
        )
        library_tab.register_callbacks(
            voice_library_elements,
            generation_service=generation_service,
            voice_library_service=voice_library_service,
        )

    return demo
