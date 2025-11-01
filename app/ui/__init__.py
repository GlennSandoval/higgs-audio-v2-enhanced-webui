"""UI composition utilities for the Higgs Audio Gradio application."""

from __future__ import annotations

import gradio as gr

from app.generation import GenerationService
from app.voice_library import VoiceLibrary
from app.ui import basic, longform, multi_speaker, voice_cloning, voice_library as library_tab


def build_demo(
    *,
    generation_service: GenerationService,
    voice_library_service: VoiceLibrary,
    whisper_available: bool,
) -> gr.Blocks:
    """Construct the Gradio Blocks demo using modular tab builders."""

    with gr.Blocks(title="Higgs Audio v2 Generator") as demo:
        gr.HTML(
            '<h1 style="text-align:center; margin-bottom:0.2em;"><a href="https://github.com/Saganaki22/higgs-audio-WebUI" target="_blank" style="text-decoration:none; color:inherit;">🎵 Higgs Audio v2 WebUI</a></h1>'
        )
        gr.HTML(
            '<div style="text-align:center; font-size:1.2em; margin-bottom:1.5em;">Generate high-quality speech from text with voice cloning, longform generation, multi speaker generation, voice library, smart batching</div>'
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
