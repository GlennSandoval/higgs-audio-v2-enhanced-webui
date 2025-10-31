# 🎵 Higgs Audio v2 Enhanced WebUI

An advanced, feature-rich web interface for the Higgs Audio v2 model with professional audio generation capabilities, multi-speaker support, volume normalization, and extensive customization options.

![Higgs Audio WebUI](figures/higgs_audio_v2_architecture_combined.png)

## ✨ Key Features

### 🎭 **Multi-Speaker Generation**
- **Dynamic Speaker Detection** - Use any character names in brackets `[Alice]`, `[Bob]`, `[Character Name]`
- **Unlimited Speakers** - No more 3-speaker limit, support for 10+ characters
- **Smart Voice Assignment** - Assign different voices to each character
- **Voice Library Integration** - Use saved voices for consistent character voices
- **Upload Voice Samples** - Upload custom voice samples for each speaker
- **Configurable Pauses** - Control timing between speaker changes (0.0-2.0 seconds)

### 🔊 **Volume Normalization**
- **Multi-Speaker Balance** - Automatic volume normalization for consistent speaker levels
- **Adaptive Normalization** - Sliding window approach for dynamic content
- **Simple Normalization** - Basic RMS-based volume leveling
- **Segment-Based** - Detect and normalize individual speaker segments
- **Configurable Target Levels** - Set desired volume levels (RMS 0.05-0.3)

### 🎛️ **Advanced Generation Parameters**
- **Exposed Hidden Parameters** - Access to `top_k`, `top_p`, `min_p`, `repetition_penalty`
- **Repetition Aware Sampling (RAS)** - `ras_win_len`, `ras_win_max_num_repeat`
- **Per-Voice Settings** - Save custom parameters for each voice in your library
- **Smart Defaults** - Optimized settings for different use cases

### 📚 **Enhanced Voice Library**
- **Per-Voice Configuration** - Each voice saves its own generation parameters
- **Auto-Populate Names** - Extract voice names from uploaded filenames
- **Voice Testing** - Test voices with custom parameters before saving
- **Organized Management** - Easy voice selection and editing
- **JSON Configuration** - Robust parameter storage and retrieval

### 🌐 **Public Sharing & Deployment**
- **Hugging Face Share** - Create public links for remote access
- **Local Network Sharing** - Share on your local network
- **Multiple Launch Options** - Cross-platform CLI flags and helper scripts for common scenarios
- **Security Controls** - Warnings and confirmations for public access

### 🚀 **Performance & Optimization**
- **Smart Caching** - Intelligent model and audio caching
- **Memory Management** - Automatic cleanup and resource optimization
- **GPU Acceleration** - Full CUDA support for fast generation
- **Cache Migration** - Tools to migrate existing model caches

## 🛠️ Installation & Setup

git clone https://github.com/psdwizzard/higgs-audio-v2-enhanced-webui.git
### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/psdwizzard/higgs-audio-v2-enhanced-webui.git
cd higgs-audio-v2-enhanced-webui

# 2. Install dependencies (creates .venv via uv)
uv sync

# 3. Sanity-check the installation
uv run python verify_setup.py

# 4. Launch the interface
uv run python higgs_audio_gradio.py
```

**macOS tip:** If you installed FFmpeg through MacPorts, run `./run.sh` instead of the last command so the script can expose the MacPorts library path before launching Gradio.

**Windows note:** `uv` works on Windows too. Run the same commands from PowerShell. If you prefer the built-in `venv`, create and activate it (`python -m venv .venv`, `.\.venv\Scripts\Activate`), install dependencies with `pip install -e .`, then run `python verify_setup.py` and `python higgs_audio_gradio.py`.

### For Public Sharing
```bash
# Simple public sharing
uv run python higgs_audio_gradio.py --share

# Bind to a specific host/port (e.g., LAN sharing)
uv run python higgs_audio_gradio.py --server-name 0.0.0.0 --server-port 7860
```

### Reuse Existing Huggingface Cache
- Point `HF_HOME`, `HF_HUB_CACHE`, or `TRANSFORMERS_CACHE` to a directory that already contains the model weights.
- For example, set `export HF_HOME=/path/to/cache` (macOS/Linux) or `set HF_HOME=C:\path\to\cache` (Windows) before running the app.

## 📖 Detailed Documentation

### Multi-Speaker Generation

Create natural conversations with multiple characters:

```
[Alice] Hello there, how are you doing today?
[Bob] I'm doing great, thank you for asking! How about yourself?
[Charlie] Mind if I join this conversation?
[Alice] Of course! The more the merrier.
```

**Features:**
- **Any character names** - Use meaningful names instead of SPEAKER0/1/2
- **Voice assignment** - Choose voices from library or upload samples
- **Natural timing** - Configurable pauses between speakers
- **Volume balance** - Automatic normalization for consistent levels

### Voice Library Management

**Adding New Voices:**
1. Upload audio sample (any format)
2. Voice name auto-populates from filename
3. Set custom generation parameters
4. Test with different settings
5. Save to library

**Using Saved Voices:**
- Select voices for basic generation
- Assign to specific characters in multi-speaker
- Edit parameters anytime
- Consistent voice characteristics

### Volume Normalization

**Methods:**
- **Adaptive** ⭐ - Best for multi-speaker, uses sliding windows
- **Simple** - Basic RMS normalization for single speaker
- **Segment-Based** - Detects and normalizes speaker segments individually

**Benefits:**
- No more volume imbalances
- Professional audio quality
- Podcast-ready output
- Consistent listening experience

## 🎯 Use Cases

### 📖 **Audiobooks & Narration**
- Multiple character voices
- Consistent volume levels
- Professional pacing
- Chapter-by-chapter generation

### 🎙️ **Podcasts & Interviews**
- Natural conversation flow
- Balanced speaker levels
- Background ambience support
- Easy editing workflow

### 🎭 **Drama & Entertainment**
- Character-specific voices
- Dramatic pauses
- Emotional range
- Scene descriptions

### 📚 **Educational Content**
- Clear narration
- Multiple presenter voices
- Consistent quality
- Accessible audio

## 🔧 Advanced Configuration

### Command Line Options
```bash
python higgs_audio_gradio.py --help

Options:
  --share              Create public shareable link
  --server-name HOST   Server host address (default: 127.0.0.1)
  --server-port PORT   Server port (default: 7860)
```

### Environment Variables
```bash
# Cache configuration (macOS/Linux)
export HF_HOME=./cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

# Cache configuration (Windows PowerShell)
$env:HF_HOME="./cache/huggingface"
$env:HF_HUB_CACHE="$env:HF_HOME/hub"
$env:TRANSFORMERS_CACHE=$env:HF_HUB_CACHE
```

### Generation Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| Temperature | Creativity vs consistency | 0.1-2.0 | 1.0 |
| Top-K | Token selection limit | 1-100 | 50 |
| Top-P | Nucleus sampling threshold | 0.1-1.0 | 0.95 |
| Min-P | Minimum probability threshold | 0.0-0.5 | 0.0 |
| Repetition Penalty | Reduce repetitions | 0.5-2.0 | 1.0 |
| RAS Window Length | Repetition detection window | 0-20 | 7 |
| RAS Max Repeats | Maximum allowed repetitions | 1-5 | 2 |

## 📁 Project Structure

```
higgs-audio-v2-enhanced-webui/
├── higgs_audio_gradio.py          # Main application
├── audio_processing_utils.py      # Volume normalization module
├── run.sh                        # Mac launch helper (sets FFmpeg paths, runs uv)
├── voice_library/                # Saved voices directory
├── output/                       # Generated audio output
├── cache/                        # Model cache directory
└── README.md                     # This file
```

## 🚀 Performance Tips

### For Better Generation:
- Use **GPU** if available (CUDA support)
- Enable **caching** for faster repeated generations
- Use **appropriate parameters** for your content type
- **Reuse existing caches** by pointing `HF_HOME` to shared storage

### For Public Sharing:
- Prefer `--share` only when you trust the network you are exposing
- Use `--server-name 0.0.0.0` to make the UI visible on your LAN
- Monitor **resource usage** when sharing publicly
- Set **reasonable limits** on generation length
- Consider **authentication** for production use

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional normalization algorithms
- More voice management features
- UI/UX enhancements
- Performance optimizations
- Documentation improvements

## 📄 License

This project is based on the original Higgs Audio v2 model. Please see the original license terms.

## 🙏 Acknowledgments

- **Boson AI** - Original Higgs Audio v2 model
- **Hugging Face** - Model hosting and sharing infrastructure
- **Gradio** - Web interface framework
- **Community Contributors** - Testing and feedback

## 🔗 Links

- [Original Higgs Audio v2](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base)
- [Gradio Documentation](https://gradio.app/docs/)
- [Issues & Bug Reports](https://github.com/psdwizzard/higgs-audio-v2-enhanced-webui/issues)

---

*Made with ❤️ for the AI audio generation community*
