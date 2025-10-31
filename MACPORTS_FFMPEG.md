# MacPorts FFmpeg Fix for Higgs Audio v2

## Problem
When FFmpeg is installed via MacPorts on macOS, torchcodec cannot find the FFmpeg libraries because they are installed in `/opt/local/lib` instead of the standard system paths.

## Solution

### Quick Fix (Temporary)
Run the application with the library path set:
```bash
export DYLD_LIBRARY_PATH="/opt/local/lib:$DYLD_LIBRARY_PATH"
uv run python higgs_audio_gradio.py
```

### Permanent Fix (Recommended)
Use the provided launch script:
```bash
./run.sh
```

The `run.sh` script automatically sets the correct environment variables.

## Alternative Solutions

### Option 1: Add to shell profile
Add this line to your `~/.zshrc`:
```bash
export DYLD_LIBRARY_PATH="/opt/local/lib:$DYLD_LIBRARY_PATH"
```
Then reload your shell: `source ~/.zshrc`

### Option 2: Create symlinks (Advanced)
You can create symlinks from the MacPorts FFmpeg libraries to a standard location, but this is not recommended as it may cause conflicts with other software.

### Option 3: Install FFmpeg via Homebrew instead
If you prefer, you can uninstall MacPorts FFmpeg and install via Homebrew:
```bash
sudo port uninstall ffmpeg
brew install ffmpeg
```
Homebrew installs to `/opt/homebrew/lib` (Apple Silicon) or `/usr/local/lib` (Intel), which are typically in the library search path.

## Verification
To verify FFmpeg is properly detected, the application should start without the torchcodec RuntimeError about missing FFmpeg libraries.

Your FFmpeg installation:
- **Location**: `/opt/local/bin/ffmpeg`
- **Version**: 4.4.6 (FFmpeg version 4)
- **Libraries**: `/opt/local/lib/libav*.dylib`
