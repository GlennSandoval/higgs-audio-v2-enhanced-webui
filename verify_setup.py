#!/usr/bin/env python3
"""
Simple script to verify that the Higgs Audio setup is working correctly.
"""

def verify_imports():
    """Verify that all required modules can be imported."""
    print("🔍 Verifying imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import gradio
        print(f"✅ Gradio {gradio.__version__}")
    except ImportError as e:
        print(f"❌ Gradio import failed: {e}")
        return False
    
    try:
        import boson_multimodal
        print("✅ Boson Multimodal package")
    except ImportError as e:
        print(f"❌ Boson Multimodal import failed: {e}")
        return False
    
    return True

def main():
    """Main verification function."""
    print("🎵 Higgs Audio Setup Verification")
    print("=" * 40)
    
    if verify_imports():
        print("\n🎉 All imports successful! Setup is working correctly.")
        print("\n📝 You can now run:")
        print("   uv run python higgs_audio_gradio.py")
        return True
    else:
        print("\n❌ Setup verification failed. Please check your installation.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)