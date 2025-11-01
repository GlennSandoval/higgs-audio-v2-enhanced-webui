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

def verify_application_wiring():
    """Ensure the central application wiring can be constructed."""
    try:
        from gradio_app.main import create_app

        demo = create_app()
        if demo is None:
            raise RuntimeError("Gradio app factory returned None")
        print("✅ Application wiring constructed successfully")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Application wiring failed: {exc}")
        return False

def main():
    """Main verification function."""
    print("🎵 Higgs Audio Setup Verification")
    print("=" * 40)
    
    imports_ok = verify_imports()
    wiring_ok = verify_application_wiring() if imports_ok else False

    if imports_ok and wiring_ok:
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
