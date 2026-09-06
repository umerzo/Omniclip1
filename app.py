"""OmniClip AI Content Studio - Streamlit Cloud Entrypoint."""
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omniclip.frontend.app import main

if __name__ == "__main__":
    main()
