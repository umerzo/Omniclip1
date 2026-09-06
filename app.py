"""OmniClip AI Content Studio - Streamlit Cloud Entrypoint."""
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from omniclip.frontend import theme
import omniclip.frontend.app as studio_app

# Set page config
try:
    st.set_page_config(page_title="OmniClip AI Studio", page_icon="🎬", layout="wide")
except Exception:
    pass

# Always inject the theme CSS stylesheet on every frame
theme.apply()

# Run application
studio_app.main()
