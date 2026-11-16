"""
Audio-Visual Deepfake Detection package.

Pipeline:  video -> audio + frames -> audio features + lip features
           -> audio-visual sync -> fusion -> Random Forest -> report
"""

# Must run before Librosa/numba are imported (see avdd/_paths.py).
from ._paths import shorten_windows_sys_path

shorten_windows_sys_path()

from .pipeline import analyze_video, environment_report  # noqa: E402,F401

__version__ = "1.0.0"
