#!/usr/bin/env python
"""Launcher for the real-time subtitle translator.

Usage:
    python run.py                          # Start live subtitle overlay
    python run.py --srt subtitles.srt      # Export to SRT
    python run.py --compare                # Also write heard/translated comparison log
    python run.py -v                       # Verbose debug logging
"""

import sys
from src.main import main

if __name__ == "__main__":
    sys.exit(main())
