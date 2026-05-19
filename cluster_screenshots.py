#!/usr/bin/env python3
"""Compatibility wrapper — delegates to k6_visual_inspector.cli.main.

All logic has been moved to the k6_visual_inspector package.
Run via:
    python3 cluster_screenshots.py <input> <output> [options]
or:
    python3 -m k6_visual_inspector <input> <output> [options]
"""

from k6_visual_inspector.cli import main

if __name__ == "__main__":
    main()
