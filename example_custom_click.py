#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

from gesture_framework import App

# Enable DEBUG logging to see detection details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "custom_gesture" / "my_click.pkl"


def main() -> None:

    
    app = App(display_camera=True)

    app.add_gesture(
        name="my_click",
        model=str(MODEL_PATH),
        action="mouse_left",
        confidence_threshold=0.70,  # With margin check (25%), this is balanced
    )

    app.run()


if __name__ == "__main__":
    main()
