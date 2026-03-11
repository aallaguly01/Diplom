"""
gesture_framework - мультиплатформенный SDK для жестового и голосового управления

Использование:
    from gesture_framework import App
    
    app = App()
    app.add_gesture("click", model="click.pkl", action="mouse_left")
    app.add_voice_command("hello", "mouse_right")
    app.run()
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

__version__ = "0.1.0"
__author__ = "Diploma Project"

from .gestures.classifier import GestureClassifier
from .gestures.base_gesture import HandLandmark
from .core.platform_abstraction import PlatformController
from .app import App, ActionType

__all__ = [
    'App',
    'ActionType',
    'GestureClassifier',
    'HandLandmark',
    'PlatformController',
]
