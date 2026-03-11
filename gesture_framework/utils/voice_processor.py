"""
VoiceProcessor - распознавание голоса и преобразование в текст

Поддерживает:
- Локальное распознавание (via speech_recognition)
- Русский и английский языки
- Фоновый шум фильтрация
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """Обработчик голосовых команд"""
    
    def __init__(self, language: str = "ru", use_google: bool = False):
        """
        Args:
            language: Язык распознавания ("ru", "en")
            use_google: Использовать ли Google Speech API (требует интернет)
        """
        self.language = language
        self.use_google = use_google
        self.recognizer = None
        self.microphone = None
        
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            logger.info(f"VoiceProcessor initialized (language: {language})")
        except ImportError:
            logger.error("speech_recognition not installed. Install with: pip install SpeechRecognition pydub")
            raise
    
    def listen(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        Слушать микрофон и распознать речь
        
        Args:
            timeout: Максимальное время ожидания начала речи (сек)
            phrase_time_limit: Максимальное время записи речи (сек)
        
        Returns:
            Распознанный текст или None если ошибка
        """
        if not self.recognizer:
            logger.error("Voice processor not initialized")
            return None
        
        try:
            with self.microphone as source:
                # Откалибровать на шум в комнате
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Записать аудио
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            # Распознать
            return self._recognize_audio(audio)
        
        except Exception as e:
            logger.debug(f"Error listening: {e}")
            return None
    
    def _recognize_audio(self, audio) -> Optional[str]:
        """Распознать аудио"""
        if not self.recognizer:
            return None
        
        try:
            if self.use_google:
                # Использовать Google Speech Recognition API
                text = self.recognizer.recognize_google(
                    audio,
                    language=self._language_code()
                )
            else:
                # Использовать встроенное распознавание (требует pocketsphinx)
                text = self.recognizer.recognize_sphinx(audio)
            
            logger.debug(f"Recognized text: {text}")
            return text.lower()
        
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
            return None
    
    def _language_code(self) -> str:
        """Преобразовать язык в код Google"""
        codes = {
            "ru": "ru-RU",
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            "it": "it-IT",
            "ja": "ja-JP",
            "zh": "zh-CN",
        }
        return codes.get(self.language, "en-US")
    
    def set_language(self, language: str):
        """Изменить язык распознавания"""
        self.language = language
        logger.info(f"Language changed to: {language}")
