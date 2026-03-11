"""
App - Главный класс для использования gesture_framework как библиотеки

Позволяет просто создать приложение:
    from gesture_framework import App
    
    app = App()
    app.add_gesture("click", model="click.pkl", action="mouse_left")
    app.add_voice_command("hello", "mouse_right")
    app.run()
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from enum import Enum
import logging

# Import gesture framework components
try:
    from .core.platform_abstraction import PlatformController
except ImportError:
    from gesture_framework.core.platform_abstraction import PlatformController

try:
    from .gestures.custom import CustomGesture
except ImportError:
    from gesture_framework.gestures.custom import CustomGesture

try:
    from .gestures.base_gesture import HandLandmark
except ImportError:
    from gesture_framework.gestures.base_gesture import HandLandmark

try:
    from .utils.voice_processor import VoiceProcessor
except ImportError:
    from gesture_framework.utils.voice_processor import VoiceProcessor

# MediaPipe imports for hand detection
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Тип действия для жеста"""
    MOUSE_MOVE = "mouse_move"
    MOUSE_LEFT = "mouse_left"
    MOUSE_RIGHT = "mouse_right"
    MOUSE_DOUBLE = "mouse_double"
    KEYBOARD = "keyboard"
    CUSTOM = "custom"


class GestureAction:
    """Действие, выполняемое при распознании жеста"""
    
    def __init__(self, action_type: ActionType, params: Dict = None, callback: Callable = None):
        """
        Args:
            action_type: Тип действия (MOUSE_LEFT, KEYBOARD и т.д.)
            params: Параметры действия (например, для KEYBOARD - какая клавиша)
            callback: Пользовательская функция для выполнения
        """
        self.action_type = action_type
        self.params = params or {}
        self.callback = callback
        self.controller = PlatformController()
    
    def execute(self, landmarks=None):
        """Выполнить действие"""
        try:
            if self.action_type == ActionType.MOUSE_LEFT:
                self.controller.mouse_click(button='left')
            elif self.action_type == ActionType.MOUSE_RIGHT:
                self.controller.mouse_click(button='right')
            elif self.action_type == ActionType.MOUSE_DOUBLE:
                self.controller.mouse_double_click()
            elif self.action_type == ActionType.KEYBOARD:
                key = self.params.get("key", "a")
                self.controller.keyboard_press(key)
            elif self.action_type == ActionType.MOUSE_MOVE:
                # Прямое управление мышью по landmarks
                if landmarks:
                    x, y = landmarks[0].x, landmarks[0].y
                    self.controller.set_mouse_position(x, y)
            elif self.action_type == ActionType.CUSTOM and self.callback:
                self.callback(landmarks)
            
            logger.debug(f"Action executed: {self.action_type}")
        except Exception as e:
            logger.error(f"Error executing action: {e}")


class GestureBinding:
    """Привязка жеста к действию"""
    
    def __init__(self, name: str, gesture_model: CustomGesture, action: GestureAction,
                 confidence_threshold: float = 0.5):
        """
        Args:
            name: Имя жеста (для логирования)
            gesture_model: Модель жеста
            action: Действие для выполнения
            confidence_threshold: Порог уверенности
        """
        self.name = name
        self.gesture_model = gesture_model
        self.action = action
        self.confidence_threshold = confidence_threshold
        self.last_trigger_time = 0
        self.cooldown = 0.5  # Минимум 500ms между срабатываниями
    
    def try_trigger(self, landmarks) -> bool:
        """
        Проверить жест и выполнить действие если распознан
        
        Args:
            landmarks: Landmarks руки
        
        Returns:
            True если жест был распознан и действие выполнено
        """
        current_time = time.time()
        
        # Cooldown проверка
        if current_time - self.last_trigger_time < self.cooldown:
            return False
        
        # Проверить жест
        is_detected, confidence = self.gesture_model.is_detected(landmarks)
        
        # Диагностический вывод
        logger.debug(f"Gesture '{self.name}': detected={is_detected}, confidence={confidence:.2f}, threshold={self.confidence_threshold}")
        
        if is_detected and confidence >= self.confidence_threshold:
            self.action.execute(landmarks)
            self.last_trigger_time = current_time
            logger.info(f"Gesture '{self.name}' triggered (confidence: {confidence:.2f})")
            return True
        
        return False


class VoiceBinding:
    """Привязка голосовой команды к действию"""
    
    def __init__(self, command: str, action: GestureAction, language: str = "ru"):
        """
        Args:
            command: Голосовая команда
            action: Действие для выполнения
            language: Язык распознавания
        """
        self.command = command
        self.action = action
        self.language = language
        self.last_trigger_time = 0
        self.cooldown = 1.0  # Минимум 1 секунда между командами
    
    def try_trigger(self, recognized_text: str) -> bool:
        """
        Проверить если распознана команда
        
        Args:
            recognized_text: Распознанный текст
        
        Returns:
            True если команда была распознана и действие выполнено
        """
        current_time = time.time()
        
        # Cooldown проверка
        if current_time - self.last_trigger_time < self.cooldown:
            return False
        
        # Проверить совпадение команды
        if self.command.lower() in recognized_text.lower():
            self.action.execute()
            self.last_trigger_time = current_time
            logger.info(f"Voice command '{self.command}' triggered")
            return True
        
        return False


class App:
    """
    Главный класс для использования gesture_framework
    
    Пример:
        app = App()
        app.add_gesture("click", model="click.pkl", action="mouse_left")
        app.add_voice_command("hello", "mouse_right")
        app.run()
    """
    
    def __init__(self, display_camera: bool = False, confidence_threshold: float = 0.5, enable_cursor_control: bool = True):
        """
        Args:
            display_camera: Показывать ли окно камеры
            confidence_threshold: Порог уверенности для всех жестов
            enable_cursor_control: Автоматическое управление курсором по руке
        """
        self.display_camera = display_camera
        self.default_confidence_threshold = confidence_threshold
        self.enable_cursor_control = enable_cursor_control
        
        # Компоненты
        self.hand_landmarker = None
        self._init_mediapipe()
        self.voice_processor = None
        self.platform_controller = PlatformController() if enable_cursor_control else None
        
        # Привязки
        self.gesture_bindings: Dict[str, GestureBinding] = {}
        self.voice_bindings: List[VoiceBinding] = []
        
        # Состояние
        self.running = False
        self.voice_thread = None
        self.cap = None
        
        # Параметры управления курсором
        if enable_cursor_control:
            screen_width, screen_height = PlatformController.get_screen_size()
            self.screen_width = screen_width
            self.screen_height = screen_height
            logger.info(f"Cursor control enabled (screen: {screen_width}x{screen_height})")
        
        logger.info("App initialized")
    
    def _init_mediapipe(self):
        """Инициализировать MediaPipe для детекции рук"""
        if not mp:
            logger.error("MediaPipe not installed. Install with: pip install mediapipe")
            return
        
        try:
            base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("MediaPipe HandLandmarker initialized")
        except Exception as e:
            logger.warning(f"Could not initialize HandLandmarker: {e}. Will try to use fallback.")
    
    def _detect_hands(self, frame) -> List:
        """
        Детектировать руки на кадре
        
        Returns:
            Список landmarks для каждой руки
        """
        if not self.hand_landmarker:
            return []
        
        try:
            # MediaPipe требует RGB изображение
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Детектировать
            detection_result = self.hand_landmarker.detect(mp_image)
            
            if not detection_result.hand_landmarks:
                return []
            
            # Конвертировать в HandLandmark объекты
            landmarks_list = []
            for hand_landmarks in detection_result.hand_landmarks:
                landmarks = [
                    HandLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z if hasattr(lm, 'z') else 0.0,
                        confidence=lm.presence if hasattr(lm, 'presence') else 1.0
                    )
                    for lm in hand_landmarks
                ]
                landmarks_list.append(landmarks)
            
            return landmarks_list
        
        except Exception as e:
            logger.debug(f"Error detecting hands: {e}")
            return []
    
    def add_gesture(self, name: str, model: str, action: str, 
                   confidence_threshold: Optional[float] = None,
                   params: Optional[Dict] = None,
                   callback: Optional[Callable] = None):
        """
        Добавить жест в приложение
        
        Args:
            name: Имя жеста (например, "click")
            model: Путь к модели .pkl файлу (например, "click.pkl")
            action: Тип действия (см. ActionType):
                - "mouse_left" - левый клик мыши
                - "mouse_right" - правый клик мыши
                - "mouse_double" - двойной клик
                - "keyboard" - нажатие клавиши (параметр params)
                - "custom" - пользовательская функция (параметр callback)
            confidence_threshold: Порог уверенности (по умолчанию default из __init__)
            params: Параметры для действия (например, {"key": "space"} для keyboard)
            callback: Пользовательская функция для action="custom"
        
        Returns:
            self (для chaining)
        
        Пример:
            app.add_gesture("click", "click.pkl", "mouse_left")
            app.add_gesture("wave", "wave.pkl", "keyboard", params={"key": "space"})
            app.add_gesture("custom", "custom.pkl", "custom", callback=my_function)
        """
        try:
            # Проверить существование файла модели
            from pathlib import Path
            import os
            
            model_path = Path(model)
            
            # Если путь относительный и не существует, попробовать из текущей директории
            if not model_path.exists():
                # Попробовать из рабочей директории
                alt_path = Path(os.getcwd()) / model
                if alt_path.exists():
                    model_path = alt_path
                    logger.info(f"Found model at: {model_path}")
                else:
                    logger.error(f"Model file not found: {model}")
                    logger.error(f"Tried paths:")
                    logger.error(f"  1. {model_path.absolute()}")
                    logger.error(f"  2. {alt_path.absolute()}")
                    return self
            
            # Загрузить модель жеста
            logger.info(f"Loading gesture model from: {model_path}")
            
            # Использовать переданный threshold или default (0.3 более чувствительный для OneClassSVM)
            threshold = confidence_threshold if confidence_threshold is not None else 0.3
            
            # Загрузить модель с переопределением threshold
            gesture_model = CustomGesture.load(str(model_path), confidence_threshold=threshold)
            
            if gesture_model is None:
                logger.error(f"Failed to load gesture model from {model}")
                return self
            
            # Создать действие
            action_type = ActionType[action.upper()]
            gesture_action = GestureAction(action_type, params=params, callback=callback)
            
            # Создать привязку (уже с правильным threshold в модели)
            binding = GestureBinding(name, gesture_model, gesture_action, threshold)
            
            self.gesture_bindings[name] = binding
            logger.info(f"[OK] Gesture '{name}' added with action '{action}'")
            
            return self
        except Exception as e:
            import traceback
            logger.error(f"[ERROR] Error adding gesture '{name}': {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self
    
    def add_voice_command(self, command: str, action: str, language: str = "ru",
                         params: Optional[Dict] = None,
                         callback: Optional[Callable] = None):
        """
        Добавить голосовую команду
        
        Args:
            command: Текст команды (например, "click", "hello")
            action: Тип действия (см. ActionType)
            language: Язык распознавания ("ru", "en")
            params: Параметры для действия (например, {"key": "space"})
            callback: Пользовательская функция для action="custom"
        
        Returns:
            self (для chaining)
        
        Пример:
            app.add_voice_command("клик", "mouse_left")
            app.add_voice_command("пробел", "keyboard", params={"key": "space"})
            app.add_voice_command("привет", "custom", callback=lambda: print("Hello!"))
        """
        try:
            # Инициализировать voice processor если нужно
            if self.voice_processor is None:
                self.voice_processor = VoiceProcessor(language=language)
            
            # Создать действие
            action_type = ActionType[action.upper()]
            voice_action = GestureAction(action_type, params=params, callback=callback)
            
            # Создать привязку
            binding = VoiceBinding(command, voice_action, language)
            self.voice_bindings.append(binding)
            
            logger.info(f"Voice command '{command}' added with action '{action}'")
            return self
        except Exception as e:
            logger.error(f"Error adding voice command '{command}': {e}")
            return self
    
    def run(self):
        """
        Запустить приложение
        
        Начинает обрабатывать видео с камеры и распознавать жесты/команды
        Нажмите ESC для выхода
        """
        logger.info("App started")
        self.running = True
        
        # Открыть камеру
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        # Запустить голосовой поток если есть команды
        if self.voice_bindings:
            self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
            self.voice_thread.start()
        
        # Основной цикл обработки видео
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Определить руки
                landmarks_list = self._detect_hands(frame)
                
                # Управление курсором (если включено)
                if self.enable_cursor_control and landmarks_list:
                    self._update_cursor_position(landmarks_list[0])
                
                # Проверить каждый жест
                for binding in self.gesture_bindings.values():
                    for landmarks in landmarks_list:
                        binding.try_trigger(landmarks)
                
                # Показать камеру если нужно
                if self.display_camera:
                    self._draw_frame(frame, landmarks_list)
                    cv2.imshow("Gesture Control", frame)
                    
                    # ESC для выхода
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                
                # Контроль FPS
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("App interrupted by user")
        
        finally:
            self.running = False
            self.stop()
    
    def _voice_loop(self):
        """Цикл обработки голосовых команд (в отдельном потоке)"""
        if not self.voice_processor:
            return
        
        logger.info("Voice recognition started")
        
        try:
            while self.running:
                # Слушать микрофон
                text = self.voice_processor.listen()
                
                if text:
                    logger.debug(f"Recognized: {text}")
                    
                    # Проверить голосовые команды
                    for binding in self.voice_bindings:
                        binding.try_trigger(text)
        
        except Exception as e:
            logger.error(f"Error in voice loop: {e}")
    
    def _draw_frame(self, frame, landmarks_list):
        """Нарисовать landmarks на кадре (для display_camera=True)"""
        h, w, _ = frame.shape
        
        # Добавить текстовую информацию
        if self.enable_cursor_control:
            cv2.putText(frame, "Cursor Control: ON", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for landmarks in landmarks_list:
            # Нарисовать точки
            for i, lm in enumerate(landmarks):
                x = int(lm.x * w)
                y = int(lm.y * h)
                
                # Выделить указательный палец (landmark 8) другим цветом
                if i == 8 and self.enable_cursor_control:
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Красный
                    cv2.putText(frame, "CURSOR", (x + 10, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Нарисовать линии между точками (пальцы)
            self._draw_hand_skeleton(frame, landmarks, (0, 255, 0))
    
    def _update_cursor_position(self, landmarks):
        """Обновить позицию курсора на основе позиции руки"""
        if not self.platform_controller or len(landmarks) < 9:
            return
        
        try:
            # Использовать кончик указательного пальца (landmark 8)
            index_finger_tip = landmarks[8]
            
            # Конвертировать координаты (0-1) в пиксели экрана
            # Инвертировать X для зеркального эффекта
            screen_x = int((1 - index_finger_tip.x) * self.screen_width)
            screen_y = int(index_finger_tip.y * self.screen_height)
            
            # Ограничить координаты в пределах экрана
            screen_x = max(0, min(screen_x, self.screen_width - 1))
            screen_y = max(0, min(screen_y, self.screen_height - 1))
            
            # Переместить курсор
            self.platform_controller.move_cursor(screen_x, screen_y)
        
        except Exception as e:
            logger.debug(f"Error updating cursor: {e}")
    
    def _draw_hand_skeleton(self, frame, landmarks, color):
        """Нарисовать скелет руки"""
        h, w, _ = frame.shape
        
        # Соединения между точками (MediaPipe hand skeleton)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # index
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        ]
        
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                x1 = int(landmarks[start].x * w)
                y1 = int(landmarks[start].y * h)
                x2 = int(landmarks[end].x * w)
                y2 = int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    def stop(self):
        """Остановить приложение"""
        logger.info("App stopped")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()
