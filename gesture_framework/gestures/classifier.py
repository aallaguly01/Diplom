# gesture_framework/gestures/classifier.py
"""
Главный классификатор жестов - обнаруживает и классифицирует жесты
"""

from typing import Optional, List, Dict
from .predefined import PREDEFINED_GESTURES
from .base_gesture import HandLandmark


class GestureClassifier:
    """
    Классификатор жестов - основной движок распознавания
    
    Использует исходные 5 жестов и применяет сглаживание
    для стабильности детекции
    """
    
    def __init__(self, smoothing_window_size: int = 5):
        """
        Args:
            smoothing_window_size: Размер окна для сглаживания (больше = стабильнее, но медленнее)
        """
        self.gestures = PREDEFINED_GESTURES
        self.gesture_history = []
        self.history_size = smoothing_window_size
        self.confidence_threshold = 0.5  # Минимальная уверенность для распознавания
    
    def classify(self, landmarks: List[HandLandmark]) -> Dict:
        """
        Классифицировать жест по ключевым точкам
        
        Args:
            landmarks: Список из 21 ключевой точки руки
            
        Returns:
            {
                'gesture': 'POINTING' или None,
                'confidence': 0.92,
                'all_scores': {'POINTING': 0.92, 'PINCH': 0.15, ...}
            }
        """
        scores = {}
        
        # Получить уверенность для каждого жеста
        for gesture_name, gesture in self.gestures.items():
            is_detected, confidence = gesture.is_detected(landmarks)
            scores[gesture_name] = confidence
        
        # Найти жест с максимальной уверенностью
        best_gesture = max(scores, key=scores.get)
        best_confidence = scores[best_gesture]
        
        # Если уверенность слишком низкая - вернуть None
        if best_confidence < self.confidence_threshold:
            best_gesture = None
        
        return {
            'gesture': best_gesture,
            'confidence': best_confidence,
            'all_scores': scores,
        }
    
    def smooth_gesture_sequence(self, current_gesture: Optional[str]) -> Optional[str]:
        """
        Сглаживание жестов во времени
        
        Избегает мерцания между POINTING и PINCH благодаря
        выбору наиболее частого жеста в последних N кадрах
        
        Args:
            current_gesture: Текущий распознанный жест
            
        Returns:
            Сглаженный жест (или None)
        """
        self.gesture_history.append(current_gesture)
        
        # Удалить старые записи
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Если история пуста - вернуть текущий
        if not self.gesture_history:
            return current_gesture
        
        # Найти наиболее частый жест в истории (исключая None)
        gesture_counts = {}
        for gesture in self.gesture_history:
            if gesture is not None:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Если никакой жест не был распознан - вернуть None
        if not gesture_counts:
            return None
        
        # Вернуть наиболее частый жест
        return max(gesture_counts, key=gesture_counts.get)
    
    def reset_smoothing(self) -> None:
        """Сбросить историю жестов (при паузе или переключении)"""
        self.gesture_history = []
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Установить минимальный порог уверенности
        
        Args:
            threshold: Значение от 0.0 до 1.0
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
    
    def get_performance_stats(self) -> Dict:
        """
        Получить статистику о классификаторе
        
        Returns:
            {'total_classifications': 100, 'history_size': 5}
        """
        return {
            'total_gestures': len(self.gestures),
            'smoothing_window_size': self.history_size,
            'current_history_length': len(self.gesture_history),
            'confidence_threshold': self.confidence_threshold,
        }
