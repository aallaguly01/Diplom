# gesture_framework/gestures/predefined.py
"""
Пять базовых жестов: POINTING, PINCH, OPEN_PALM, CLOSED_FIST, PEACE
"""

from typing import List, Tuple
from .base_gesture import BaseGesture, HandLandmark


class PointingGesture(BaseGesture):
    """
    POINTING жест: указательный палец вытянут, остальные согнуты
    
    Алгоритм:
    - index_tip находится выше index_pip (палец вытянут вверх)
    - middle_pip находится выше middle_tip (согнут)
    - ring_pip находится выше ring_tip (согнут)
    - pinky_pip находится выше pinky_tip (согнут)
    """
    
    def __init__(self):
        super().__init__("POINTING")
    
    def is_detected(self, landmarks: List[HandLandmark]) -> Tuple[bool, float]:
        """Обнаружить жест POINTING"""
        if len(landmarks) != 21:
            return False, 0.0
        
        # Индексы точек
        INDEX_TIP = 8
        INDEX_PIP = 6
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        RING_TIP = 16
        RING_PIP = 14
        PINKY_TIP = 20
        PINKY_PIP = 18
        
        checks = []
        
        # 1. Указательный палец вытянут (tip выше pip)
        if landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y:
            checks.append(True)
        else:
            checks.append(False)
        
        # 2. Средний палец согнут
        if landmarks[MIDDLE_PIP].y < landmarks[MIDDLE_TIP].y:
            checks.append(True)
        else:
            checks.append(False)
        
        # 3. Безымянный палец согнут
        if landmarks[RING_PIP].y < landmarks[RING_TIP].y:
            checks.append(True)
        else:
            checks.append(False)
        
        # 4. Мизинец согнут
        if landmarks[PINKY_PIP].y < landmarks[PINKY_TIP].y:
            checks.append(True)
        else:
            checks.append(False)
        
        # Confidence = % успешных проверок
        confidence = sum(checks) / len(checks) if checks else 0.0
        is_detected = confidence >= 0.75
        
        return is_detected, confidence


class PinchGesture(BaseGesture):
    """
    PINCH жест: большой и указательный пальцы соединены
    
    Алгоритм:
    - Расстояние между thumb_tip и index_tip < 0.08
    - Этот жест используется для нажатия кнопки
    """
    
    def __init__(self):
        super().__init__("PINCH")
    
    def is_detected(self, landmarks: List[HandLandmark]) -> Tuple[bool, float]:
        """Обнаружить жест PINCH"""
        if len(landmarks) != 21:
            return False, 0.0
        
        THUMB_TIP = 4
        INDEX_TIP = 8
        
        # Расстояние между большим и указательным
        distance = landmarks[THUMB_TIP].distance_to(landmarks[INDEX_TIP])
        
        # Если расстояние маленькое (< 0.08) - это PINCH
        # Создаем confidence от 1 до 0 по мере увеличения расстояния
        if distance < 0.05:
            confidence = 1.0
        elif distance < 0.08:
            confidence = 1.0 - (distance - 0.05) / 0.03
        else:
            confidence = 0.0
        
        is_detected = confidence >= 0.7
        
        return is_detected, confidence


class OpenPalmGesture(BaseGesture):
    """
    OPEN_PALM жест: все пальцы расправлены
    
    Алгоритм:
    - Все кончики пальцев находятся ВЫШЕ их PIP суставов
    """
    
    def __init__(self):
        super().__init__("OPEN_PALM")
    
    def is_detected(self, landmarks: List[HandLandmark]) -> Tuple[bool, float]:
        """Обнаружить жест OPEN_PALM"""
        if len(landmarks) != 21:
            return False, 0.0
        
        # (tip_index, pip_index) для каждого пальца
        finger_pairs = [
            (4, 3),    # thumb
            (8, 6),    # index
            (12, 10),  # middle
            (16, 14),  # ring
            (20, 18),  # pinky
        ]
        
        checks = []
        for tip_idx, pip_idx in finger_pairs:
            # Распрямлен если tip выше pip
            if landmarks[tip_idx].y < landmarks[pip_idx].y:
                checks.append(True)
            else:
                checks.append(False)
        
        confidence = sum(checks) / len(checks) if checks else 0.0
        is_detected = confidence >= 0.8
        
        return is_detected, confidence


class ClosedFistGesture(BaseGesture):
    """
    CLOSED_FIST жест: кулак (все пальцы согнуты)
    
    Алгоритм:
    - Все кончики пальцев находятся НИЖЕ их PIP суставов
    """
    
    def __init__(self):
        super().__init__("CLOSED_FIST")
    
    def is_detected(self, landmarks: List[HandLandmark]) -> Tuple[bool, float]:
        """Обнаружить жест CLOSED_FIST"""
        if len(landmarks) != 21:
            return False, 0.0
        
        finger_pairs = [
            (4, 3),    # thumb
            (8, 6),    # index
            (12, 10),  # middle
            (16, 14),  # ring
            (20, 18),  # pinky
        ]
        
        checks = []
        for tip_idx, pip_idx in finger_pairs:
            # Согнут если tip ниже pip
            if landmarks[tip_idx].y > landmarks[pip_idx].y:
                checks.append(True)
            else:
                checks.append(False)
        
        confidence = sum(checks) / len(checks) if checks else 0.0
        is_detected = confidence >= 0.8
        
        return is_detected, confidence


class PeaceGesture(BaseGesture):
    """
    PEACE жест: V-sign (указательный и средний пальцы вытянуты)
    
    Алгоритм:
    - index_tip выше index_pip (вытянут)
    - middle_tip выше middle_pip (вытянут)
    - ring_tip ниже ring_pip (согнут)
    - pinky_tip ниже pinky_pip (согнут)
    """
    
    def __init__(self):
        super().__init__("PEACE")
    
    def is_detected(self, landmarks: List[HandLandmark]) -> Tuple[bool, float]:
        """Обнаружить жест PEACE"""
        if len(landmarks) != 21:
            return False, 0.0
        
        checks = [
            landmarks[8].y < landmarks[6].y,    # index extended
            landmarks[12].y < landmarks[10].y,  # middle extended
            landmarks[16].y > landmarks[14].y,  # ring curled
            landmarks[20].y > landmarks[18].y,  # pinky curled
        ]
        
        confidence = sum(checks) / len(checks) if checks else 0.0
        is_detected = confidence >= 0.75
        
        return is_detected, confidence


# Стандартный набор из 5 базовых жестов
PREDEFINED_GESTURES = {
    'POINTING': PointingGesture(),
    'PINCH': PinchGesture(),
    'OPEN_PALM': OpenPalmGesture(),
    'CLOSED_FIST': ClosedFistGesture(),
    'PEACE': PeaceGesture(),
}
