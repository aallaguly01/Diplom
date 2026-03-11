# gesture_framework/gestures/base_gesture.py
"""
Базовые классы для работы с жестами и ключевыми точками руки
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional


class HandLandmark:
    """
    Одна ключевая точка руки (часть набора из 21 точки от MediaPipe)
    
    Координаты нормализованы (0.0 - 1.0)
    """
    
    def __init__(self, x: float, y: float, z: float, confidence: float = 1.0):
        """
        Args:
            x: Координата по оси X (0.0 - 1.0)
            y: Координата по оси Y (0.0 - 1.0)
            z: Глубина / координата по оси Z (0.0 - 1.0)
            confidence: Уверенность детекции (0.0 - 1.0)
        """
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence
    
    def distance_to(self, other: 'HandLandmark') -> float:
        """
        Евклидово расстояние до другой точки
        
        Args:
            other: Другая точка ключевой точки
            
        Returns:
            Расстояние (в нормализованных координатах)
        """
        return ((self.x - other.x)**2 + 
                (self.y - other.y)**2 + 
                (self.z - other.z)**2) ** 0.5
    
    def __repr__(self) -> str:
        return f"HandLandmark(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class BaseGesture(ABC):
    """
    Базовый класс для любого жеста
    
    Определяет интерфейс для обнаружения жестов по ключевым точкам руки
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Название жеста (например 'POINTING', 'PINCH')
        """
        self.name = name
        self.landmarks: List[HandLandmark] = []
    
    @abstractmethod
    def is_detected(self, landmarks: List[HandLandmark]) -> Tuple[bool, float]:
        """
        Проверить является ли набор точек данным жестом
        
        Args:
            landmarks: Список из 21 ключевой точки руки
            
        Returns:
            Кортеж (is_detected: bool, confidence: float 0.0-1.0)
        """
        pass
    
    def get_bounding_box(self) -> Dict[str, float]:
        """
        Получить bounding box вокруг руки
        
        Returns:
            {'min_x', 'max_x', 'min_y', 'max_y', 'width', 'height'}
        """
        if not self.landmarks:
            return {}
        
        xs = [l.x for l in self.landmarks]
        ys = [l.y for l in self.landmarks]
        
        return {
            'min_x': min(xs),
            'max_x': max(xs),
            'min_y': min(ys),
            'max_y': max(ys),
            'width': max(xs) - min(xs),
            'height': max(ys) - min(ys),
        }


# Индексы ключевых точек MediaPipe Hands (всего 21 точка)
LANDMARK_INDICES = {
    # Запястье
    'wrist': 0,
    
    # Большой палец (4 точки)
    'thumb_cmc': 1,      # Карпометакарпальный сустав
    'thumb_mcp': 2,      # Пястно-фаланговый сустав
    'thumb_ip': 3,       # Межфаланговый сустав
    'thumb_tip': 4,      # Кончик
    
    # Указательный палец (4 точки)
    'index_mcp': 5,      # Пястно-фаланговый сустав
    'index_pip': 6,      # Проксимальный суставfinger inter-phalangeal
    'index_dip': 7,      # Дистальный межфаланговый сустав
    'index_tip': 8,      # Кончик
    
    # Средний палец (4 точки)
    'middle_mcp': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    'middle_tip': 12,
    
    # Безымянный палец (4 точки)
    'ring_mcp': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    'ring_tip': 16,
    
    # Мизинец (4 точки)
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20,
}

# Обратный индекс (номер → название)
LANDMARK_NAMES = {v: k for k, v in LANDMARK_INDICES.items()}
