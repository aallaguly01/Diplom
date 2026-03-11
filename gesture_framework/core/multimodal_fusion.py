"""
Multimodal Fusion - объединение жестов и голосовых команд

Интеллектуально комбинирует данные из нескольких модальностей
(жесты, голос) для выполнения действий
"""

from typing import Optional, Dict, Tuple, Callable
from enum import Enum
from dataclasses import dataclass


class ConflictResolution(Enum):
    """Стратегия разрешения конфликтов"""
    PREFER_HIGH_CONFIDENCE = "prefer_high_confidence"  # Выбрать более уверенный сигнал
    PREFER_GESTURE = "prefer_gesture"                   # Приоритет жестам
    PREFER_VOICE = "prefer_voice"                       # Приоритет голосу
    REQUIRE_AGREEMENT = "require_agreement"             # Оба сигнала должны согласиться
    WEIGHTED_AVERAGE = "weighted_average"               # Взвешенное среднее


@dataclass
class MultimodalContext:
    """Контекст для мультимодальной обработки"""
    gesture: Optional[str] = None
    gesture_confidence: float = 0.0
    voice_command: Optional[str] = None
    voice_confidence: float = 0.0
    timestamp: float = 0.0
    user_context: Dict = None  # Дополнительный контекст (какое приложение активно и т.д.)


class MultimodalFusion:
    """
    Мультимодальное объединение жестов и голоса
    
    Комбинирует сигналы из нескольких модальностей для лучшего распознавания
    """
    
    def __init__(self,
                 gesture_weight: float = 0.6,
                 voice_weight: float = 0.4,
                 conflict_resolution: ConflictResolution = ConflictResolution.PREFER_HIGH_CONFIDENCE):
        """
        Args:
            gesture_weight: Вес жестов (0.0-1.0)
            voice_weight: Вес голоса (0.0-1.0)
            conflict_resolution: Стратегия разрешения конфликтов
        """
        # Нормализировать веса
        total = gesture_weight + voice_weight
        self.gesture_weight = gesture_weight / total
        self.voice_weight = voice_weight / total
        
        self.conflict_resolution = conflict_resolution
        
        # Маппинг голосовых команд на действия
        self.voice_commands = {
            'клик': ('click', 0.8),
            'правый клик': ('right_click', 0.8),
            'скролл вверх': ('scroll_up', 0.7),
            'скролл вниз': ('scroll_down', 0.7),
            'двойной клик': ('double_click', 0.8),
        }
        
        # История контекстов для анализа
        self.context_history = []
        self.max_history = 10
        
        # Логирование
        self.enable_logging = True
        
        print("MultimodalFusion инициализирован")
        print(f"   Веса: жесты {self.gesture_weight:.1%}, голос {self.voice_weight:.1%}")
        print(f"   Стратегия: {conflict_resolution.value}")
    
    def process(self, 
                gesture: Optional[str] = None,
                gesture_confidence: float = 0.0,
                voice_command: Optional[str] = None,
                voice_confidence: float = 0.0,
                user_context: Optional[Dict] = None) -> Dict:
        """
        Обработать мультимодальные сигналы
        
        Args:
            gesture: Распознанный жест (e.g., 'POINTING')
            gesture_confidence: Уверенность жеста (0.0-1.0)
            voice_command: Голосовая команда (e.g., 'click')
            voice_confidence: Уверенность голоса (0.0-1.0)
            user_context: Контекст (какое приложение активно и т.д.)
        
        Returns:
            {
                'action': 'click',
                'confidence': 0.85,
                'source': 'multimodal',  # 'gesture', 'voice', or 'multimodal'
                'reasoning': 'description of decision',
                'context': MultimodalContext
            }
        """
        import time
        
        # Создать контекст
        context = MultimodalContext(
            gesture=gesture,
            gesture_confidence=gesture_confidence,
            voice_command=voice_command,
            voice_confidence=voice_confidence,
            timestamp=time.time(),
            user_context=user_context or {}
        )
        
        # Сохранить в историю
        self.context_history.append(context)
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)
        
        # Проверить наличие сигналов
        has_gesture = gesture is not None and gesture_confidence > 0.0
        has_voice = voice_command is not None and voice_confidence > 0.0
        
        # Если нет сигналов - вернуть пусто
        if not has_gesture and not has_voice:
            return {
                'action': None,
                'confidence': 0.0,
                'source': 'none',
                'reasoning': 'No signals detected',
                'context': context
            }
        
        # Единственный сигнал
        if has_gesture and not has_voice:
            return {
                'action': gesture,
                'confidence': gesture_confidence,
                'source': 'gesture',
                'reasoning': f'Only gesture detected: {gesture}',
                'context': context
            }
        
        if has_voice and not has_gesture:
            action = self._voice_to_action(voice_command)
            return {
                'action': action,
                'confidence': voice_confidence,
                'source': 'voice',
                'reasoning': f'Only voice detected: {voice_command}',
                'context': context
            }
        
        # Оба сигнала присутствуют - применить стратегию
        return self._resolve_conflict(context)
    
    def _voice_to_action(self, voice_command: str) -> Optional[str]:
        """
        Конвертировать голосовую команду в действие
        
        Args:
            voice_command: Голосовая команда
        
        Returns:
            Имя действия или None
        """
        voice_lower = voice_command.lower() if voice_command else ""
        
        # Прямой поиск в маппинге
        if voice_lower in self.voice_commands:
            action, _ = self.voice_commands[voice_lower]
            return action
        
        # Нечеткий поиск (если есть ключевые слова)
        for command_key, (action, _) in self.voice_commands.items():
            if any(word in voice_lower for word in command_key.split()):
                return action
        
        return voice_command  # Вернуть как есть
    
    def _resolve_conflict(self, context: MultimodalContext) -> Dict:
        """
        Разрешить конфликт между жестом и голосом
        
        Args:
            context: Мультимодальный контекст
        
        Returns:
            Результат обработки
        """
        gesture_action = context.gesture
        voice_action = self._voice_to_action(context.voice_command)
        
        gesture_conf = context.gesture_confidence
        voice_conf = context.voice_confidence
        
        if self.conflict_resolution == ConflictResolution.PREFER_HIGH_CONFIDENCE:
            # Выбрать более уверенный сигнал
            if gesture_conf >= voice_conf:
                return {
                    'action': gesture_action,
                    'confidence': gesture_conf,
                    'source': 'gesture',
                    'reasoning': f'Gesture has higher confidence ({gesture_conf:.2f} vs {voice_conf:.2f})',
                    'context': context
                }
            else:
                return {
                    'action': voice_action,
                    'confidence': voice_conf,
                    'source': 'voice',
                    'reasoning': f'Voice has higher confidence ({voice_conf:.2f} vs {gesture_conf:.2f})',
                    'context': context
                }
        
        elif self.conflict_resolution == ConflictResolution.PREFER_GESTURE:
            return {
                'action': gesture_action,
                'confidence': gesture_conf,
                'source': 'gesture',
                'reasoning': 'Gesture prioritized over voice',
                'context': context
            }
        
        elif self.conflict_resolution == ConflictResolution.PREFER_VOICE:
            return {
                'action': voice_action,
                'confidence': voice_conf,
                'source': 'voice',
                'reasoning': 'Voice prioritized over gesture',
                'context': context
            }
        
        elif self.conflict_resolution == ConflictResolution.REQUIRE_AGREEMENT:
            # Оба сигнала должны указывать на одно действие
            # Для жестов это очевидно, для голоса нужно сопоставить
            if gesture_action == voice_action or self._actions_compatible(gesture_action, voice_action):
                avg_conf = (gesture_conf * self.gesture_weight + 
                           voice_conf * self.voice_weight)
                return {
                    'action': gesture_action,
                    'confidence': avg_conf,
                    'source': 'multimodal',
                    'reasoning': 'Both gesture and voice agree',
                    'context': context
                }
            else:
                return {
                    'action': None,
                    'confidence': 0.0,
                    'source': 'conflict',
                    'reasoning': f'Conflict: gesture={gesture_action} vs voice={voice_action}',
                    'context': context
                }
        
        elif self.conflict_resolution == ConflictResolution.WEIGHTED_AVERAGE:
            # Взвешенное среднее
            avg_conf = (gesture_conf * self.gesture_weight + 
                       voice_conf * self.voice_weight)
            
            # Выбрать действие с большей уверенностью
            if gesture_conf >= voice_conf:
                action = gesture_action
            else:
                action = voice_action
            
            return {
                'action': action,
                'confidence': avg_conf,
                'source': 'multimodal',
                'reasoning': f'Weighted average: ({gesture_conf:.2f} * {self.gesture_weight:.1%} + {voice_conf:.2f} * {self.voice_weight:.1%})',
                'context': context
            }
    
    @staticmethod
    def _actions_compatible(action1: str, action2: str) -> bool:
        """
        Проверить, совместимы ли два действия
        
        Например, "PINCH" (жест) и "click" (голос) совместимы
        """
        compatibility_map = {
            'PINCH': ['click'],
            'PEACE': ['right_click'],
            'POINTING': ['move_cursor'],
        }
        
        if action1 in compatibility_map:
            return action2 in compatibility_map[action1]
        
        return action1 == action2
    
    def set_weights(self, gesture_weight: float, voice_weight: float) -> None:
        """
        Установить новые веса
        
        Args:
            gesture_weight: Вес жестов
            voice_weight: Вес голоса
        """
        total = gesture_weight + voice_weight
        self.gesture_weight = gesture_weight / total
        self.voice_weight = voice_weight / total
        print(f"Веса обновлены: жесты {self.gesture_weight:.1%}, голос {self.voice_weight:.1%}")
    
    def get_context_summary(self) -> Dict:
        """Получить сводку по истории контекстов"""
        if not self.context_history:
            return {}
        
        gesture_count = sum(1 for c in self.context_history if c.gesture is not None)
        voice_count = sum(1 for c in self.context_history if c.voice_command is not None)
        both_count = sum(1 for c in self.context_history 
                        if c.gesture is not None and c.voice_command is not None)
        
        return {
            'total_events': len(self.context_history),
            'gesture_only': gesture_count - both_count,
            'voice_only': voice_count - both_count,
            'multimodal': both_count,
            'gesture_percentage': gesture_count / len(self.context_history) * 100,
            'voice_percentage': voice_count / len(self.context_history) * 100,
        }


if __name__ == "__main__":
    print("Testing MultimodalFusion...")
    
    fusion = MultimodalFusion(
        gesture_weight=0.6,
        voice_weight=0.4,
        conflict_resolution=ConflictResolution.PREFER_HIGH_CONFIDENCE
    )
    
    # Тест 1: Только жест
    print("\n[1] Only gesture:")
    result = fusion.process(gesture='PINCH', gesture_confidence=0.9)
    print(f"  Action: {result['action']}, Confidence: {result['confidence']:.2f}")
    
    # Тест 2: Только голос
    print("\n[2] Only voice:")
    result = fusion.process(voice_command='клик', voice_confidence=0.8)
    print(f"  Action: {result['action']}, Confidence: {result['confidence']:.2f}")
    
    # Тест 3: Оба сигнала (жест сильнее)
    print("\n[3] Both signals (gesture stronger):")
    result = fusion.process(
        gesture='PINCH', gesture_confidence=0.95,
        voice_command='move', voice_confidence=0.6
    )
    print(f"  Action: {result['action']}, Confidence: {result['confidence']:.2f}")
    
    # Тест 4: Оба сигнала (голос сильнее)
    print("\n[4] Both signals (voice stronger):")
    result = fusion.process(
        gesture='POINTING', gesture_confidence=0.5,
        voice_command='клик', voice_confidence=0.95
    )
    print(f"  Action: {result['action']}, Confidence: {result['confidence']:.2f}")
    
    print("\n[OK] MultimodalFusion test completed")
