from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseSentimentModel(ABC):
    """
    الكلاس الأب الموحد لجميع نماذج المشروع الأربعة.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """
        يجب أن يُرجع القاموس الموحد:
        {
            "model_name": str,
            "label": str,
            "confidence": float,
            "latency_ms": float
        }
        """
        pass

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """تطبيق التنبؤ على قائمة تويتات (لخيار رفع CSV في Tab 1)."""
        return [self.predict(text) for text in texts]