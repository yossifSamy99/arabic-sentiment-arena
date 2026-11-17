import time
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.base import BaseSentimentModel
from core.preprocessing import clean_classical, clean_transformer


# ==========================================
# 1. TF-IDF + Logistic Regression Wrapper
# ==========================================
class TfidfWrapper(BaseSentimentModel):
    def __init__(
        self,
        repo_id: str = "YossifSamy99/arabic-nlp-models",
        id2label: Optional[Dict[Any, str]] = None
    ):
        super().__init__(model_name="TF-IDF + Logistic Regression")
        model_path = hf_hub_download(repo_id=repo_id, filename="tfidf_pipeline.joblib")
        self.pipeline = joblib.load(model_path)
        self.id2label = id2label or {}

    def _format_label(self, raw_label: Any) -> str:
        return str(self.id2label.get(raw_label, raw_label))

    def predict(self, text: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        cleaned_text = clean_classical(text)
        probs = self.pipeline.predict_proba([cleaned_text])[0]
        pred_idx = int(np.argmax(probs))

        raw_label = self.pipeline.classes_[pred_idx]
        confidence = round(float(probs[pred_idx]), 4)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "label": self._format_label(raw_label),
            "confidence": confidence,
            "latency_ms": latency_ms
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        start_time = time.perf_counter()

        cleaned_texts = [clean_classical(t) for t in texts]
        all_probs = self.pipeline.predict_proba(cleaned_texts)
        pred_indices = np.argmax(all_probs, axis=1)

        total_ms = (time.perf_counter() - start_time) * 1000
        per_item_ms = round(total_ms / len(texts), 2)

        return [
            {
                "model_name": self.model_name,
                "label": self._format_label(self.pipeline.classes_[idx]),
                "confidence": round(float(probs[idx]), 4),
                "latency_ms": per_item_ms
            }
            for idx, probs in zip(pred_indices, all_probs)
        ]


# ==========================================
# 2. FastText + Logistic Regression Wrapper
# ==========================================
class FastTextWrapper(BaseSentimentModel):
    def __init__(
        self,
        repo_id: str = "YossifSamy99/arabic-nlp-models",
        id2label: Optional[Dict[Any, str]] = None
    ):
        super().__init__(model_name="FastText + Logistic Regression")
        vectors_path = hf_hub_download(repo_id=repo_id, filename="ft_word_vectors.joblib")
        clf_path = hf_hub_download(repo_id=repo_id, filename="ft_clf.joblib")

        self.word_vectors: Dict[str, np.ndarray] = joblib.load(vectors_path)
        self.clf = joblib.load(clf_path)
        self.dim = next(iter(self.word_vectors.values())).shape[0] if self.word_vectors else 300
        self.id2label = id2label or {}

    def _format_label(self, raw_label: Any) -> str:
        return str(self.id2label.get(raw_label, raw_label))

    def _text_to_vector(self, text: str) -> np.ndarray:
        cleaned = clean_classical(text)
        vecs = [self.word_vectors[w] for w in cleaned.split() if w in self.word_vectors]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0, dtype=np.float32)

    def predict(self, text: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        vec = self._text_to_vector(text).reshape(1, -1)
        probs = self.clf.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))

        raw_label = self.clf.classes_[pred_idx]
        confidence = round(float(probs[pred_idx]), 4)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "label": self._format_label(raw_label),
            "confidence": confidence,
            "latency_ms": latency_ms
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        start_time = time.perf_counter()

        matrix = np.vstack([self._text_to_vector(t) for t in texts])
        all_probs = self.clf.predict_proba(matrix)
        pred_indices = np.argmax(all_probs, axis=1)

        total_ms = (time.perf_counter() - start_time) * 1000
        per_item_ms = round(total_ms / len(texts), 2)

        return [
            {
                "model_name": self.model_name,
                "label": self._format_label(self.clf.classes_[idx]),
                "confidence": round(float(probs[idx]), 4),
                "latency_ms": per_item_ms
            }
            for idx, probs in zip(pred_indices, all_probs)
        ]


# ==========================================
# 3. Bi-LSTM (PyTorch) Architecture & Wrapper
# ==========================================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)
        feat = self.dropout(torch.cat([avg_pool, max_pool], dim=1))
        return self.fc(feat)


class BiLstmWrapper(BaseSentimentModel):
    def __init__(
        self,
        repo_id: str = "YossifSamy99/arabic-nlp-models",
        id2label: Optional[Dict[Any, str]] = None
    ):
        super().__init__(model_name="Bi-LSTM (PyTorch)")
        self.device = torch.device("cpu")

        weights_path = hf_hub_download(repo_id=repo_id, filename="bilstm_weights.pt")
        vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.pkl")

        with open(vocab_path, "rb") as f:
            vocab_artifact = pickle.load(f)

        self.word2idx: Dict[str, int] = vocab_artifact["word2idx"]
        self.idx2label: Dict[int, str] = id2label or vocab_artifact["idx2label"]
        cfg = vocab_artifact["config"]
        self.max_len: int = cfg["max_len"]

        self.model = BiLSTMClassifier(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"]
        )
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, text: str) -> List[int]:
        cleaned = clean_classical(text)
        tokens = cleaned.split()[:self.max_len]
        ids = [self.word2idx.get(w, 1) for w in tokens]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return ids

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        tensor_x = torch.tensor([self._encode(text)], dtype=torch.long, device=self.device)
        probs = torch.softmax(self.model(tensor_x), dim=1)[0].numpy()
        pred_idx = int(np.argmax(probs))

        label = str(self.idx2label.get(pred_idx, pred_idx))
        confidence = round(float(probs[pred_idx]), 4)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "label": label,
            "confidence": confidence,
            "latency_ms": latency_ms
        }

    @torch.inference_mode()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        start_time = time.perf_counter()
        encoded_batch = [self._encode(t) for t in texts]
        tensor_x = torch.tensor(encoded_batch, dtype=torch.long, device=self.device)
        all_probs = torch.softmax(self.model(tensor_x), dim=1).numpy()
        pred_indices = np.argmax(all_probs, axis=1)

        total_ms = (time.perf_counter() - start_time) * 1000
        per_item_ms = round(total_ms / len(texts), 2)

        return [
            {
                "model_name": self.model_name,
                "label": str(self.idx2label.get(int(idx), idx)),
                "confidence": round(float(probs[idx]), 4),
                "latency_ms": per_item_ms
            }
            for idx, probs in zip(pred_indices, all_probs)
        ]


# ==========================================
# 4. Fine-tuned CAMeLBERT-Mix Wrapper
# ==========================================
class CamelBertWrapper(BaseSentimentModel):
    def __init__(
        self,
        repo_id: str = "YossifSamy99/arabic-nlp-models",
        id2label: Optional[Dict[Any, str]] = None,
        max_length: int = 128
    ):
        super().__init__(model_name="CAMeLBERT-Mix (Fine-tuned)")
        self.device = torch.device("cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        self.model.to(self.device)
        self.model.eval()

        # تطبيع مفاتيح id2label لأرقام صحيحة لتجنب اختلاف نوع المفتاح في config.json
        config_labels = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.id2label = id2label or config_labels

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        cleaned = clean_transformer(text)
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        pred_idx = int(np.argmax(probs))

        label = str(self.id2label.get(pred_idx, pred_idx))
        confidence = round(float(probs[pred_idx]), 4)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "label": label,
            "confidence": confidence,
            "latency_ms": latency_ms
        }

    @torch.inference_mode()
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        if not texts:
            return []

        start_time = time.perf_counter()
        cleaned_texts = [clean_transformer(t) for t in texts]
        all_probs_list = []

        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            all_probs_list.append(probs)

        all_probs = np.vstack(all_probs_list)
        pred_indices = np.argmax(all_probs, axis=1)

        total_ms = (time.perf_counter() - start_time) * 1000
        per_item_ms = round(total_ms / len(texts), 2)

        return [
            {
                "model_name": self.model_name,
                "label": str(self.id2label.get(int(idx), idx)),
                "confidence": round(float(probs[idx]), 4),
                "latency_ms": per_item_ms
            }
            for idx, probs in zip(pred_indices, all_probs)
        ]