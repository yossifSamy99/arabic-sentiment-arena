import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from core.config import (
    TRAINER_MIN_ROWS,
    TRAINER_MAX_ROWS,
    TRAINER_MIN_CLASSES,
    TRAINER_MIN_SAMPLES_PER_CLASS,
)
from core.preprocessing import clean_classical, clean_transformer


class GuardrailError(ValueError):
    """خطأ مخصص يظهر للمستخدم عند مخالفة شروط ملف البيانات."""
    pass


def validate_user_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    if text_col not in df.columns or label_col not in df.columns:
        raise GuardrailError("تأكد من اختيار عمود النص وعمود التصنيف بشكل صحيح.")

    if text_col == label_col:
        raise GuardrailError("لا يمكن اختيار نفس العمود للنص والتصنيف معاً.")

    clean_df = df[[text_col, label_col]].dropna().copy()
    clean_df[text_col] = clean_df[text_col].astype(str).str.strip()
    clean_df[label_col] = clean_df[label_col].astype(str).str.strip()
    clean_df = clean_df[(clean_df[text_col] != "") & (clean_df[label_col] != "")]

    n_rows = len(clean_df)
    if n_rows < TRAINER_MIN_ROWS:
        raise GuardrailError(
            f"عدد الصفوف الصالحة بعد التنظيف ({n_rows}) أقل من الحد الأدنى ({TRAINER_MIN_ROWS} صف)."
        )

    if n_rows > TRAINER_MAX_ROWS:
        raise GuardrailError(
            f"عدد الصفوف ({n_rows}) يتجاوز الحد الأقصى المسموح به ({TRAINER_MAX_ROWS} صف) لحماية الذاكرة."
        )

    class_counts = clean_df[label_col].value_counts()
    if len(class_counts) < TRAINER_MIN_CLASSES:
        raise GuardrailError("يجب أن يحتوي الملف على فئتين (Classes) مختلفتين على الأقل.")

    if class_counts.min() < TRAINER_MIN_SAMPLES_PER_CLASS:
        raise GuardrailError(
            f"أقل فئة تحتوي على {class_counts.min()} عينات فقط؛ الحد الأدنى المطلوب هو {TRAINER_MIN_SAMPLES_PER_CLASS} عينات لكل فئة."
        )

    return clean_df.reset_index(drop=True)


@torch.inference_mode()
def extract_camelbert_embeddings(
    texts: list[str],
    tokenizer,
    encoder_model,
    batch_size: int = 32,
    max_length: int = 128
) -> np.ndarray:
    """
    استخراج متجهات 768-D من CAMeLBERT-Mix المجمد باستخدام Mean Pooling على CPU.
    """
    cleaned_texts = [clean_transformer(t) for t in texts]
    all_embeddings = []

    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        outputs = encoder_model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
        all_embeddings.append(mean_pooled)

    return np.vstack(all_embeddings)


def train_dual_custom_models(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tokenizer,
    encoder_model
) -> Dict[str, Any]:
    """
    تدريب متوازي لنموذج TF-IDF ونموذج CAMeLBERT Embeddings على بيانات المستخدم.
    """
    valid_df = validate_user_dataframe(df, text_col, label_col)

    X_train, X_val, y_train, y_val = train_test_split(
        valid_df[text_col].tolist(),
        valid_df[label_col].tolist(),
        test_size=0.20,
        random_state=42,
        stratify=valid_df[label_col]
    )

    # ------------------------------------------
    # 1. تدريب TF-IDF + Logistic Regression
    # ------------------------------------------
    t0 = time.perf_counter()
    X_train_classical = [clean_classical(t) for t in X_train]
    X_val_classical = [clean_classical(t) for t in X_val]

    tfidf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    tfidf_pipe.fit(X_train_classical, y_train)
    tfidf_train_sec = round(time.perf_counter() - t0, 3)

    t_inf0 = time.perf_counter()
    tfidf_preds = tfidf_pipe.predict(X_val_classical)
    tfidf_latency_ms = round(((time.perf_counter() - t_inf0) / max(len(X_val), 1)) * 1000, 3)

    # ------------------------------------------
    # 2. تدريب Frozen CAMeLBERT Embeddings + Logistic Regression
    # ------------------------------------------
    t1 = time.perf_counter()
    X_train_emb = extract_camelbert_embeddings(X_train, tokenizer, encoder_model)
    X_val_emb = extract_camelbert_embeddings(X_val, tokenizer, encoder_model)

    bert_clf = LogisticRegression(max_iter=1000)
    bert_clf.fit(X_train_emb, y_train)
    bert_train_sec = round(time.perf_counter() - t1, 3)

    t_inf1 = time.perf_counter()
    bert_preds = bert_clf.predict(X_val_emb)
    bert_latency_ms = round(((time.perf_counter() - t_inf1) / max(len(X_val), 1)) * 1000, 3)

    return {
        "dataset_info": {
            "total_rows": len(valid_df),
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "classes": sorted(list(set(y_train)))
        },
        "tfidf": {
            "model_name": "TF-IDF + Logistic Regression",
            "pipeline": tfidf_pipe,
            "accuracy": round(float(accuracy_score(y_val, tfidf_preds)), 4),
            "macro_f1": round(float(f1_score(y_val, tfidf_preds, average="macro")), 4),
            "train_time_sec": tfidf_train_sec,
            "avg_latency_ms": tfidf_latency_ms
        },
        "camelbert_emb": {
            "model_name": "CAMeLBERT-Mix Embeddings + LogReg",
            "classifier": bert_clf,
            "accuracy": round(float(accuracy_score(y_val, bert_preds)), 4),
            "macro_f1": round(float(f1_score(y_val, bert_preds, average="macro")), 4),
            "train_time_sec": bert_train_sec,
            "avg_latency_ms": bert_latency_ms
        }
    }


def predict_custom_tweet(text: str, trained_bundle: Dict[str, Any], tokenizer, encoder_model) -> list[dict]:
    """اختبار تويتة جديدة على النموذجين المتدربين لحظياً."""
    results = []

    # 1. TF-IDF
    t0 = time.perf_counter()
    pipe = trained_bundle["tfidf"]["pipeline"]
    probs_tfidf = pipe.predict_proba([clean_classical(text)])[0]
    idx_tfidf = int(np.argmax(probs_tfidf))
    results.append({
        "model_name": trained_bundle["tfidf"]["model_name"],
        "label": str(pipe.classes_[idx_tfidf]),
        "confidence": round(float(probs_tfidf[idx_tfidf]), 4),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
    })

    # 2. CAMeLBERT Embeddings
    t1 = time.perf_counter()
    clf = trained_bundle["camelbert_emb"]["classifier"]
    emb = extract_camelbert_embeddings([text], tokenizer, encoder_model)
    probs_bert = clf.predict_proba(emb)[0]
    idx_bert = int(np.argmax(probs_bert))
    results.append({
        "model_name": trained_bundle["camelbert_emb"]["model_name"],
        "label": str(clf.classes_[idx_bert]),
        "confidence": round(float(probs_bert[idx_bert]), 4),
        "latency_ms": round((time.perf_counter() - t1) * 1000, 2)
    })

    return results