import os
import gc
import json
import time
import joblib
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from huggingface_hub import HfApi, login

from core.preprocessing import clean_classical

# ==========================================
# 1. تحميل الـ 6 ملفات المتكيشة من مسارك في Colab
# ==========================================
DATA_DIR = "/content"  # عدل المسار لمكان ملفاتك
os.makedirs("artifacts", exist_ok=True)


def load_cached_series(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath)
    return df.iloc[:, -1].reset_index(drop=True)


X_train = load_cached_series(f"{DATA_DIR}/X_train.csv")
y_train = load_cached_series(f"{DATA_DIR}/y_train.csv")

X_valid = load_cached_series(f"{DATA_DIR}/X_valid.csv")
y_valid = load_cached_series(f"{DATA_DIR}/y_valid.csv")

X_test = load_cached_series(f"{DATA_DIR}/X_test.csv")
y_test = load_cached_series(f"{DATA_DIR}/y_test.csv")

X_train_clean = X_train.fillna("").astype(str).apply(clean_classical)
X_valid_clean = X_valid.fillna("").astype(str).apply(clean_classical)
X_test_clean = X_test.fillna("").astype(str).apply(clean_classical)


# ==========================================
# 2. تحميل FastText العربي واستخلاص قاموس الداتا فقط لتقليل الحجم
# ==========================================
# بيحمل cc.ar.300.bin تلقائياً لو مش موجود (حوالي 4.2GB مضغوط)
fasttext.util.download_model("ar", if_exists="ignore")
ft_model = fasttext.load_model("cc.ar.300.bin")
EMBEDDING_DIM = ft_model.get_dimension()  # 300

# استخراج الكلمات الفريدة من بيانات التدريب فقط (لمنع Data Leakage)
# ميزة cc.ar.300.bin إنه بيولّد متجهات Subword حتى للكلمات العامية أو غير الموجودة في القاموس
train_vocab = set(" ".join(X_train_clean).split())
print(f"Unique words in X_train: {len(train_vocab)}")

ft_word_vectors = {
    word: ft_model.get_word_vector(word).astype(np.float32)
    for word in train_vocab
}

# تفريغ الموديل الضخم (7GB) من الرامات فوراً بعد استخلاص القاموس المصغر (~30MB)
del ft_model
gc.collect()


# ==========================================
# 3. تحويل التويتات لمتجهات (متوسط متجهات الكلمات) وتدريب Classifier
# ==========================================
def texts_to_avg_vectors(texts: pd.Series, word_vectors: dict, dim: int = 300) -> np.ndarray:
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, text in enumerate(texts):
        vecs = [word_vectors[w] for w in text.split() if w in word_vectors]
        if vecs:
            matrix[idx] = np.mean(vecs, axis=0)
    return matrix


X_train_vec = texts_to_avg_vectors(X_train_clean, ft_word_vectors, EMBEDDING_DIM)
X_valid_vec = texts_to_avg_vectors(X_valid_clean, ft_word_vectors, EMBEDDING_DIM)
X_test_vec = texts_to_avg_vectors(X_test_clean, ft_word_vectors, EMBEDDING_DIM)

ft_clf = LogisticRegression(max_iter=1000)
ft_clf.fit(X_train_vec, y_train)

# فحص الأداء على Validation Set
y_val_pred = ft_clf.predict(X_valid_vec)
print("\n=== Validation Results (FastText) ===")
print(f"Val Accuracy: {accuracy_score(y_valid, y_val_pred):.4f}")
print(f"Val Macro F1: {f1_score(y_valid, y_val_pred, average='macro'):.4f}")


# ==========================================
# 4. التقييم النهائي على Test Set وحساب Latency
# ==========================================
y_test_pred = ft_clf.predict(X_test_vec)
test_acc = round(float(accuracy_score(y_test, y_test_pred)), 4)
test_macro_f1 = round(float(f1_score(y_test, y_test_pred, average="macro")), 4)

print("\n=== Test Results (Final Benchmark - FastText) ===")
print(classification_report(y_test, y_test_pred))

# قياس متوسط زمن التنبؤ للتويتة الواحدة (تنظيف + تحويل لمتجه + تصنيف) بالمللي ثانية
sample_tweets = X_test.fillna("").astype(str).iloc[:500].tolist()
start_inf = time.perf_counter()
for tweet in sample_tweets:
    cleaned = clean_classical(tweet)
    vecs = [ft_word_vectors[w] for w in cleaned.split() if w in ft_word_vectors]
    avg_vec = np.mean(vecs, axis=0, dtype=np.float32) if vecs else np.zeros(EMBEDDING_DIM, dtype=np.float32)
    _ = ft_clf.predict_proba(avg_vec.reshape(1, -1))
avg_latency_ms = round(((time.perf_counter() - start_inf) / len(sample_tweets)) * 1000, 3)

# تحديث ملف artifacts/benchmark_results.json مع الحفاظ على نتائج TF-IDF
benchmark_path = "artifacts/benchmark_results.json"
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
else:
    benchmark_data = {}

benchmark_data["fasttext"] = {
    "model_name": "FastText + Logistic Regression",
    "accuracy": test_acc,
    "macro_f1": test_macro_f1,
    "avg_latency_ms": avg_latency_ms
}

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, ensure_ascii=False, indent=2)


# ==========================================
# 5. حفظ الملفين ورفعهما على Hugging Face Hub
# ==========================================
REPO_ID = "YossifSamy99/arabic-nlp-models"
VECTORS_FILE = "ft_word_vectors.joblib"
CLF_FILE = "ft_clf.joblib"

joblib.dump(ft_word_vectors, VECTORS_FILE, compress=3)
joblib.dump(ft_clf, CLF_FILE, compress=3)

# login(token="hf_...")  # فعّل السطر ده لو مش مسجل دخول في الجلسة
api = HfApi()
for filename in [VECTORS_FILE, CLF_FILE]:
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="model"
    )

print("\nSaved & Uploaded FastText Successfully:", benchmark_data["fasttext"])