import os
import json
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
    # يأخذ آخر عمود لتجنب عمود الـ index الإضافي تلقائياً إن وُجد
    return df.iloc[:, -1].reset_index(drop=True)


X_train = load_cached_series(f"{DATA_DIR}/X_train.csv")
y_train = load_cached_series(f"{DATA_DIR}/y_train.csv")

X_valid = load_cached_series(f"{DATA_DIR}/X_valid.csv")
y_valid = load_cached_series(f"{DATA_DIR}/y_valid.csv")

X_test = load_cached_series(f"{DATA_DIR}/X_test.csv")
y_test = load_cached_series(f"{DATA_DIR}/y_test.csv")

print(f"Loaded -> Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

# تنظيف النصوص مع fillna("") للحفاظ على تطابق ترتيب الصفوف مع y بدون أي إزاحة
X_train_clean = X_train.fillna("").astype(str).apply(clean_classical)
X_valid_clean = X_valid.fillna("").astype(str).apply(clean_classical)
X_test_clean = X_test.fillna("").astype(str).apply(clean_classical)


# ==========================================
# 2. تدريب TF-IDF + Logistic Regression
# ==========================================
tfidf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
    ("clf", LogisticRegression(max_iter=1000))
])

tfidf_pipeline.fit(X_train_clean, y_train)

# فحص سريع على Validation Set
y_val_pred = tfidf_pipeline.predict(X_valid_clean)
print("\n=== Validation Results ===")
print(f"Val Accuracy: {accuracy_score(y_valid, y_val_pred):.4f}")
print(f"Val Macro F1: {f1_score(y_valid, y_val_pred, average='macro'):.4f}")


# ==========================================
# 3. التقييم النهائي على Test Set وحساب Latency
# ==========================================
y_test_pred = tfidf_pipeline.predict(X_test_clean)
test_acc = round(float(accuracy_score(y_test, y_test_pred)), 4)
test_macro_f1 = round(float(f1_score(y_test, y_test_pred, average="macro")), 4)

print("\n=== Test Results (Final Benchmark) ===")
print(classification_report(y_test, y_test_pred))

# قياس متوسط زمن التنبؤ للتويتة الواحدة (تنظيف + استدلال) بالمللي ثانية
sample_tweets = X_test.fillna("").astype(str).iloc[:500].tolist()
start_inf = time.perf_counter()
for tweet in sample_tweets:
    _ = tfidf_pipeline.predict_proba([clean_classical(tweet)])
avg_latency_ms = round(((time.perf_counter() - start_inf) / len(sample_tweets)) * 1000, 3)

# تحديث ملف artifacts/benchmark_results.json
benchmark_path = "artifacts/benchmark_results.json"
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
else:
    benchmark_data = {}

benchmark_data["tfidf"] = {
    "model_name": "TF-IDF + Logistic Regression",
    "accuracy": test_acc,
    "macro_f1": test_macro_f1,
    "avg_latency_ms": avg_latency_ms
}

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, ensure_ascii=False, indent=2)


# ==========================================
# 4. حفظ الأوزان ورفعها على Hugging Face Hub
# ==========================================
REPO_ID = "YossifSamy99/arabic-nlp-models"
MODEL_FILE = "tfidf_pipeline.joblib"

joblib.dump(tfidf_pipeline, MODEL_FILE, compress=3)

# login(token="hf_...")  # فعّل السطر ده لو مش مسجل دخول في Colab
api = HfApi()
api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=REPO_ID,
    repo_type="model"
)

print("\nSaved & Uploaded Successfully:", benchmark_data["tfidf"])