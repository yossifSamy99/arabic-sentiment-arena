import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from huggingface_hub import login

from core.preprocessing import clean_transformer

# ==========================================
# 1. تحميل الـ 6 ملفات المتكيشة وتنظيفها بـ clean_transformer
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

X_train_clean = X_train.fillna("").astype(str).apply(clean_transformer).tolist()
X_valid_clean = X_valid.fillna("").astype(str).apply(clean_transformer).tolist()
X_test_clean = X_test.fillna("").astype(str).apply(clean_transformer).tolist()

# توحيد الـ Labels لأرقام وحفظ القاموسين
unique_labels = sorted(y_train.unique().tolist())
label2id = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
id2label = {idx: str(lbl) for idx, lbl in enumerate(unique_labels)}

y_train_idx = y_train.astype(str).map(label2id).tolist()
y_valid_idx = y_valid.astype(str).map(label2id).tolist()
y_test_idx = y_test.astype(str).map(label2id).tolist()


# ==========================================
# 2. تجهيز Tokenizer و Dataset (max_length=128)
# ==========================================
MODEL_CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

train_encodings = tokenizer(X_train_clean, truncation=True, max_length=MAX_LENGTH)
valid_encodings = tokenizer(X_valid_clean, truncation=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(X_test_clean, truncation=True, max_length=MAX_LENGTH)


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


train_dataset = SentimentDataset(train_encodings, y_train_idx)
valid_dataset = SentimentDataset(valid_encodings, y_valid_idx)
test_dataset = SentimentDataset(test_encodings, y_test_idx)


# ==========================================
# 3. تحميل CAMeLBERT-Mix وتدريبه 3 Epochs على T4 GPU
# ==========================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


training_args = TrainingArguments(
    output_dir="./camelbert_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=360,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()


# ==========================================
# 4. التقييم النهائي على Test Set وحساب Latency على CPU
# ==========================================
test_output = trainer.predict(test_dataset)
test_preds = np.argmax(test_output.predictions, axis=-1)

test_acc = round(float(accuracy_score(y_test_idx, test_preds)), 4)
test_macro_f1 = round(float(f1_score(y_test_idx, test_preds, average="macro")), 4)

print("\n=== Test Results (Final Benchmark - CAMeLBERT-Mix) ===")
print(classification_report(y_test_idx, test_preds, target_names=[id2label[i] for i in range(len(unique_labels))]))

# نقل الموديل إلى CPU لقياس زمن الاستجابة الفعلي المطابق لسيرفر HF Space
model_cpu = trainer.model.to("cpu")
model_cpu.eval()

sample_tweets = X_test.fillna("").astype(str).iloc[:200].tolist()
start_inf = time.perf_counter()
with torch.inference_mode():
    for tweet in sample_tweets:
        cleaned = clean_transformer(tweet)
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )
        _ = torch.softmax(model_cpu(**inputs).logits, dim=-1)
avg_latency_ms = round(((time.perf_counter() - start_inf) / len(sample_tweets)) * 1000, 3)

# تحديث ملف artifacts/benchmark_results.json النهائي للـ 4 نماذج
benchmark_path = "artifacts/benchmark_results.json"
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
else:
    benchmark_data = {}

benchmark_data["camelbert"] = {
    "model_name": "CAMeLBERT-Mix (Fine-tuned)",
    "accuracy": test_acc,
    "macro_f1": test_macro_f1,
    "avg_latency_ms": avg_latency_ms
}

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, ensure_ascii=False, indent=2)


# ==========================================
# 5. رفع الموديل والـ Tokenizer مباشرة إلى Hugging Face Hub
# ==========================================
REPO_ID = "YossifSamy99/arabic-nlp-models"

# login(token="hf_...")  # فعّل السطر ده لو مش مسجل دخول في الجلسة
model_cpu.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)

print("\nSaved & Uploaded CAMeLBERT-Mix Successfully:", benchmark_data["camelbert"])
print("\n=== Complete benchmark_results.json ===")
print(json.dumps(benchmark_data, ensure_ascii=False, indent=2))