import os
import json
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report
from huggingface_hub import HfApi, login

from core.preprocessing import clean_classical

# ==========================================
# 1. إعداد الجهاز وتحميل الـ 6 ملفات المتكيشة
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Bi-LSTM on device: {DEVICE}")

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

# توحيد الـ Labels لأرقام (0, 1, ...) وحفظ الخريطة العكسية
unique_labels = sorted(y_train.unique().tolist())
label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
idx2label = {idx: str(lbl) for lbl, idx in label2idx.items()}

y_train_idx = y_train.map(label2idx).values
y_valid_idx = y_valid.map(label2idx).values
y_test_idx = y_test.map(label2idx).values


# ==========================================
# 2. بناء الـ Vocabulary ومصفوفة أوزان FastText الأولية
# ==========================================
MAX_LEN = 64
MIN_FREQ = 2
EMBED_DIM = 300

word_counts = Counter(" ".join(X_train_clean).split())
word2idx = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= MIN_FREQ:
        word2idx[word] = len(word2idx)

VOCAB_SIZE = len(word2idx)
print(f"Vocabulary Size (min_freq={MIN_FREQ}): {VOCAB_SIZE}")

# تحميل متجهات FastText لبناء مصفوفة Initial Weights
embedding_matrix = np.random.normal(scale=0.1, size=(VOCAB_SIZE, EMBED_DIM)).astype(np.float32)
embedding_matrix[0] = np.zeros(EMBED_DIM, dtype=np.float32)  # <PAD> = أصفار

if os.path.exists("ft_word_vectors.joblib"):
    print("Loading cached ft_word_vectors.joblib...")
    ft_vectors = joblib.load("ft_word_vectors.joblib")
    hits = 0
    for word, idx in word2idx.items():
        if word in ft_vectors:
            embedding_matrix[idx] = ft_vectors[word]
            hits += 1
    print(f"Matched {hits}/{VOCAB_SIZE} words from cached FastText vectors.")
else:
    import fasttext
    import fasttext.util
    fasttext.util.download_model("ar", if_exists="ignore")
    ft_model = fasttext.load_model("cc.ar.300.bin")
    for word, idx in word2idx.items():
        if idx >= 2:
            embedding_matrix[idx] = ft_model.get_word_vector(word).astype(np.float32)
    del ft_model


# ==========================================
# 3. تجهيز Dataset و DataLoader
# ==========================================
def encode_text(text: str, word2idx: dict, max_len: int) -> list:
    tokens = text.split()[:max_len]
    ids = [word2idx.get(w, 1) for w in tokens]  # 1 = <UNK>
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))       # 0 = <PAD>
    return ids


class TweetDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: np.ndarray, word2idx: dict, max_len: int):
        self.X = np.array([encode_text(t, word2idx, max_len) for t in texts], dtype=np.int64)
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


train_loader = DataLoader(TweetDataset(X_train_clean, y_train_idx, word2idx, MAX_LEN), batch_size=128, shuffle=True)
valid_loader = DataLoader(TweetDataset(X_valid_clean, y_valid_idx, word2idx, MAX_LEN), batch_size=256, shuffle=False)
test_loader = DataLoader(TweetDataset(X_test_clean, y_test_idx, word2idx, MAX_LEN), batch_size=256, shuffle=False)


# ==========================================
# 4. بناء معمارية Bi-LSTM وتدريبها على T4 GPU
# ==========================================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: np.ndarray = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        # دمج Mean Pooling + Max Pooling لاتجاهي الـ Bi-LSTM (hidden_dim * 2 * 2)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))       # (B, L, E)
        out, _ = self.lstm(emb)                     # (B, L, H*2)
        avg_pool = torch.mean(out, dim=1)           # (B, H*2)
        max_pool, _ = torch.max(out, dim=1)         # (B, H*2)
        feat = self.dropout(torch.cat([avg_pool, max_pool], dim=1))
        return self.fc(feat)


HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_CLASSES = len(unique_labels)

model = BiLSTMClassifier(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pretrained_embeddings=embedding_matrix
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 8
best_val_f1 = 0.0
best_weights = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    # التقييم على Validation Set لاختيار أفضل نسخة
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_x, _ in valid_loader:
            logits = model(batch_x.to(DEVICE))
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    val_acc = accuracy_score(y_valid_idx, val_preds)
    val_f1 = f1_score(y_valid_idx, val_preds, average="macro")
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

# تحميل أفضل أوزان
model.load_state_dict(best_weights)


# ==========================================
# 5. التقييم النهائي على Test Set وحساب Latency على CPU
# ==========================================
model.to(DEVICE)
model.eval()
test_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        logits = model(batch_x.to(DEVICE))
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

test_acc = round(float(accuracy_score(y_test_idx, test_preds)), 4)
test_macro_f1 = round(float(f1_score(y_test_idx, test_preds, average="macro")), 4)

print("\n=== Test Results (Final Benchmark - Bi-LSTM) ===")
print(classification_report(y_test_idx, test_preds, target_names=[idx2label[i] for i in range(NUM_CLASSES)]))

# نقل الموديل لـ CPU لقياس زمن الاستجابة الحقيقي المطابق لسيرفر HF Space
model_cpu = model.to("cpu")
model_cpu.eval()
sample_tweets = X_test.fillna("").astype(str).iloc[:500].tolist()

start_inf = time.perf_counter()
with torch.inference_mode():
    for tweet in sample_tweets:
        cleaned = clean_classical(tweet)
        tensor_x = torch.tensor([encode_text(cleaned, word2idx, MAX_LEN)], dtype=torch.long)
        probs = torch.softmax(model_cpu(tensor_x), dim=1)
avg_latency_ms = round(((time.perf_counter() - start_inf) / len(sample_tweets)) * 1000, 3)

# تحديث ملف artifacts/benchmark_results.json
benchmark_path = "artifacts/benchmark_results.json"
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
else:
    benchmark_data = {}

benchmark_data["bilstm"] = {
    "model_name": "Bi-LSTM (PyTorch)",
    "accuracy": test_acc,
    "macro_f1": test_macro_f1,
    "avg_latency_ms": avg_latency_ms
}

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, ensure_ascii=False, indent=2)


# ==========================================
# 6. حفظ bilstm_weights.pt و vocab.pkl ورفعهما للـ Hub
# ==========================================
REPO_ID = "YossifSamy99/arabic-nlp-models"
WEIGHTS_FILE = "bilstm_weights.pt"
VOCAB_FILE = "vocab.pkl"

torch.save(best_weights, WEIGHTS_FILE)

vocab_artifact = {
    "word2idx": word2idx,
    "idx2label": idx2label,
    "config": {
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "max_len": MAX_LEN
    }
}
with open(VOCAB_FILE, "wb") as f:
    pickle.dump(vocab_artifact, f)

# login(token="hf_...")  # فعّل السطر ده لو مش مسجل دخول في الجلسة
api = HfApi()
for filename in [WEIGHTS_FILE, VOCAB_FILE]:
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="model"
    )

print("\nSaved & Uploaded Bi-LSTM Successfully:", benchmark_data["bilstm"])