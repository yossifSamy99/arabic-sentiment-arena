REPO_ID = "YossifSamy99/arabic-nlp-models"
BASE_CAMELBERT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
BENCHMARK_FILE_PATH = "artifacts/benchmark_results.json"

MODEL_OPTIONS = {
    "tfidf": "TF-IDF + Logistic Regression",
    "fasttext": "FastText + Logistic Regression",
    "bilstm": "Bi-LSTM (PyTorch)",
    "camelbert": "CAMeLBERT-Mix (Fine-tuned)",
}

# حدود حماية السيرفر (16GB RAM CPU) في Live Arena و Live Trainer
MAX_BATCH_PREDICT_ROWS = 1000
TRAINER_MIN_ROWS = 50
TRAINER_MAX_ROWS = 5000
TRAINER_MIN_CLASSES = 2
TRAINER_MIN_SAMPLES_PER_CLASS = 3