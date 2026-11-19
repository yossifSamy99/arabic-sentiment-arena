# 🎯 Arabic Sentiment Arena

An end-to-end Arabic Sentiment Analysis benchmarking and inference platform comparing **4 distinct NLP paradigms**—from classical linear models to fine-tuned Arabic Transformers—evaluated on a unified test set of **11,520 Arabic tweets**.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/YossifSamy99/arabic-sentiment-arena)
[![Model Weights](https://img.shields.io/badge/🤗%20Model%20Registry-YossifSamy99%2Farabic--nlp--models-green)](https://huggingface.co/YossifSamy99/arabic-nlp-models)

---

## 📊 Unified Benchmark Results

Trained and evaluated on **~58,000 Arabic tweets** (`80% Train / 10% Validation / 10% Test` split, 11,520 test samples) spanning Egyptian, Gulf, Levantine, and Modern Standard Arabic (MSA):

| Rank | Model Architecture | Paradigm | Accuracy | Macro F1 | CPU Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | **CAMeLBERT-Mix (Fine-tuned)** | Transformer (`CAMeL-Lab/bert-base-arabic-camelbert-mix`) | **94.42%** | **0.9442** | 94.72 ms |
| **2** | **Bi-LSTM (PyTorch)** | Deep Learning (Pre-trained Arabic Embeddings) | **78.26%** | **0.7823** | 3.65 ms |
| **3** | **TF-IDF + Logistic Regression** | Classical Sparse (`(1, 2)` N-grams) | **78.04%** | **0.7802** | 0.79 ms |
| **4** | **FastText + Logistic Regression** | Dense Static Embeddings (Mean Pooling) | **69.90%** | **0.6989** | **0.24 ms** |

### Key Engineering Takeaways
* **Why TF-IDF outperformed FastText (+8.14%):** Mean-pooling static word vectors washes out word order and negations (`"مش"` + `"ممتاز"`), whereas `(1, 2)` TF-IDF n-grams preserve negation bigrams (`"مش حلو"`) as explicit discriminative features.
* **Why CAMeLBERT jumped +16.16%:** Dialectal pre-training combined with self-attention and emoji preservation captures sarcasm and multi-dialect context that models trained from scratch cannot learn from 40K tweets alone.
* **Accuracy vs. Latency Trade-off:** `TF-IDF` processes >1,200 tweets/sec on CPU at ~78% accuracy, while `CAMeLBERT-Mix` delivers 94.42% accuracy under 95 ms/tweet on CPU.

---

## 🏗️ System Architecture

The codebase decouples the **ML inference/training engine (`core/`)** from the **Gradio UI (`app.py`)** and **model artifacts (Hugging Face Hub)**:

1. **Unified Predictor Interface (`core/base.py` & `core/wrappers.py`):** All four heterogeneous frameworks (`Scikit-Learn`, `FastText`, `PyTorch`, `Transformers`) implement a standardized base contract returning predicted labels, confidence scores, and exact CPU inference latency.
2. **Dual Preprocessing Pipelines (`core/preprocessing.py`):**
   * `clean_classical`: Strips URLs, mentions, and emojis while explicitly preserving Arabic negation particles (`مش`, `لا`, `لن`, `لم`, `غير`) so linear and recurrent models retain polarity flips.
   * `clean_transformer`: Retains emojis and punctuation marks (`😍`, `😡`, `🙄`, `!`) as high-signal sentiment cues for `CAMeLBERT-Mix`.
3. **Interactive Application Modules (`app.py` & `core/live_trainer.py`):**
   * **⚔️ Live Arena:** Real-time 4-model head-to-head inference on single tweets plus batch CSV scoring.
   * **📊 Benchmark Dashboard:** Interactive accuracy, macro-F1, and latency trade-off visualizations.
   * **⚡ Custom Live Trainer:** Guardrailed on-the-fly training and evaluation comparing `TF-IDF + LR` vs. frozen `CAMeLBERT Embeddings + LR` on user-uploaded Arabic CSV datasets.

---

## 📁 Repository Structure

```text
arabic-sentiment-arena/
├── artifacts/
│   └── benchmark_results.json                  # Unified test set metrics across all 4 models
├── core/
│   ├── __init__.py
│   ├── base.py                                 # Abstract base class for unified model inference
│   ├── config.py                               # Hub repo IDs, model registry, and trainer guardrails
│   ├── live_trainer.py                         # CSV validation & on-the-fly dual model trainer
│   ├── preprocessing.py                        # Separate classical & transformer text pipelines
│   └── wrappers.py                             # Model loaders & inference wrappers for all 4 paradigms
├── hotel_dataset_for_trainig_testing/
│   ├── 1000_hotel_samples_for_testing.csv      # Ready-to-use sample CSV for Batch Prediction
│   └── 5000_hotel_samples_for_training.csv     # Domain-shifted Arabic dataset for Custom Live Trainer
├── training/
│   ├── train_bilstm_colab.py                   # PyTorch Bi-LSTM training & evaluation script
│   ├── train_camelbert_colab.py                # CAMeLBERT-Mix fine-tuning script (T4 GPU)
│   ├── train_fasttext_colab.py                 # FastText + Logistic Regression training script
│   └── train_tfidf_colab.py                    # TF-IDF (1,2)-gram + Logistic Regression training script
├── app.py                                      # Gradio UI (Dark Theme, RTL layout, Plotly charts)
├── README.md
└── requirements.txt                            # Locked dependency tree
```

---

## 🧪 Ready-to-Use Sample Datasets

To test domain adaptation on out-of-distribution Arabic reviews (Hotel Reviews) directly in the app:
* Upload `hotel_dataset_for_trainig_testing/5000_hotel_samples_for_training.csv` inside the **⚡ Custom Trainer** tab to train and benchmark custom models on the fly.
* Upload `hotel_dataset_for_trainig_testing/1000_hotel_samples_for_testing.csv` inside the **⚔️ Live Arena (Batch)** accordion for bulk inference.

---

## 🚀 Run Locally

```bash
git clone [https://github.com/yossifSamy99/arabic-sentiment-arena.git](https://github.com/yossifSamy99/arabic-sentiment-arena.git)
cd arabic-sentiment-arena
pip install -r requirements.txt
python app.py
```

Pre-trained model weights are fetched automatically on first run from [`YossifSamy99/arabic-nlp-models`](https://huggingface.co/YossifSamy99/arabic-nlp-models) and cached in memory.