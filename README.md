---
title: Arabic Sentiment Arena
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
models:
  - YossifSamy99/arabic-nlp-models
pinned: false
---

# 🎯 Arabic Sentiment Arena

Interactive benchmark and live inference platform comparing **4 Arabic Sentiment Analysis architectures** on a unified 11,520-tweet evaluation set:

* **CAMeLBERT-Mix (Fine-tuned):** `94.42%` Accuracy | `94.72 ms` CPU latency
* **Bi-LSTM (PyTorch):** `78.26%` Accuracy | `3.65 ms` CPU latency
* **TF-IDF + Logistic Regression:** `78.04%` Accuracy | `0.79 ms` CPU latency
* **FastText + Logistic Regression:** `69.90%` Accuracy | `0.24 ms` CPU latency

### 🔗 Project Links
* **Source Code & Training Pipeline:** [github.com/yossifSamy99/arabic-sentiment-arena](https://github.com/yossifSamy99/arabic-sentiment-arena)
* **Pre-trained Model Weights:** [huggingface.co/YossifSamy99/arabic-nlp-models](https://huggingface.co/YossifSamy99/arabic-nlp-models)