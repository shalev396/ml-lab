# 🧪 ml-lab

My AI lab: every machine-learning model I've built, end to end, in one place. Each project has the
training notebook that builds it, the saved model (a Hugging Face model repo), and a live Gradio app
with a public API (a Hugging Face Space).

- **Runs anywhere**: every notebook picks the device it finds (CUDA GPU → Apple MPS → CPU), so the
  same code runs on a laptop, a GPU box or Google Colab.
- **One environment**: [`requirements.txt`](requirements.txt) covers every project (latest versions).
- **Same shape everywhere**: `<project>/model` · `<project>/space` · `<project>/training`. See
  [ADDING_A_PROJECT.md](ADDING_A_PROJECT.md).

```bash
git clone --recurse-submodules https://github.com/shalev396/ml-lab && cd ml-lab
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook <project>/training/notebook.ipynb  # or open it in Colab from the project page
```

---

## 👁️ Vision

### 🍔 [FoodVision Big](foodvision-big/)
`7.8M params` · `PyTorch` · `Food-101` · **87.46% top-1 accuracy**
Sorts a food photo into one of 101 dishes and shows the top 5. It is a fine-tuned EfficientNet-B2.

### 🍣 [FoodVision Mini](foodvision-mini/)
`7.7M params` · `PyTorch` · `Food-101 subset` · **96.0% accuracy**
Tells pizza, steak and sushi apart. The ImageNet EfficientNet-B2 backbone is frozen and only a small new head is trained.

### 🎂 [Age & Gender Estimator](age-estimator/)
`7.8M params` · `PyTorch` · `UTKFace` · **age MAE 4.59 years · gender 93.3%**
Finds every face in a photo (MTCNN), then estimates age and gender (EfficientNet-B2). Grad-CAM heatmaps show what the model looked at.

### 🧑 [Face Recognition](face-recognition/)
`22.8M params` · `PyTorch` · `LFW (42 people) + Open Images` · **99.69% top-1 accuracy**
Its own face detector finds and aligns the face, two embedders trained here turn it into an embedding, and an MLP names who it is. It can also check whether two photos show the same person.

### 👗 [Fashion Image Search](fashion-image-search/)
`2.26M params` · `TensorFlow/Keras` · `Fashion Product Images` · **P@5 0.97 (category) / 0.73 (article type)**
Upload a clothing photo and get the most similar products from an 8,000-item catalog. It uses MobileNetV2 embeddings and nearest-neighbour search.

## 📝 Language

### 📬 [Email Spam Classifier](email-spam-classifier/)
`66.4M params` · `PyTorch + Transformers` · `Enron-Spam` · **F1 0.994**
Paste an email and a fine-tuned DistilBERT decides whether it belongs in the inbox or the spam folder.

### 📰 [Fake News Detector](fake-news-detector/)
`2.1M params` · `TensorFlow/Keras` · `Fake & Real News` · **98.5% accuracy**
A bidirectional LSTM reads a headline and article and says how likely it is to be fake. It beat SimpleRNN, LSTM and GRU versions trained on the same data.

### 💬 [FLAN-T5 Dialogue Summarizer + RAG](flan-t5-dialogue-summarizer/)
`248M params (1.8M LoRA)` · `PyTorch + PEFT` · `DialogSum` · **ROUGE-L 35.3 (+14.6 over base)**
Summarizes conversations with a LoRA-tuned FLAN-T5 and answers store questions from a small knowledge base (RAG).

### 🎭 [Tiny Shakespeare Chat](tiny-shakespeare-chat/)
`10.75M params` · `PyTorch` · `Tiny Shakespeare` · **held-out loss 1.495 nats/char**
A GPT written and trained from scratch, one character at a time. It is chat-tuned so it answers you in Shakespearean verse.

## 🎬 Audio & video apps
Pipelines built from pretrained models. Nothing is trained here.

### 🎬 [AI Video Summarizer](video-summarizer/)
`Whisper + Qwen2.5` · `PyTorch + Transformers` · `sample clips` · **WER 0.00 on the JFK clip**
Upload a video or audio clip (or paste a transcript) and get a timestamped transcript, a summary and 5 key points.

### 📝 [AI Quiz Generator](video-quiz-generator/)
`Whisper + Qwen2.5` · `PyTorch + Transformers` · `JFK clip + Gettysburg Address` · **schema-valid 100% (6/6) · answer agreement 0.75**
Turns a video, audio file or transcript into a multiple-choice quiz in strict, schema-checked JSON.

## 📊 Tabular

### 💳 [Credit Card Fraud Detector](credit-card-fraud/)
`Random forest + SMOTE` · `scikit-learn + TensorFlow` · `ULB Credit Card Fraud` · **PR-AUC 0.80**
Scores card transactions for fraud risk. The best of 12 models is chosen on validation PR-AUC.

### 🛒 [Purchase Propensity + RFM](purchase-propensity/)
`24 params (logistic regression)` · `scikit-learn + PyTorch` · `Customer propensity + Online Retail` · **ROC-AUC 0.997**
Predicts whether a website visit ends in a purchase, and groups retail customers into RFM segments.

### 🏠 [California Housing Prices](california-housing/)
`XGBoost` · `XGBoost + PyTorch` · `California Housing` · **RMSE $44k · R² 0.851**
Predicts a district's median house value from 8 census features, comparing four regressors.

### 🛗 [Elevator Predictive Maintenance](elevator-maintenance/)
`Random forest, 300 trees` · `scikit-learn + TensorFlow` · `synthetic sensor data` · **F1 0.988**
Uses the last hour of 11 sensor readings to warn that an elevator will fail within 10 minutes.

## 📈 Time series

### 📈 [Stock Price LSTM](stock-lstm-forecast/)
`16,961 params` · `TensorFlow/Keras` · `AAPL daily closes` · **MAPE 1.24% (naive: RMSE 4.469)**
An LSTM forecasts Apple's price from any cutoff date. It honestly does not beat the "tomorrow = today" baseline.

### 📊 [EU Stock Market Forecasting](eu-stock-forecasting/)
`4 params (Holt-Winters; VAR(10) on CAC)` · `statsmodels (+ Keras compared)` · `EuStockMarkets 1991–1998` · **DAX MAPE 1.047% (naive 1.048%)**
Forecasts the DAX, SMI, CAC and FTSE from any cutoff date with the simplest model that matches the best one on validation. It honestly ties "tomorrow = today".
