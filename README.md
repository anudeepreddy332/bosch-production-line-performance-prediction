# Bosch Production Line Failure Detection (Production ML System)

## 🚀 Overview
This project simulates a real-world production ML system for detecting manufacturing failures in highly imbalanced industrial data (~0.58% failure rate).

Unlike leaderboard-focused solutions, this system is designed for:
- Real-world deployment
- Business decision-making
- Cost-sensitive optimization
- Monitoring and drift detection

---

## 🧠 Key Philosophy

We intentionally moved away from pure MCC optimization.

Why?

Top Kaggle solutions (~0.52 MCC) rely on:
- Data leakage
- Future information
- Non-deployable tricks

This project focuses on:
✅ Reproducibility  
✅ Leakage-safe modeling  
✅ Production reliability  

---

## 🏗️ Architecture

### 🔵 Production Pipeline (`main` branch)
- Decision system (threshold + cost optimization)
- Batch simulation (streaming-like behavior)
- Drift detection (Evidently)
- API (FastAPI)
- Dashboard (Streamlit)

### 🟢 Training Pipeline (`training-pipeline` branch)
- Data ingestion (CSV → Parquet, memory safe)
- Feature engineering (baseline + G + H)
- Chunk-aware CV (leakage-safe)
- LightGBM training
- Meta-model stacking
- OOF prediction generation

---

## 🔁 End-to-End Flow

Raw CSV → Parquet → Features → Models → OOF predictions
↓
Meta model
↓
Production system (thresholding + cost + simulation)
↓
Dashboard + API + Monitoring

---

## 📊 Key Results

- Best MCC: ~0.317
- Recall @ 10% inspection: ~0.63
- Precision: low (expected due to imbalance)
- Fully production-safe pipeline

---

## ⚙️ How to Run

### Training (separate branch)
```bash
git checkout training-pipeline
python scripts/prepare_data.py --zip-path ~/Downloads/bosch-production-line-performance.zip
python scripts/train_meta_model.py
````

### Production

```bash
git checkout main
python scripts/run_full_system.py
streamlit run apps/streamlit_dashboard/app.py
```

---

## 📈 Dashboard Features

* Threshold tuning
* Inspection budget simulation
* Recall vs precision trade-offs
* Cost optimization
* Failure analysis

---

## 🧠 Business Interpretation

This system answers:

* How many failures are we catching?
* What is the inspection cost?
* What is the optimal threshold?
* What happens if we increase recall?

---

## 🛠️ Tech Stack

* Python, Pandas, LightGBM
* Streamlit (dashboard)
* FastAPI (serving)
* Evidently (monitoring)
* Docker (deployment-ready)

---

## 🔮 Future Improvements

* Real-time streaming (Kafka)
* Automated retraining
* Cloud deployment (AWS/GCP)
* Model registry

---

## 👨‍💻 Author

Anudeep Reddy Mutyala