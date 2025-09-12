# 🧬 Cancer Image Classifier (with MLOps & MLflow)

## Overview  
This project builds a full **MLOps pipeline** for training, evaluating, and deploying a deep learning model to classify cancer images (e.g. Adenocarcinoma vs. Normal) using **TensorFlow/Keras**. It integrates modular pipeline stages, MLflow tracking, and a Flask API for real-time predictions.

---

## 🔧 Key Features

- ✅ Image classification using pretrained CNNs (e.g. VGG16)
- ✅ End-to-end training pipeline with DVC or `main.py`
- ✅ Multi-stage modular design (ingestion, training, evaluation)
- ✅ Model logging, scoring, and experiment tracking with **MLflow** + **DagsHub**
- ✅ Live image prediction with a **Flask web app**
- ✅ Ready for Docker/Cloud deployment

---

## 🏗️ Pipeline Architecture

| Stage                   | Script                             | Description |
|------------------------|-------------------------------------|-------------|
| **Data Ingestion**     | `stage_01_data_ingestion.py`       | Downloads and extracts dataset from Google Drive:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}  
| **Model Initialization** | `stage_02_prepare_base_model.py`  | Loads and modifies pretrained CNN base model:contentReference[oaicite:2]{index=2}  
| **Training**           | `stage_03_model_trainer.py`        | Trains model with augmentation and saves weights:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}  
| **Evaluation**         | `stage_04_model_evaluation.py`     | Evaluates model and logs to MLflow:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}  
| **API & Prediction**   | `app.py`, `prediction.py`          | Flask app to handle real-time image classification:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}  

---

## 📁 Project Structure

```
CancerClassifier/
│
├── stage_01_data_ingestion.py
├── stage_02_prepare_base_model.py
├── stage_03_model_trainer.py
├── stage_04_model_evaluation.py
├── model_evaluation_with_mlflow.py
│
├── app.py                  # Flask API
├── prediction.py          # Image preprocessing and prediction
├── main.py                # Orchestrator
│
├── artifacts/             # Contains model, scores, etc.
├── templates/index.html   # Frontend for upload
```

---

## 🧪 How to Run

### 🔁 Full Pipeline

You can either use DVC or run:

```bash
python main.py
```

This will:

- Ingest and unzip data
- Prepare base model
- Train with Keras
- Evaluate & log metrics

### 🚀 Serve API

```bash
python app.py
```

Visit [http://localhost:8080](http://localhost:8080)  
Upload an image (Base64), and receive prediction (`Normal` or `Adenocarcinoma Cancer`)

### 🧬 Predict Programmatically

POST a Base64 string of an image:

```json
POST /predict
{
  "image": "base64_encoded_image_here"
}
```

You’ll receive:

```json
{"image": "Normal"}
```

---

## 📦 Installation

```bash
git clone https://github.com/your-user/cancer-classifier-mlops.git
cd cancer-classifier-mlops
pip install -r requirements.txt
```

Also, configure `.env` for Google Drive access (if needed).

---

## 📊 MLflow Tracking (via DagsHub)

Logs model, accuracy, and loss to MLflow:

- `mlflow.log_metrics()`  
- `mlflow.keras.log_model(...)`  
- Uses `dagshub.init()` to connect with repo:contentReference[oaicite:9]{index=9}

---

## 📈 Model Output

- `model/model.h5`: trained Keras model
- `scores.json`: test accuracy & loss
- `MLflow UI`: experiment runs and metrics

---

## 🧰 Tech Stack

| Layer              | Libraries / Tools                        |
|--------------------|-------------------------------------------|
| Framework          | TensorFlow / Keras                        |
| Serving            | Flask + HTML frontend                     |
| ML Lifecycle       | MLflow + DagsHub                          |
| Data Pipeline      | DVC / custom orchestrator                 |
| Image Ingestion    | Pillow, `keras.preprocessing.image`       |
| Logging            | Custom Python logger                      |

---

## 📄 License

MIT

