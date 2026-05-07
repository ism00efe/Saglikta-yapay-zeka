# Genetic Variant Analysis Project

This repository contains an end-to-end machine learning pipeline for classifying genetic mutations (missense variants) as Pathogenic or Benign. 

The project implements a hybrid feature extraction architecture that combines biological rules with state-of-the-art Protein Language Models (ESM2), followed by a hyperparameter-optimized Gradient Boosting classifier. It includes model serving via a REST API and an interactive web interface.

## Project Scope

Primary objective: Predict the pathogenicity of amino acid substitutions based on contextual embeddings and biochemical property shifts.

Implemented capabilities:
- Custom biochemical feature extraction (21-dimensional physical property deltas)
- Integration with Hugging Face `facebook/esm2_t6_8M_UR50D` for contextual protein sequence embeddings (320-dimensional)
- Elimination of train/serving data skew via unified feature concatenation
- XGBoost model training with Bayesian hyperparameter optimization (`Optuna`)
- Online inference via FastAPI and Streamlit

## Repository Structure
```text
.
├── data/
│   ├── raw/                 # Raw ClinVar datasets
│   └── processed/           # Extracted features (biochemical + ESM2 embeddings)
├── models/
│   ├── hybrid_xgboost_v1.pkl # Trained and optimized model artifact
│   └── confusion_matrix.png
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI service
│   ├── app/
│   │   └── app.py           # Streamlit interface
│   ├── features/
│   │   └── preprocessor.py  # Biochemical property translation logic
│   └── models/
│       └── train_model.py   # Training entry point & Optuna optimization
├── requirements.txt
└── README.md
```

## Modeling Pipeline

Training pipeline is implemented in `src/models/train_model.py`.

- **Feature Pipeline**: 341-dimensional array (`[Biochemical Features (21)]` + `[ESM2 [CLS] Token Embedding (320)]`)
- **Base Estimator**: `XGBClassifier`
- **Class Imbalance Handling**: Dynamic `scale_pos_weight` calculation
- **Hyperparameter Search**: `Optuna` (Bayesian Optimization) with Stratified K-Fold CV
- **Optimization Metric**: Weighted F1-Score
- **Saved Artifact**: `models/hybrid_xgboost_v1.pkl`

## Installation

### 1) Clone repository
```bash
git clone <your-repository-url>
cd genetic-variant-analysis
```

### 2) Create virtual environment

```bash
python -m venv venv
```

Activate environment:
- **Windows (PowerShell)**:
```powershell
.\venv\Scripts\Activate.ps1
```
- **Linux / macOS**:
```bash
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Training Workflow

Run model training and hyperparameter optimization:
```bash
python src/models/train_model.py
```

What this run does:
- Loads pre-extracted features
- Executes Optuna trials to find optimal tree depth, learning rate, and estimators
- Trains the final XGBoost model using the best parameters
- Evaluates against the test set and exports the `.pkl` artifact

## Run FastAPI Service

Start the backend inference server:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Available endpoints:
- `GET /` (Health check & GPU status)
- `POST /predict`
- Interactive API docs: `http://127.0.0.1:8000/docs`

### Example prediction request
```bash
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
  -H "Content-Type: application/json" \
  -d '{
    "aa_ref": "A",
    "aa_alt": "V",
    "gene_symbol": "BRCA1",
    "numeric_features": [89.1, 1.8, 88.6, 6.0, 0, 0, 0, 117.1, 4.2, 140.0, 5.96, 0, 0, 0, 28.0, 2.4, 51.4, -0.04, 0, 0, 0]
  }'
```

Expected response format:
```json
{
  "prediction": "Benign",
  "confidence": 0.776,
  "status": "success"
}
```

## Run Streamlit App

Start the interactive frontend:

```bash
streamlit run src/app/app.py
```
The application will securely send the user's variant inputs to the local FastAPI service, extract the required ESM2 embeddings on the fly, and display the confidence score dynamically.

## Future Improvements

- Containerize the API and UI services using `Docker` and `docker-compose`.
- Implement SHAP (SHapley Additive exPlanations) values to interpret whether the model relies more on biochemical changes or ESM2 embeddings for specific predictions.
- Add comprehensive PyTest coverage for the `MutationPreprocessor` logic.
```
