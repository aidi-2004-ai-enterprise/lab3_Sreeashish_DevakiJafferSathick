# Lab 3: Penguin Species Classification API

A machine learning API that predicts penguin species based on physical measurements using XGBoost and FastAPI.

## 🐧 Overview

This project classifies penguins into three species (Adelie, Chinstrap, Gentoo) based on:
- Bill length and depth
- Flipper length  
- Body mass
- Sex and island location

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost seaborn matplotlib pydantic python-multipart
```

### 2. Train the Model
```bash
python train.py
```

### 3. Start the API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API
Open http://localhost:8000/docs in your browser

## 📊 Model Performance

- **Training Accuracy**: 100%
- **Test Accuracy**: 100%
- **F1-Score**: 1.0

## 🧪 API Testing

### Successful Prediction Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "bill_length_mm": 39.1,
         "bill_depth_mm": 18.7,
         "flipper_length_mm": 181.0,
         "body_mass_g": 3750.0,
         "sex": "male",
         "island": "Torgersen"
     }'
```

**Response:**
```json
{
  "predicted_species": "Adelie",
  "confidence": 0.99,
  "probabilities": {
    "Adelie": 0.99,
    "Chinstrap": 0.005,
    "Gentoo": 0.005
  }
}
```

### Error Handling
Invalid inputs return HTTP 422 with clear error messages:
```json
{
  "detail": "Invalid sex value. Must be 'male' or 'female'"
}
```

## 📁 Project Structure
```
├── train.py           # Model training script
├── app/
│   ├── main.py        # FastAPI application
│   └── data/          # Model artifacts (created after training)
│       ├── model.json
│       ├── label_encoder.pkl
│       └── metadata.json
├── pyproject.toml     # Dependencies
└── README.md
```

## 🔧 Valid Input Values

- **sex**: "male" or "female"
- **island**: "Torgersen", "Biscoe", or "Dream"  
- **measurements**: All positive numbers

## 📱 API Endpoints

- **GET /**: Basic API information
- **GET /health**: Health check
- **POST /predict**: Species prediction
- **GET /docs**: Interactive API documentation

## 🎯 Features

- ✅ XGBoost model with overfitting prevention
- ✅ Input validation with Pydantic
- ✅ Comprehensive error handling
- ✅ Detailed logging
- ✅ Interactive API documentation
- ✅ 100% test accuracy

## 👨‍💻 Author

**Sreeashish DevakiJafferSathick**  
AIDI 2004 - AI Enterprise Lab 3