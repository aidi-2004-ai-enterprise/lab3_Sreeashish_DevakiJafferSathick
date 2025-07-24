import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any
from enum import Enum

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)


class Island(str, Enum):
    """Enum for valid island values."""
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"


class Sex(str, Enum):
    """Enum for valid sex values."""
    Male = "male"  
    Female = "female"


class PenguinFeatures(BaseModel):
    """
    Pydantic model for penguin feature input validation.
    
    This model ensures that all required features are provided and that
    categorical features are restricted to valid values seen during training.
    """
    bill_length_mm: float = Field(..., gt=0, description="Bill length in millimeters")
    bill_depth_mm: float = Field(..., gt=0, description="Bill depth in millimeters") 
    flipper_length_mm: float = Field(..., gt=0, description="Flipper length in millimeters")
    body_mass_g: float = Field(..., gt=0, description="Body mass in grams")
    sex: Sex = Field(..., description="Sex of the penguin")
    island: Island = Field(..., description="Island where penguin was observed")

    class Config:
        schema_extra = {
            "example": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "sex": "male",
                "island": "Torgersen"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    predicted_species: str = Field(..., description="Predicted penguin species")
    confidence: float = Field(..., description="Prediction confidence (max probability)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each species")


class ModelManager:
    """Manages model loading and prediction operations."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.species_classes = None
        self._load_model_artifacts()
    
    def _load_model_artifacts(self) -> None:
        """Load the trained model and associated artifacts."""
        try:
            logger.info("Loading model artifacts...")
            
            # Load XGBoost model
            model_path = Path("app/data/model.json")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path))
            logger.info("XGBoost model loaded successfully")
            
            # Load label encoder
            encoder_path = Path("app/data/label_encoder.pkl")
            if not encoder_path.exists():
                raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
                
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("Label encoder loaded successfully")
            
            # Load metadata
            metadata_path = Path("app/data/metadata.json")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.species_classes = metadata['species_classes']
            
            logger.info(f"Loaded metadata - Features: {len(self.feature_columns)}, Classes: {self.species_classes}")
            logger.info("All model artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {str(e)}")
            raise
    
    def _prepare_features(self, features: PenguinFeatures) -> pd.DataFrame:
        """
        Prepare input features for prediction by applying the same preprocessing
        as used during training.
        
        Args:
            features: Input features from API request
            
        Returns:
            pd.DataFrame: Preprocessed features ready for prediction
        """
        try:
            logger.debug(f"Preparing features for prediction: {features.dict()}")
            
            # Convert input to dictionary
            feature_dict = features.dict()
            
            # Create DataFrame with a single row
            input_df = pd.DataFrame([feature_dict])
            
            # Apply one-hot encoding to categorical features
            # This must match exactly what was done during training
            categorical_features = ['island', 'sex']
            input_encoded = pd.get_dummies(input_df, columns=categorical_features, dtype=int)
            
            # Ensure all training features are present, fill missing with 0
            for col in self.feature_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match training order
            input_encoded = input_encoded[self.feature_columns]
            
            logger.debug(f"Features prepared successfully, shape: {input_encoded.shape}")
            return input_encoded
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def predict(self, features: PenguinFeatures) -> PredictionResponse:
        """
        Make a prediction for the given penguin features.
        
        Args:
            features: Input features for prediction
            
        Returns:
            PredictionResponse: Prediction results with confidence scores
        """
        try:
            logger.info("Making prediction for penguin species")
            
            # Prepare features
            X = self._prepare_features(features)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Convert prediction back to species name
            predicted_species = self.label_encoder.inverse_transform([prediction])[0]
            
            # Create probability dictionary
            prob_dict = {
                species: float(prob) 
                for species, prob in zip(self.species_classes, probabilities)
            }
            
            # Get confidence (maximum probability)
            confidence = float(max(probabilities))
            
            logger.info(f"Prediction completed: {predicted_species} (confidence: {confidence:.4f})")
            
            return PredictionResponse(
                predicted_species=predicted_species,
                confidence=confidence,
                probabilities=prob_dict
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )


# Initialize FastAPI app and model manager
app = FastAPI(
    title="Penguin Species Classification API",
    description="API for predicting penguin species based on physical measurements",
    version="1.0.0"
)

model_manager = ModelManager()


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with basic API information."""
    logger.info("Root endpoint accessed")
    return {
        "message": "Penguin Species Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "prediction_endpoint": "/predict"
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "model_loaded": model_manager.model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_species(features: PenguinFeatures) -> PredictionResponse:
    """
    Predict penguin species based on input features.
    
    Args:
        features: Penguin measurements and categorical features
        
    Returns:
        PredictionResponse: Predicted species with confidence scores
        
    Raises:
        HTTPException: For invalid input or prediction errors
    """
    try:
        logger.info(f"Prediction request received: {features.dict()}")
        
        # Validate that model is loaded
        if model_manager.model is None:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=500,
                detail="Model not available"
            )
        
        # Additional validation for enum values (handled by Pydantic but logged)
        if features.sex not in [Sex.Male, Sex.Female]:
            logger.warning(f"Invalid sex value attempted: {features.sex}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sex value. Must be one of: {[e.value for e in Sex]}"
            )
        
        if features.island not in [Island.Torgersen, Island.Biscoe, Island.Dream]:
            logger.warning(f"Invalid island value attempted: {features.island}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid island value. Must be one of: {[e.value for e in Island]}"
            )
        
        # Make prediction
        result = model_manager.predict(features)
        
        logger.info(f"Prediction successful: {result.predicted_species}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions with appropriate HTTP responses."""
    logger.error(f"ValueError: {str(exc)}")
    return HTTPException(
        status_code=400,
        detail=f"Invalid input: {str(exc)}"
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)