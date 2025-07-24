import os
import json
import pickle
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score
import xgboost as xgb


def load_data() -> pd.DataFrame:
    """
    Load the penguins dataset from Seaborn.
    
    Returns:
        pd.DataFrame: The penguins dataset
    """
    print("Loading penguins dataset...")
    penguins = sns.load_dataset('penguins')
    print(f"Dataset loaded with shape: {penguins.shape}")
    return penguins


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, list]:
    """
    Preprocess the penguins dataset by handling missing values, encoding categorical
    variables, and preparing features and target.
    
    Args:
        df: Raw penguins dataframe
        
    Returns:
        Tuple containing:
        - X: Feature matrix with one-hot encoded categorical variables
        - y: Label encoded target variable
        - label_encoder: Fitted label encoder for species
        - feature_columns: List of final feature column names
    """
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Drop rows with missing values
    print(f"Rows before dropping NaN: {len(data)}")
    data = data.dropna()
    print(f"Rows after dropping NaN: {len(data)}")
    
    # Separate features and target
    target_col = 'species'
    feature_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                   'body_mass_g', 'island', 'sex']
    
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Apply one-hot encoding to categorical features
    categorical_features = ['island', 'sex']
    print(f"Applying one-hot encoding to: {categorical_features}")
    
    X_encoded = pd.get_dummies(X, columns=categorical_features, dtype=int)
    feature_columns = X_encoded.columns.tolist()
    
    print(f"Features after one-hot encoding: {feature_columns}")
    
    # Apply label encoding to target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Species classes: {label_encoder.classes_}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return X_encoded, y_encoded, label_encoder, feature_columns


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with parameters to prevent overfitting.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        xgb.XGBClassifier: Trained XGBoost model
    """
    print("Training XGBoost model...")
    
    # Initialize XGBoost classifier with overfitting prevention parameters
    model = xgb.XGBClassifier(
        max_depth=3,           # Limit tree depth to prevent overfitting
        n_estimators=100,      # Number of boosting rounds
        learning_rate=0.1,     # Step size shrinkage
        subsample=0.8,         # Subsample ratio of training instances
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        random_state=42,       # For reproducibility
        eval_metric='mlogloss' # Multi-class log loss
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model


def evaluate_model(model: xgb.XGBClassifier, X_train: pd.DataFrame, y_train: np.ndarray,
                  X_test: pd.DataFrame, y_test: np.ndarray, label_encoder: LabelEncoder) -> Dict[str, float]:
    """
    Evaluate the trained model on training and test sets.
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for species names
        
    Returns:
        Dict containing evaluation metrics
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1_score': train_f1,
        'test_f1_score': test_f1
    }
    
    # Print results
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Detailed classification report for test set
    print(f"\nClassification Report (Test Set):")
    species_names = label_encoder.classes_
    print(classification_report(y_test, y_test_pred, target_names=species_names))
    
    return metrics


def save_model_and_metadata(model: xgb.XGBClassifier, label_encoder: LabelEncoder, 
                           feature_columns: list, metrics: Dict[str, float]) -> None:
    """
    Save the trained model and associated metadata to the app/data directory.
    
    Args:
        model: Trained XGBoost model
        label_encoder: Fitted label encoder
        feature_columns: List of feature column names
        metrics: Model performance metrics
    """
    print("Saving model and metadata...")
    
    # Create app/data directory if it doesn't exist
    os.makedirs('app/data', exist_ok=True)
    
    # Save the XGBoost model
    model.save_model('app/data/model.json')
    
    # Save label encoder
    with open('app/data/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save feature columns and metadata
    metadata = {
        'feature_columns': feature_columns,
        'species_classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_),
        'metrics': metrics,
        'model_params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    }
    
    with open('app/data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model and metadata saved successfully!")
    print(f"Files saved:")
    print(f"- app/data/model.json")
    print(f"- app/data/label_encoder.pkl")
    print(f"- app/data/metadata.json")


def main() -> None:
    """
    Main training pipeline that orchestrates the entire process.
    """
    print("Starting penguin species classification model training pipeline...")
    print("=" * 60)
    
    try:
        # Load data
        df = load_data()
        
        # Preprocess data
        X, y, label_encoder, feature_columns = preprocess_data(df)
        
        # Split data into train and test sets with stratification
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y  # Ensure balanced split across classes
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder)
        
        # Save model and metadata
        save_model_and_metadata(model, label_encoder, feature_columns, metrics)
        
        print("\n" + "=" * 60)
        print("Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()