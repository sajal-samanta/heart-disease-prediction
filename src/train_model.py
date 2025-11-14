import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def train_and_save_model():
    """Train the heart disease model and save all artifacts"""
    
    # Load data
    df = pd.read_csv('../data/heart.csv')
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Convert to binary classification for better performance
    y_binary = (y > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1
    )
    
    model.fit(X_train_imputed, y_train)
    
    # Evaluate model
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = model.predict(X_test_imputed)
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save artifacts
    joblib.dump(model, '../models/heart_disease_model.pkl')
    joblib.dump(list(X.columns), '../models/feature_names.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(imputer, '../models/imputer.pkl')
    
    print("Model and artifacts saved to '../models/' directory")

if __name__ == "__main__":
    train_and_save_model()