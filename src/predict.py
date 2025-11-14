import joblib
import numpy as np
import pandas as pd

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessing artifacts"""
        try:
            self.model = joblib.load('models/heart_disease_model.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.imputer = joblib.load('models/imputer.pkl')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, input_data):
        """Make prediction on new data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy array and ensure correct shape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Preprocess
        input_imputed = self.imputer.transform(input_array)
        input_scaled = self.scaler.transform(input_imputed)
        
        # Predict
        probability = self.model.predict_proba(input_scaled)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Determine risk category
        if probability >= 0.7:
            risk_category = "High Risk"
        elif probability >= 0.3:
            risk_category = "Moderate Risk"
        else:
            risk_category = "Low Risk"
        
        return {
            'probability': probability,
            'prediction': prediction,
            'risk_category': risk_category,
            'risk_percentage': probability * 100
        }
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None

# Example usage
if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    
    # Example prediction
    example_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    result = predictor.predict(example_data)
    print(f"Prediction result: {result}")