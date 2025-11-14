import sqlite3
import pandas as pd
from datetime import datetime

class HeartDiseaseDB:
    def __init__(self, db_path='heart_predictions.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                patient_name TEXT,
                age INTEGER,
                sex INTEGER,
                cp INTEGER,
                trestbps INTEGER,
                chol INTEGER,
                fbs INTEGER,
                restecg INTEGER,
                thalach INTEGER,
                exang INTEGER,
                oldpeak REAL,
                slope INTEGER,
                ca INTEGER,
                thal INTEGER,
                prediction_prob REAL,
                prediction_result INTEGER,
                risk_category TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_prediction(self, data):
        """Save prediction to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions 
            (timestamp, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, 
             thalach, exang, oldpeak, slope, ca, thal, prediction_prob, prediction_result, risk_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        conn.close()
    
    def get_predictions(self, limit=None):
        """Get predictions from database"""
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM predictions ORDER BY timestamp DESC'
        if limit:
            query += f' LIMIT {limit}'
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_statistics(self):
        """Get prediction statistics"""
        df = self.get_predictions()
        if len(df) == 0:
            return {}
        
        stats = {
            'total_predictions': len(df),
            'high_risk_cases': len(df[df['prediction_result'] == 1]),
            'low_risk_cases': len(df[df['prediction_result'] == 0]),
            'average_risk': df['prediction_prob'].mean() * 100,
            'recent_activity': len(df[df['timestamp'] >= datetime.now().strftime('%Y-%m-%d')])
        }
        return stats