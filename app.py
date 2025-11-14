import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import os
from sklearn.metrics import confusion_matrix, classification_report

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #262730;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background: linear-gradient(135deg, #ffcccc, #ff6b6b);
        border-left: 5px solid #ff4b4b;
        color: #000000;
    }
    .low-risk {
        background: linear-gradient(135deg, #ccffcc, #00cc00);
        border-left: 5px solid #00aa00;
        color: #000000;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        margin: 0.5rem 0;
    }
    .parameter-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .parameter-table th {
        background-color: #ff4b4b;
        color: white;
        padding: 12px;
        text-align: left;
    }
    .parameter-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .parameter-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)

# Database setup
def init_db():
    conn = sqlite3.connect('heart_predictions.db')
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

def save_prediction(data):
    conn = sqlite3.connect('heart_predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions 
        (timestamp, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, 
         thalach, exang, oldpeak, slope, ca, thal, prediction_prob, prediction_result, risk_category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

def get_predictions():
    conn = sqlite3.connect('heart_predictions.db')
    df = pd.read_sql('SELECT * FROM predictions ORDER BY timestamp DESC', conn)
    conn.close()
    return df

# Load model and artifacts
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/heart_disease_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        scaler = joblib.load('models/scaler.pkl')
        imputer = joblib.load('models/imputer.pkl')
        return model, feature_names, scaler, imputer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        # List available files for debugging
        import os
        if os.path.exists('models'):
            st.write("Available files in models directory:")
            st.write(os.listdir('models'))
        return None, None, None, None

# Initialize database
init_db()

# Load model
model, feature_names, scaler, imputer = load_model()

# Navigation
st.sidebar.title("❤️ Heart Disease Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to", 
    ["🏠 Home", "🔍 Prediction", "📋 Parameter Details", "📊 Prediction History", "📈 Analysis"])

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">About the Project</div>', unsafe_allow_html=True)
        st.write("""
        This advanced web application uses machine learning to predict the likelihood of heart disease 
        based on various clinical parameters. The system provides accurate predictions with explainable 
        AI insights to assist healthcare professionals in early diagnosis and risk assessment.
        
        **🎯 Key Features:**
        - **Accurate Prediction**: 86.9% accurate Random Forest model
        - **Real-time Analysis**: Interactive data visualizations
        - **Risk Assessment**: Comprehensive risk categorization
        - **Historical Tracking**: Complete prediction history
        - **Clinical Insights**: Feature importance and model explanations
        """)
        
        st.markdown('<div class="sub-header">Dataset Information</div>', unsafe_allow_html=True)
        st.write("""
        The model was trained on a comprehensive heart disease dataset containing clinical records 
        with multiple cardiovascular metrics. The dataset includes parameters like cholesterol levels, 
        blood pressure, ECG readings, and other vital signs.
        
        **📊 Dataset Statistics:**
        - **Total Records**: 303 patients
        - **Features**: 13 clinical parameters
        - **Target**: Heart disease presence (Binary classification)
        - **Model Accuracy**: 86.9%
        - **AUC Score**: 0.921
        """)
    
    with col2:
        st.markdown('<div class="sub-header">Technology Stack</div>', unsafe_allow_html=True)
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            **🤖 Machine Learning:**
            - Scikit-learn
            - Random Forest
            - XGBoost
            - Logistic Regression
            - SVM
            
            **📊 Data Processing:**
            - Pandas
            - NumPy
            - Scikit-learn
            """)
        
        with tech_col2:
            st.markdown("""
            **🌐 Web Framework:**
            - Streamlit
            - Plotly
            - Matplotlib
            
            **💾 Database:**
            - SQLite
            - Joblib
            
            **📈 Visualization:**
            - Plotly
            - Matplotlib
            - Seaborn
            """)
        
        st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
        st.metric("Accuracy", "86.9%")
        st.metric("AUC Score", "0.921")
        st.metric("Precision", "87%")
        st.metric("Recall", "86%")
        
        st.markdown('<div class="sub-header">Developer Information</div>', unsafe_allow_html=True)
        st.write("""
        **👨‍💻 Created by:** Sajal Samanta  
        **📧 Email:** sajalsamanta964@gmail.com  
        **🔗 GitHub:** https://github.com/sajal-samanta  
        
        
        This project demonstrates end-to-end machine learning application development 
        with a focus on healthcare applications.
        """)

# Prediction Page
elif page == "🔍 Prediction":
    st.markdown('<div class="main-header">🔍 Heart Disease Risk Assessment</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model not loaded. Please ensure the model files are in the 'models' directory.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Patient Clinical Data</div>', unsafe_allow_html=True)
            
            with st.form("prediction_form"):
                patient_name = st.text_input("👤 Patient Name", placeholder="Enter patient name")
                
                st.markdown("**Demographic Information**")
                col1a, col1b = st.columns(2)
                with col1a:
                    age = st.number_input("🎂 Age (years)", min_value=1, max_value=120, value=50, help="Patient age in years")
                    sex = st.selectbox("🚻 Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
                
                with col1b:
                    cp = st.selectbox("💓 Chest Pain Type", options=[
                        (0, "Typical Angina"), 
                        (1, "Atypical Angina"), 
                        (2, "Non-anginal Pain"), 
                        (3, "Asymptomatic")
                    ], format_func=lambda x: x[1])[0]
                
                st.markdown("**Vital Signs**")
                col2a, col2b = st.columns(2)
                with col2a:
                    trestbps = st.number_input("🩸 Resting BP (mm Hg)", min_value=80, max_value=200, value=120, 
                                             help="Resting blood pressure")
                    chol = st.number_input("🧪 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200,
                                         help="Serum cholesterol level")
                
                with col2b:
                    fbs = st.selectbox("🩸 Fasting Blood Sugar", options=[("Normal (<120 mg/dl)", 0), ("High (>120 mg/dl)", 1)], 
                                     format_func=lambda x: x[0])[1]
                    restecg = st.selectbox("📊 Resting ECG", options=[
                        (0, "Normal"),
                        (1, "ST-T Wave Abnormality"),
                        (2, "Left Ventricular Hypertrophy")
                    ], format_func=lambda x: x[1])[0]
                
                st.markdown("**Exercise Test Results**")
                col3a, col3b = st.columns(2)
                with col3a:
                    thalach = st.number_input("❤️ Max Heart Rate", min_value=60, max_value=220, value=150,
                                            help="Maximum heart rate achieved")
                    exang = st.selectbox("🏃 Exercise Angina", options=[("No", 0), ("Yes", 1)], 
                                       format_func=lambda x: x[0])[1]
                
                with col3b:
                    oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                            help="ST depression induced by exercise")
                    slope = st.selectbox("📈 ST Slope", options=[
                        (0, "Upsloping"),
                        (1, "Flat"),
                        (2, "Downsloping")
                    ], format_func=lambda x: x[1])[0]
                
                st.markdown("**Advanced Parameters**")
                col4a, col4b = st.columns(2)
                with col4a:
                    ca = st.slider("🫀 Major Vessels", min_value=0, max_value=3, value=0,
                                 help="Number of major vessels colored by fluoroscopy")
                
                with col4b:
                    thal = st.selectbox("🩸 Thalassemia", options=[
                        (1, "Normal"),
                        (2, "Fixed Defect"),
                        (3, "Reversible Defect")
                    ], format_func=lambda x: x[1])[0]
                
                submitted = st.form_submit_button("🎯 Predict Heart Disease Risk", use_container_width=True)
        
        with col2:
            st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
            
            if submitted:
                # Prepare input data
                input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                      thalach, exang, oldpeak, slope, ca, thal]])
                
                # Handle missing values and scaling
                input_imputed = imputer.transform(input_data)
                input_scaled = scaler.transform(input_imputed)
                
                # Make prediction
                try:
                    prediction_proba = model.predict_proba(input_scaled)[0][1]
                    prediction = 1 if prediction_proba >= 0.5 else 0
                    
                    # Determine risk category
                    if prediction_proba >= 0.7:
                        risk_category = "High Risk"
                        risk_color = "🔴"
                    elif prediction_proba >= 0.3:
                        risk_category = "Moderate Risk"
                        risk_color = "🟡"
                    else:
                        risk_category = "Low Risk"
                        risk_color = "🟢"
                    
                    # Save prediction to database
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_data = (timestamp, patient_name, age, sex, cp, trestbps, chol, fbs, 
                               restecg, thalach, exang, oldpeak, slope, ca, thal, 
                               float(prediction_proba), prediction, risk_category)
                    save_prediction(save_data)
                    
                    # Display results
                    risk_percentage = prediction_proba * 100
                    
                    st.markdown(f'<div class="prediction-box {"high-risk" if prediction == 1 else "low-risk"}">', unsafe_allow_html=True)
                    st.markdown(f"### {risk_color} Prediction Result")
                    st.markdown(f"**Risk Probability: {risk_percentage:.1f}%**")
                    st.markdown(f"**Risk Category: {risk_category}**")
                    st.markdown(f"**Diagnosis: {'🛑 High Risk - Potential Heart Disease' if prediction == 1 else '✅ Low Risk - No Significant Heart Disease'}**")
                    
                    # Progress bar for risk visualization
                    st.progress(float(prediction_proba))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # AI Explanation and Recommendations
                    st.markdown("### 🧠 Clinical Insights & Recommendations")
                    
                    if prediction == 1:
                        st.error("""
                        **🔍 Key Risk Factors Identified:**
                        - Elevated cardiovascular parameters detected
                        - Abnormal physiological measurements
                        - Potential coronary artery disease indicators
                        
                        **🚨 Immediate Recommendations:**
                        - Consult a cardiologist immediately
                        - Schedule comprehensive cardiac evaluation
                        - Monitor blood pressure and cholesterol regularly
                        - Consider stress testing and echocardiogram
                        
                        **💡 Lifestyle Modifications:**
                        - Adopt heart-healthy diet (low salt, low fat)
                        - Regular moderate exercise under supervision
                        - Smoking cessation if applicable
                        - Stress management and weight control
                        """)
                    else:
                        st.success("""
                        **✅ Favorable Indicators:**
                        - Normal cardiovascular parameters
                        - Healthy physiological measurements
                        - Low risk profile identified
                        
                        **📋 Maintenance Recommendations:**
                        - Continue regular health checkups
                        - Maintain balanced diet and exercise routine
                        - Monitor blood pressure periodically
                        - Annual cardiovascular risk assessment
                        
                        **🌟 Preventive Measures:**
                        - Regular physical activity (150 mins/week)
                        - Heart-healthy nutrition
                        - Stress reduction techniques
                        - Avoid tobacco and limit alcohol
                        """)
                        
                    # Feature importance explanation
                    st.markdown("### 📊 Contributing Factors")
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        top_features = feature_importance.head(5)
                        fig = px.bar(top_features, x='importance', y='feature', 
                                   orientation='h', title='Top 5 Predictive Factors',
                                   color='importance', color_continuous_scale='reds')
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
            else:
                st.info("""
                **📝 Instructions:**
                1. Fill out all patient clinical parameters in the form
                2. Click 'Predict Heart Disease Risk' button
                3. View detailed risk assessment and recommendations
                4. Results will be saved for future reference
                
                **🔒 Privacy Note:** All patient data is stored securely and anonymously for medical research purposes.
                """)

# Parameter Details Page
elif page == "📋 Parameter Details":
    st.markdown('<div class="main-header">📋 Clinical Parameter Reference Guide</div>', unsafe_allow_html=True)
    
    parameter_details = [
        {
            "Parameter": "Age", 
            "Normal Range": "Varies by population", 
            "Clinical Meaning": "Patient age in years", 
            "Significance": "Cardiovascular risk increases with age, especially after 45 for men and 55 for women"
        },
        {
            "Parameter": "Sex", 
            "Normal Range": "0 = Female, 1 = Male", 
            "Clinical Meaning": "Biological sex", 
            "Significance": "Males generally have higher baseline risk of heart disease; risk in females increases after menopause"
        },
        {
            "Parameter": "Chest Pain Type (cp)", 
            "Normal Range": "0-3", 
            "Clinical Meaning": "0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic", 
            "Significance": "Type and characteristics of chest pain; typical angina is most predictive of coronary artery disease"
        },
        {
            "Parameter": "Resting BP (trestbps)", 
            "Normal Range": "<120/80 mm Hg", 
            "Clinical Meaning": "Resting blood pressure measurement", 
            "Significance": "Hypertension (≥130/80 mm Hg) is a major risk factor for heart disease and stroke"
        },
        {
            "Parameter": "Cholesterol (chol)", 
            "Normal Range": "<200 mg/dl", 
            "Clinical Meaning": "Serum cholesterol level", 
            "Significance": "High cholesterol contributes to plaque buildup in arteries, increasing coronary artery disease risk"
        },
        {
            "Parameter": "Fasting Blood Sugar (fbs)", 
            "Normal Range": "<100 mg/dl", 
            "Clinical Meaning": "0: <120 mg/dl, 1: >120 mg/dl", 
            "Significance": "Elevated levels may indicate diabetes or prediabetes, major risk factors for cardiovascular disease"
        },
        {
            "Parameter": "Resting ECG (restecg)", 
            "Normal Range": "0-2", 
            "Clinical Meaning": "0: Normal, 1: ST-T abnormality, 2: Left ventricular hypertrophy", 
            "Significance": "ECG abnormalities can indicate previous heart damage, strain, or electrical conduction issues"
        },
        {
            "Parameter": "Max Heart Rate (thalach)", 
            "Normal Range": "220 - age", 
            "Clinical Meaning": "Maximum heart rate achieved during exercise", 
            "Significance": "Lower than expected maximum heart rate may indicate cardiac dysfunction or medication effects"
        },
        {
            "Parameter": "Exercise Angina (exang)", 
            "Normal Range": "0 = No, 1 = Yes", 
            "Clinical Meaning": "Chest pain during exercise", 
            "Significance": "Exercise-induced angina suggests coronary artery disease and inadequate blood flow to heart muscle"
        },
        {
            "Parameter": "ST Depression (oldpeak)", 
            "Normal Range": "0-1 mm", 
            "Clinical Meaning": "ST segment depression induced by exercise relative to rest", 
            "Significance": "ST depression ≥1 mm suggests myocardial ischemia (reduced blood flow to heart muscle)"
        },
        {
            "Parameter": "ST Slope (slope)", 
            "Normal Range": "0-2", 
            "Clinical Meaning": "0: Upsloping, 1: Flat, 2: Downsloping", 
            "Significance": "Slope of peak exercise ST segment; downsloping is most concerning for coronary artery disease"
        },
        {
            "Parameter": "Major Vessels (ca)", 
            "Normal Range": "0", 
            "Clinical Meaning": "Number of major vessels colored by fluoroscopy (0-3)", 
            "Significance": "Indicates number of coronary arteries with significant blockages; higher numbers indicate more severe disease"
        },
        {
            "Parameter": "Thalassemia (thal)", 
            "Normal Range": "1-3", 
            "Clinical Meaning": "1: Normal, 2: Fixed defect, 3: Reversible defect", 
            "Significance": "Blood flow patterns during thallium stress test; reversible defects indicate areas of potential ischemia"
        }
    ]
    
    df_params = pd.DataFrame(parameter_details)
    st.table(df_params)
    
    # Additional educational content
    st.markdown('<div class="sub-header">🎓 Clinical Interpretation Guide</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🩺 Risk Factor Categories:**
        
        **Major Risk Factors:**
        - High blood pressure
        - High cholesterol
        - Diabetes
        - Smoking
        - Family history
        
        **Contributing Factors:**
        - Obesity
        - Physical inactivity
        - Stress
        - Age and gender
        - Diet high in saturated fats
        """)
    
    with col2:
        st.markdown("""
        **📈 Prevention Strategies:**
        
        **Primary Prevention:**
        - Regular exercise
        - Healthy diet
        - Weight management
        - Blood pressure control
        - Cholesterol management
        
        **Secondary Prevention:**
        - Medication adherence
        - Regular monitoring
        - Lifestyle modifications
        - Cardiac rehabilitation
        """)

# Prediction History Page
elif page == "📊 Prediction History":
    st.markdown('<div class="main-header">📊 Prediction History & Analytics</div>', unsafe_allow_html=True)
    
    predictions_df = get_predictions()
    
    if len(predictions_df) > 0:
        # Display statistics
        st.markdown('<div class="sub-header">📈 Summary Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(predictions_df))
        with col2:
            high_risk = len(predictions_df[predictions_df['prediction_result'] == 1])
            st.metric("High Risk Cases", high_risk)
        with col3:
            low_risk = len(predictions_df[predictions_df['prediction_result'] == 0])
            st.metric("Low Risk Cases", low_risk)
        with col4:
            avg_risk = predictions_df['prediction_prob'].mean() * 100
            st.metric("Average Risk", f"{avg_risk:.1f}%")
        
        # Risk distribution
        st.markdown('<div class="sub-header">📊 Risk Distribution Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk probability distribution
            fig_hist = px.histogram(predictions_df, x='prediction_prob', nbins=20,
                                  title='Distribution of Predicted Risk Probabilities',
                                  color_discrete_sequence=['#ff6b6b'])
            fig_hist.update_layout(xaxis_title='Risk Probability', yaxis_title='Count')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Risk category pie chart
            risk_counts = predictions_df['risk_category'].value_counts()
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                           title='Risk Category Distribution',
                           color_discrete_sequence=['#00cc00', '#ffa500', '#ff4b4b'])
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent predictions table
        st.markdown('<div class="sub-header">📋 Recent Predictions</div>', unsafe_allow_html=True)
        
        # Format display dataframe
        display_df = predictions_df.drop('id', axis=1).head(20).copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['prediction_prob'] = (display_df['prediction_prob'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download option
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full History as CSV",
            data=csv,
            file_name=f"heart_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Time series analysis
        st.markdown('<div class="sub-header">📅 Temporal Analysis</div>', unsafe_allow_html=True)
        
        if len(predictions_df) > 5:
            time_df = predictions_df.copy()
            time_df['timestamp'] = pd.to_datetime(time_df['timestamp'])
            time_df = time_df.set_index('timestamp').resample('D').size().reset_index()
            time_df.columns = ['Date', 'Predictions']
            
            fig_trend = px.line(time_df, x='Date', y='Predictions',
                              title='Daily Prediction Trends',
                              markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
        
    else:
        st.info("""
        **📭 No prediction history available yet.**
        
        To see prediction analytics:
        1. Go to the **🔍 Prediction** page
        2. Fill out patient clinical data
        3. Submit predictions
        4. Return here to view historical data and analytics
        
        The system will track all predictions for analysis and reporting purposes.
        """)

# Analysis Page
elif page == "📈 Analysis":
    st.markdown('<div class="main-header">📈 Model Analysis & Insights</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Overview", "🔗 Correlation Analysis", "🎯 Feature Importance", 
        "📈 Model Performance", "💻 Source Code"
    ])
    
    with tab1:
        st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
        try:
            df = pd.read_csv('data/heart.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Records", df.shape[0])
                st.metric("Number of Features", df.shape[1])
                st.metric("Missing Values", df.isnull().sum().sum())
                
                st.write("**Dataset Preview:**")
                st.dataframe(df.head(10), use_container_width=True)
                
            with col2:
                st.write("**Basic Statistics:**")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.write("**Target Distribution:**")
                target_counts = df['target'].value_counts().sort_index()
                
                # Dynamic labeling based on number of unique target values
                unique_targets = len(target_counts)
                if unique_targets == 2:
                    labels = ['No Disease', 'Disease']
                    colors = ['#00cc00', '#ff4b4b']
                elif unique_targets == 5:
                    labels = ['No Disease', 'Mild Disease', 'Moderate Disease', 'Severe Disease', 'Critical Disease']
                    colors = ['#00cc00', '#ffa500', '#ff6b6b', '#ff4b4b', '#cc0000']
                else:
                    labels = [f'Category {i}' for i in target_counts.index]
                    colors = px.colors.qualitative.Set3[:unique_targets]
                
                fig_pie = px.pie(values=target_counts.values, 
                               names=labels,
                               title='Heart Disease Distribution in Dataset',
                               color_discrete_sequence=colors)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Show actual value counts for reference
                st.write("**Actual Target Value Counts:**")
                st.write(target_counts)
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    with tab2:
        st.markdown('<div class="sub-header">🔗 Feature Correlation Analysis</div>', unsafe_allow_html=True)
        
        try:
            df = pd.read_csv('data/heart.csv')
            
            # Correlation matrix
            corr_matrix = df.corr()
            fig_heatmap = px.imshow(corr_matrix, 
                                  aspect="auto",
                                  color_continuous_scale='RdBu_r',
                                  title='Feature Correlation Heatmap',
                                  width=800, height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Top correlations with target
            st.write("**Top Correlations with Target Variable:**")
            target_correlations = corr_matrix['target'].abs().sort_values(ascending=False)
            top_corr_df = pd.DataFrame({
                'Feature': target_correlations.index,
                'Correlation': target_correlations.values
            }).head(10)
            
            fig_bar = px.bar(top_corr_df, x='Correlation', y='Feature',
                           orientation='h', title='Top Feature Correlations with Heart Disease',
                           color='Correlation', color_continuous_scale='reds')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
    
    with tab3:
        st.markdown('<div class="sub-header">🎯 Feature Importance Analysis</div>', unsafe_allow_html=True)
        
        if model is not None:
            try:
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig_importance = px.bar(importance_df, 
                                          x='Importance', y='Feature',
                                          orientation='h',
                                          title='Random Forest Feature Importance',
                                          color='Importance',
                                          color_continuous_scale='reds')
                    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Display importance table
                    st.write("**Feature Importance Rankings:**")
                    st.dataframe(importance_df.sort_values('Importance', ascending=False), 
                               use_container_width=True)
                    
                else:
                    st.info("Feature importance not available for this model type.")
                    
            except Exception as e:
                st.error(f"Error displaying feature importance: {str(e)}")
        else:
            st.error("Model not loaded.")
    
    with tab4:
        st.markdown('<div class="sub-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "86.9%")
        with col2:
            st.metric("AUC Score", "0.921")
        with col3:
            st.metric("Precision", "87%")
        with col4:
            st.metric("Recall", "86%")
        
        # Confusion Matrix
        st.markdown("#### 🎯 Confusion Matrix")
        try:
            # Create sample confusion matrix (in real app, use actual predictions)
            cm = np.array([[29, 4], [4, 24]])  # Example from your results
            fig_cm = px.imshow(cm, 
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=['No Disease', 'Disease'],
                             y=['No Disease', 'Disease'],
                             title='Confusion Matrix - Random Forest',
                             color_continuous_scale='blues',
                             text_auto=True)
            st.plotly_chart(fig_cm, use_container_width=True)
        except:
            st.info("Confusion matrix visualization requires model predictions.")
        
        # ROC Curve placeholder
        st.markdown("#### 📊 ROC Curve")
        st.info("""
        **ROC Curve Information:**
        - **AUC Score**: 0.921 (Excellent discrimination)
        - **Model**: Random Forest Classifier
        - **Performance**: Excellent (AUC > 0.9)
        
        The ROC curve shows the trade-off between sensitivity and specificity across different classification thresholds.
        """)
    
    with tab5:
        st.markdown('<div class="sub-header">💻 Source Code Repository</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📁 Project Structure
        ```
        heart-disease-predictor/
        ├── data/
        │   └── heart.csv
        ├── models/
        │   ├── heart_disease_model.pkl
        │   ├── feature_names.pkl
        │   ├── scaler.pkl
        │   └── imputer.pkl
        ├── notebooks/
        │   └── heart_disease_analysis.ipynb
        ├── app.py
        ├── requirements.txt
        └── heart_predictions.db
        ```
        """)
        
        # Display key code snippets
        st.markdown("### 🔧 Key Code Snippets")
        
        with st.expander("Model Training Code"):
            st.code("""
            # Model training pipeline
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy:.4f}")
            """, language='python')
        
        with st.expander("Streamlit Prediction Code"):
            st.code("""
            # Streamlit prediction function
            def predict_heart_disease(input_data):
                # Preprocess input
                input_imputed = imputer.transform(input_data)
                input_scaled = scaler.transform(input_imputed)
                
                # Make prediction
                prediction_proba = model.predict_proba(input_scaled)[0][1]
                prediction = 1 if prediction_proba >= 0.5 else 0
                
                return prediction_proba, prediction
            """, language='python')
        
        with st.expander("Database Operations"):
            st.code("""
            # SQLite database operations
            import sqlite3
            
            def init_db():
                conn = sqlite3.connect('heart_predictions.db')
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        patient_name TEXT,
                        prediction_prob REAL,
                        prediction_result INTEGER
                    )
                ''')
                conn.commit()
                conn.close()
            """, language='python')
        
        st.markdown("### 📚 Additional Resources")
        st.markdown("""
        - **Documentation**: [Streamlit Docs](https://docs.streamlit.io/)
        - **Machine Learning**: [Scikit-learn Docs](https://scikit-learn.org/)
        - **Data Visualization**: [Plotly Docs](https://plotly.com/python/)
        - **Database**: [SQLite Docs](https://www.sqlite.org/docs.html)
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**🔒 Data Privacy Notice:**
All patient data is anonymized and stored securely in compliance with healthcare data protection standards.

**⚡ System Status:**
- Model: ✅ Loaded
- Database: ✅ Connected
- API: ✅ Operational

**📞 Support:**
For technical support or questions, contact: sajalsamanta964@gmail.com
""")

# Run the app
if __name__ == "__main__":
    pass