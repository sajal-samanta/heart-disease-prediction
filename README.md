# heart-disease-prediction
# ❤️ Heart Disease Prediction System

![Heart Disease Predictor](https://img.shields.io/badge/Heart-Disease%20Predictor-red) ![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue) ![Streamlit](https://img.shields.io/badge/Web-Streamlit-green) ![Accuracy](https://img.shields.io/badge/Accuracy-86.9%25-brightgreen)

A comprehensive machine learning web application that predicts heart disease risk with **86.9% accuracy** using clinical parameters. Built with Streamlit and powered by Random Forest algorithm.

## 🎯 Live Demo

🚀 **Try the live application:** [Coming Soon](#)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-predictor.streamlit.app/)

## 📊 Project Highlights

- **🎯 86.9% Accuracy** - Highly reliable predictions
- **🩺 Clinical Insights** - Explainable AI with feature importance
- **📊 Interactive Dashboard** - Real-time data visualizations
- **💾 Data Management** - Complete prediction history tracking
- **🎨 Professional UI** - Healthcare-focused design

## 🚀 Features

### 🔍 Prediction Engine
- **Real-time Risk Assessment** - Instant heart disease probability calculation
- **Multi-Parameter Analysis** - 13 clinical parameters evaluated
- **Risk Categorization** - High, Moderate, and Low risk classification
- **AI Explanations** - Understand why specific predictions are made

### 📈 Analytics Dashboard
- **Interactive Visualizations** - Plotly charts for data exploration
- **Feature Importance** - See which factors matter most
- **Model Performance** - Comprehensive metrics and ROC analysis
- **Correlation Heatmaps** - Understand parameter relationships

### 💾 Data Management
- **SQLite Database** - Secure prediction history storage
- **Export Capabilities** - Download prediction data as CSV
- **Patient History** - Track and analyze past predictions
- **Privacy Focused** - Anonymized data handling

### 🎨 User Experience
- **Responsive Design** - Works on all devices
- **Clinical Reference** - Parameter interpretation guide
- **Professional Interface** - Healthcare-grade UI/UX
- **Multi-page Navigation** - Intuitive user journey

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **🤖 Machine Learning** | Scikit-learn, Random Forest, XGBoost, Logistic Regression |
| **🌐 Web Framework** | Streamlit, Plotly, Matplotlib, Seaborn |
| **💾 Database** | SQLite, Joblib |
| **📊 Data Processing** | Pandas, NumPy, Scikit-learn |
| **🎨 Visualization** | Plotly, Matplotlib, Seaborn |

## 📁 Project Structure

```
heart-disease-predictor/
├── 📊 data/
│   └── heart.csv                 # Dataset
├── 🤖 models/
│   ├── heart_disease_model.pkl   # Trained model
│   ├── feature_names.pkl         # Feature names
│   ├── scaler.pkl               # Scaler object
│   └── imputer.pkl              # Imputer object
├── 📓 notebooks/
│   └── heart_disease_analysis.ipynb  # Jupyter analysis
├── 🔧 src/
│   ├── train_model.py           # Model training script
│   └── predict.py              # Prediction functions
├── 🌐 app.py                    # Streamlit application
├── 📋 requirements.txt          # Dependencies
└── 📖 README.md                 # Documentation
```

## 🎯 Model Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 86.9% | Excellent prediction capability |
| **AUC Score** | 0.921 | Outstanding discrimination |
| **Precision** | 87% | High true positive rate |
| **Recall** | 86% | Good disease detection rate |

## 🩺 Clinical Parameters

The model analyzes 13 critical clinical parameters:

| Parameter | Description | Clinical Significance |
|-----------|-------------|----------------------|
| **Age** | Patient age | Risk increases with age |
| **Sex** | Biological sex | Gender-based risk factors |
| **Chest Pain** | Pain type (0-3) | Coronary artery disease indicator |
| **Resting BP** | Blood pressure | Hypertension risk |
| **Cholesterol** | Serum levels | Artery plaque buildup |
| **Fasting Sugar** | Blood sugar | Diabetes indicator |
| **Resting ECG** | Electrical activity | Heart damage detection |
| **Max Heart Rate** | Exercise capacity | Cardiac function |
| **Exercise Angina** | Pain during exercise | Blood flow issues |
| **ST Depression** | ECG measurement | Ischemia indicator |
| **ST Slope** | Exercise ST segment | Coronary disease severity |
| **Major Vessels** | Fluoroscopy count | Artery blockage extent |
| **Thalassemia** | Blood flow patterns | Ischemic areas detection |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager




### Usage

1. **Navigate to Prediction Page**
2. **Enter patient clinical parameters**
3. **Click "Predict Heart Disease Risk"**
4. **View detailed risk assessment and recommendations**
5. **Explore analytics in other sections**

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository
- **Records**: 303 patient cases
- **Features**: 13 clinical parameters
- **Target**: Heart disease presence (Binary classification)
- **Missing Values**: Handled with median imputation

## 🎨 Screenshots

### 🏠 Home Page
![Home Page](https://via.placeholder.com/800x400/FF6B6B/FFFFFF?text=Heart+Disease+Prediction+System)

### 🔍 Prediction Interface
![Prediction](https://via.placeholder.com/800x400/4ECDC4/FFFFFF?text=Real-time+Risk+Assessment)

### 📈 Analytics Dashboard
![Analytics](https://via.placeholder.com/800x400/45B7D1/FFFFFF?text=Interactive+Visualizations)

## 🔧 Development

### Model Training
```python
from src.train_model import train_and_save_model
train_and_save_model()
```

### Custom Prediction
```python
from src.predict import HeartDiseasePredictor
predictor = HeartDiseasePredictor()
result = predictor.predict([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1])
```

### Adding New Features
1. Update feature engineering in `src/train_model.py`
2. Retrain the model
3. Update the Streamlit form in `app.py`
4. Test the new parameters

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)
```bash
# Push to GitHub
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main

# Deploy via share.streamlit.io
```

### Other Platforms
- **Heroku**: Use Procfile and requirements.txt
- **Hugging Face**: Docker-based deployment
- **Railway**: Automatic from GitHub
- **PythonAnywhere**: WSGI configuration

## 📈 Performance Optimization

- **Model**: Optimized Random Forest with hyperparameter tuning
- **Preprocessing**: Efficient data imputation and scaling
- **Caching**: Streamlit caching for faster reloads
- **Database**: SQLite with optimized queries

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Project Wiki](#)
- **Issues**: [GitHub Issues](https://github.com/yourusername/heart-disease-predictor/issues)
- **Email**: support@example.com
- **Discord**: [Join our community](#)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Streamlit team for the amazing framework
- Scikit-learn community for ML tools
- Medical professionals for clinical insights

## 📞 Contact

**Developer**: Your Name  
**Email**: your.email@example.com  
**GitHub**: [@yourusername](https://github.com/yourusername)  
**LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)  
**Portfolio**: [yourwebsite.com](https://yourwebsite.com)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for better healthcare through AI**

</div>

## 🎯 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced explainable AI (SHAP/LIME)
- [ ] Mobile app version
- [ ] API endpoints
- [ ] Additional medical datasets
- [ ] Real-time data integration
- [ ] Advanced visualization options
- [ ] Patient management system

---

**Disclaimer**: This tool is for educational and research purposes only. Always consult healthcare professionals for medical diagnoses.



cd heart-disease-predictor

python -m streamlit run app.py
