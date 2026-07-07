# ❤️ Heart Disease Risk Prediction Model

## IIT Jammu - Machine Learning Assignment (Week-3)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Assignment Information](#assignment-information)
- [Dataset](#dataset)
- [Features](#features)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

---

## 📖 Overview

This project builds a machine learning model to predict the **10-year risk of coronary heart disease (CHD)** using the Framingham Heart Study dataset. The model is deployed as an interactive **Streamlit web application** with a user-friendly interface for healthcare risk assessment.

### Key Highlights:
- ✅ Multiple ML models comparison (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ Hyperparameter tuning for optimal performance
- ✅ Feature engineering and selection
- ✅ Interactive web application with real-time predictions
- ✅ Comprehensive visualizations and risk assessment

---

## 🎓 Assignment Information

| **Category** | **Details** |
|-------------|-------------|
| **Institution** | IIT Jammu |
| **Course** | Machine Learning |
| **Assignment** | Week-3 |
| **Topic** | Heart Disease Prediction |
| **Submission Date** | {date} |
| **Techniques** | Supervised Learning, Ensemble Methods |

**Learning Objectives:**
1. Understand and implement classification algorithms
2. Perform feature engineering and selection
3. Evaluate model performance using appropriate metrics
4. Deploy ML models using Streamlit
5. Create interactive data visualization

---

## 📊 Dataset

### Framingham Heart Study Dataset

The dataset contains medical records of patients from the Framingham Heart Study, a long-term cardiovascular study.

| **Property** | **Value** |
|-------------|-----------|
| **Source** | Framingham Heart Study |
| **Total Samples** | 4,240 |
| **Features** | 15 original + 4 derived |
| **Target Variable** | TenYearCHD (0 = No, 1 = Yes) |
| **CHD Prevalence** | ~15.2% |
| **Time Period** | 10-year follow-up |

---

## 🔍 Features

### Original Features (15)
| Feature | Description | Type |
|---------|-------------|------|
| `male` | Gender (0=Female, 1=Male) | Binary |
| `age` | Age in years | Continuous |
| `education` | Education level (1-4) | Ordinal |
| `currentSmoker` | Current smoker status | Binary |
| `cigsPerDay` | Cigarettes smoked per day | Discrete |
| `BPMeds` | Blood pressure medication | Binary |
| `prevalentStroke` | History of stroke | Binary |
| `prevalentHyp` | History of hypertension | Binary |
| `diabetes` | Diabetes diagnosis | Binary |
| `totChol` | Total cholesterol (mg/dL) | Continuous |
| `sysBP` | Systolic blood pressure (mmHg) | Continuous |
| `diaBP` | Diastolic blood pressure (mmHg) | Continuous |
| `BMI` | Body Mass Index (kg/m²) | Continuous |
| `heartRate` | Heart rate (beats/min) | Continuous |
| `glucose` | Glucose level (mg/dL) | Continuous |

### Derived Features (4)
| Feature | Formula | Purpose |
|---------|---------|---------|
| `age_squared` | age² | Capture non-linear age effect |
| `bmi_age` | BMI × age | Interaction effect |
| `bp_ratio` | sysBP / diaBP | Blood pressure ratio |
| `cholesterol_bp` | totChol × sysBP | Interaction effect |

---

## 📈 Model Performance

### Model Comparison
| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| **Logistic Regression** | 0.852 | 0.879 | 0.456 |
| **Random Forest** | 0.856 | 0.883 | 0.461 |
| **Gradient Boosting** | **0.861** | **0.891** | **0.468** |

### Best Model: Gradient Boosting

**Performance Metrics:**
- ✅ **Accuracy:** 86.1%
- ✅ **ROC-AUC:** 89.1%
- ✅ **Precision:** 0.48
- ✅ **Recall:** 0.45
- ✅ **F1-Score:** 0.468

### Top 10 Important Features
1. **age** - Most significant predictor
2. **sysBP** - Systolic blood pressure
3. **totChol** - Total cholesterol
4. **BMI** - Body Mass Index
5. **cigsPerDay** - Smoking intensity
6. **glucose** - Blood sugar level
7. **diaBP** - Diastolic blood pressure
8. **heartRate** - Heart rate
9. **diabetes** - Diabetes status
10. **age_squared** - Non-linear age effect

# ❤️ Heart Disease Risk Prediction Model

## IIT Jammu - Machine Learning Assignment (Week-3)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Assignment Information](#assignment-information)
- [Dataset](#dataset)
- [Features](#features)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

---

## 📖 Overview

This project builds a machine learning model to predict the **10-year risk of coronary heart disease (CHD)** using the Framingham Heart Study dataset. The model is deployed as an interactive **Streamlit web application** with a user-friendly interface for healthcare risk assessment.

### Key Highlights:
- ✅ Multiple ML models comparison (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ Hyperparameter tuning for optimal performance
- ✅ Feature engineering and selection
- ✅ Interactive web application with real-time predictions
- ✅ Comprehensive visualizations and risk assessment

---

## 🎓 Assignment Information

| **Category** | **Details** |
|-------------|-------------|
| **Institution** | IIT Jammu |
| **Course** | Machine Learning |
| **Assignment** | Week-3 |
| **Topic** | Heart Disease Prediction |
| **Submission Date** | {date} |
| **Techniques** | Supervised Learning, Ensemble Methods |

**Learning Objectives:**
1. Understand and implement classification algorithms
2. Perform feature engineering and selection
3. Evaluate model performance using appropriate metrics
4. Deploy ML models using Streamlit
5. Create interactive data visualization

---

## 📊 Dataset

### Framingham Heart Study Dataset

The dataset contains medical records of patients from the Framingham Heart Study, a long-term cardiovascular study.

| **Property** | **Value** |
|-------------|-----------|
| **Source** | Framingham Heart Study |
| **Total Samples** | 4,240 |
| **Features** | 15 original + 4 derived |
| **Target Variable** | TenYearCHD (0 = No, 1 = Yes) |
| **CHD Prevalence** | ~15.2% |
| **Time Period** | 10-year follow-up |

---

## 🔍 Features

### Original Features (15)
| Feature | Description | Type |
|---------|-------------|------|
| `male` | Gender (0=Female, 1=Male) | Binary |
| `age` | Age in years | Continuous |
| `education` | Education level (1-4) | Ordinal |
| `currentSmoker` | Current smoker status | Binary |
| `cigsPerDay` | Cigarettes smoked per day | Discrete |
| `BPMeds` | Blood pressure medication | Binary |
| `prevalentStroke` | History of stroke | Binary |
| `prevalentHyp` | History of hypertension | Binary |
| `diabetes` | Diabetes diagnosis | Binary |
| `totChol` | Total cholesterol (mg/dL) | Continuous |
| `sysBP` | Systolic blood pressure (mmHg) | Continuous |
| `diaBP` | Diastolic blood pressure (mmHg) | Continuous |
| `BMI` | Body Mass Index (kg/m²) | Continuous |
| `heartRate` | Heart rate (beats/min) | Continuous |
| `glucose` | Glucose level (mg/dL) | Continuous |

### Derived Features (4)
| Feature | Formula | Purpose |
|---------|---------|---------|
| `age_squared` | age² | Capture non-linear age effect |
| `bmi_age` | BMI × age | Interaction effect |
| `bp_ratio` | sysBP / diaBP | Blood pressure ratio |
| `cholesterol_bp` | totChol × sysBP | Interaction effect |

---

## 📈 Model Performance

### Model Comparison
| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| **Logistic Regression** | 0.852 | 0.879 | 0.456 |
| **Random Forest** | 0.856 | 0.883 | 0.461 |
| **Gradient Boosting** | **0.861** | **0.891** | **0.468** |

### Best Model: Gradient Boosting

**Performance Metrics:**
- ✅ **Accuracy:** 86.1%
- ✅ **ROC-AUC:** 89.1%
- ✅ **Precision:** 0.48
- ✅ **Recall:** 0.45
- ✅ **F1-Score:** 0.468

### Top 10 Important Features
1. **age** - Most significant predictor
2. **sysBP** - Systolic blood pressure
3. **totChol** - Total cholesterol
4. **BMI** - Body Mass Index
5. **cigsPerDay** - Smoking intensity
6. **glucose** - Blood sugar level
7. **diaBP** - Diastolic blood pressure
8. **heartRate** - Heart rate
9. **diabetes** - Diabetes status
10. **age_squared** - Non-linear age effect

---

## 📁 Project Structure
heart-disease-prediction/
│
├── framingham.csv # Dataset file
├── Heart_Disease_Prediction.ipynb # Jupyter notebook
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── run_app.bat # Windows launcher
├── run_app.sh # Mac/Linux launcher
├── README.md # Project documentation
│
└── model_outputs/ # Generated output folder
├── charts/ # Visualization charts
│ ├── missing_values.png
│ ├── target_distribution.png
│ ├── correlation_matrix.png
│ ├── feature_boxplots.png
│ ├── feature_distributions.png
│ ├── preliminary_feature_importance.png
│ ├── model_comparison.png
│ ├── confusion_matrix.png
│ ├── roc_curves.png
│ └── tuned_feature_importance.png
│
├── models/ # Saved model files
│ ├── best_model.pkl
│ ├── scaler.pkl
│ ├── imputer.pkl
│ ├── feature_names.pkl
│ └── model_summary.json
│
├── data/ # Data templates
│ ├── preprocessed_data.csv
│ ├── sample_data.csv
│ ├── data_template.csv
│ ├── data_dictionary.json
│ └── feature_importance.csv
│
├── predict_heart_disease.py # Standalone prediction script
└── README.md # Documentation
