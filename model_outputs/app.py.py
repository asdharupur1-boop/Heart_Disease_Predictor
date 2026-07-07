"""
Heart Disease Prediction App
A Streamlit web application for predicting 10-year CHD risk
IIT Jammu - Assignment Work Week-3
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor - IIT Jammu",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .risk-low {
        background-color: #51cf66;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .risk-moderate {
        background-color: #fcc419;
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
        font-size: 1.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .social-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .social-btn {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .github-btn {
        background-color: #24292e;
        color: white;
    }
    .github-btn:hover {
        background-color: #2d3748;
        transform: scale(1.05);
        color: white;
    }
    .linkedin-btn {
        background-color: #0077b5;
        color: white;
    }
    .linkedin-btn:hover {
        background-color: #0088cc;
        transform: scale(1.05);
        color: white;
    }
    .assignment-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">❤️ Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IIT Jammu - Assignment Work Week-3 | Machine Learning Project</div>', unsafe_allow_html=True)

# Model loading function
@st.cache_resource
def load_model():
    """Load the saved model and preprocessing objects"""
    try:
        model_dir = 'model_outputs/models'
        
        # Check if model files exist
        required_files = ['best_model.pkl', 'scaler.pkl', 'imputer.pkl', 'feature_names.pkl']
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                st.error(f"❌ Model file not found: {file}")
                st.info("Please train the model first using the Jupyter notebook.")
                return None, None, None, None
        
        # Load model files
        model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
        features = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        
        return model, scaler, imputer, features
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model is trained and saved in 'model_outputs/models/'")
        return None, None, None, None

# Load model
model, scaler, imputer, features = load_model()

# Load model summary
@st.cache_resource
def load_model_summary():
    try:
        with open('model_outputs/models/model_summary.json', 'r') as f:
            return json.load(f)
    except:
        return None

model_summary = load_model_summary()

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    if model_summary:
        st.info(f"**Best Model:** {model_summary.get('best_model', 'N/A')}")
        st.info(f"**Accuracy:** {model_summary.get('test_accuracy', 0):.2%}")
        st.info(f"**ROC-AUC:** {model_summary.get('test_roc_auc', 0):.2%}")
    
    st.markdown("---")
    st.header("📋 Instructions")
    st.markdown("""
    1. Fill in all patient details
    2. Click **Predict Risk** button
    3. View the risk assessment
    4. Review feature contribution
    
    ⚠️ **Note:** This is a prediction tool. Always consult with healthcare professionals.
    """)
    
    st.markdown("---")
    
    # Assignment Information
    st.header("🎓 IIT Jammu Assignment")
    st.markdown("""
    **Course:** Machine Learning  
    **Week:** 3  
    **Topic:** Heart Disease Prediction  
    **Dataset:** Framingham Heart Study  
    **Model:** Ensemble Learning
    """)
    
    st.markdown("---")
    
    # Social Links
    st.header("🔗 Connect With Me")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <a href="https://github.com/asdharupur1-boop" target="_blank" style="text-decoration: none;">
            <div style="background: #24292e; padding: 0.5rem; border-radius: 5px; text-align: center; color: white;">
                <b>🐙 GitHub</b>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href="https://www.linkedin.com/in/ayush-shukla-ds/" target="_blank" style="text-decoration: none;">
            <div style="background: #0077b5; padding: 0.5rem; border-radius: 5px; text-align: center; color: white;">
                <b>🔗 LinkedIn</b>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Made with ❤️ for IIT Jammu Assignment")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📝 Patient Input", "📊 Results", "📈 Feature Analysis", "ℹ️ About"])

# Initialize session state for results
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probability' not in st.session_state:
    st.session_state.probability = None

# Input tab
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Personal Information")
        
        male = st.selectbox("Gender", ["Female", "Male"], help="Select gender")
        male_val = 1 if male == "Male" else 0
        
        age = st.slider("Age (years)", 20, 80, 50, help="Age in years")
        
        education = st.selectbox(
            "Education Level",
            ["1 - Less than high school", "2 - High school", "3 - Some college", "4 - College"],
            help="Education level"
        )
        education_val = int(education.split(" - ")[0])
        
        currentSmoker = st.selectbox("Current Smoker", ["No", "Yes"])
        currentSmoker_val = 1 if currentSmoker == "Yes" else 0
        
        cigsPerDay = st.number_input("Cigarettes per Day", 0, 80, 0, help="Number of cigarettes smoked daily")
        
        BPMeds = st.selectbox("Blood Pressure Medication", ["No", "Yes"])
        BPMeds_val = 1 if BPMeds == "Yes" else 0
    
    with col2:
        st.subheader("🩺 Medical History")
        
        prevalentStroke = st.selectbox("History of Stroke", ["No", "Yes"])
        prevalentStroke_val = 1 if prevalentStroke == "Yes" else 0
        
        prevalentHyp = st.selectbox("History of Hypertension", ["No", "Yes"])
        prevalentHyp_val = 1 if prevalentHyp == "Yes" else 0
        
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        diabetes_val = 1 if diabetes == "Yes" else 0
        
        totChol = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 220, help="Normal: < 200 mg/dL")
        
        glucose = st.number_input("Glucose Level (mg/dL)", 40, 400, 90, help="Normal: < 100 mg/dL")
    
    with col3:
        st.subheader("💓 Vital Signs")
        
        sysBP = st.number_input("Systolic BP (mmHg)", 80, 300, 120, help="Normal: < 120 mmHg")
        
        diaBP = st.number_input("Diastolic BP (mmHg)", 50, 200, 80, help="Normal: < 80 mmHg")
        
        BMI = st.number_input("BMI (kg/m²)", 15, 60, 25, help="Normal: 18.5 - 24.9")
        
        heartRate = st.number_input("Heart Rate (bpm)", 40, 150, 75, help="Normal: 60 - 100 bpm")
    
    # Prediction button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("🔬 Predict Heart Disease Risk", type="primary", use_container_width=True)

# Prediction logic
if predict_button:
    if model is None:
        st.error("❌ Model not loaded. Please train the model first.")
    else:
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'male': [male_val],
                'age': [age],
                'education': [education_val],
                'currentSmoker': [currentSmoker_val],
                'cigsPerDay': [cigsPerDay],
                'BPMeds': [BPMeds_val],
                'prevalentStroke': [prevalentStroke_val],
                'prevalentHyp': [prevalentHyp_val],
                'diabetes': [diabetes_val],
                'totChol': [totChol],
                'sysBP': [sysBP],
                'diaBP': [diaBP],
                'BMI': [BMI],
                'heartRate': [heartRate],
                'glucose': [glucose]
            })
            
            # Create derived features
            input_data['age_squared'] = input_data['age'] ** 2
            input_data['bmi_age'] = input_data['BMI'] * input_data['age']
            input_data['bp_ratio'] = input_data['sysBP'] / input_data['diaBP']
            input_data['cholesterol_bp'] = input_data['totChol'] * input_data['sysBP']
            
            # Ensure all features are present
            for feature in features:
                if feature not in input_data.columns:
                    input_data[feature] = 0
            
            input_data = input_data[features]
            
            # Impute and scale
            input_imputed = pd.DataFrame(imputer.transform(input_data), columns=features)
            input_scaled = pd.DataFrame(scaler.transform(input_imputed), columns=features)
            
            # Predict
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            # Store in session state
            st.session_state.prediction = prediction[0]
            st.session_state.probability = probability[0][1]
            st.session_state.input_data = input_data
            st.session_state.feature_importance = None
            
            # Get feature importance for this prediction
            if hasattr(model, 'feature_importances_'):
                st.session_state.feature_importance = dict(zip(features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
                st.session_state.feature_importance = dict(zip(features, importance))
            
            # Auto-switch to results tab
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Results tab
with tab2:
    if st.session_state.prediction is not None:
        pred = st.session_state.prediction
        prob = st.session_state.probability
        
        st.markdown("---")
        
        # Display risk score
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                title = {'text': "CHD Risk Probability"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#e74c3c"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': '#51cf66'},
                        {'range': [20, 40], 'color': '#fcc419'},
                        {'range': [40, 60], 'color': '#ff922b'},
                        {'range': [60, 80], 'color': '#e74c3c'},
                        {'range': [80, 100], 'color': '#c0392b'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk classification
        if prob < 0.2:
            st.markdown(f'<div class="risk-low">✅ Low Risk - {prob*100:.1f}% Probability of CHD</div>', unsafe_allow_html=True)
            st.success("📌 Maintain healthy lifestyle and regular check-ups")
        elif prob < 0.4:
            st.markdown(f'<div class="risk-moderate">⚠️ Moderate Risk - {prob*100:.1f}% Probability of CHD</div>', unsafe_allow_html=True)
            st.warning("📌 Consider lifestyle changes and consult healthcare provider")
        else:
            st.markdown(f'<div class="risk-high">🚨 High Risk - {prob*100:.1f}% Probability of CHD</div>', unsafe_allow_html=True)
            st.error("📌 Consult healthcare professional immediately")
        
        # Additional metrics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Category", "High" if prob > 0.4 else "Moderate" if prob > 0.2 else "Low")
        with col2:
            st.metric("Confidence Level", f"{max(prob, 1-prob)*100:.1f}%")
        with col3:
            st.metric("Risk Score", f"{prob*100:.1f}%")
        
        # Disclaimer
        st.markdown("---")
        st.info("ℹ️ This prediction is based on statistical models and should not replace professional medical advice.")
        
    else:
        st.info("👈 Enter patient details and click 'Predict Heart Disease Risk' to see results")

# Feature Analysis tab
with tab3:
    if st.session_state.prediction is not None:
        st.subheader("📊 Feature Contribution Analysis")
        
        if st.session_state.feature_importance:
            # Sort features by importance
            importance_dict = st.session_state.feature_importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Create dataframe for visualization
            importance_df = pd.DataFrame(sorted_features[:15], columns=['Feature', 'Importance'])
            
            # Bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Feature Contributions',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show input data
        with st.expander("📋 Patient Input Data"):
            if 'input_data' in st.session_state:
                st.dataframe(st.session_state.input_data)
        
        # Recommendations based on risk
        st.subheader("💡 Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Lifestyle Changes**")
            st.markdown("""
            - 🏃 Regular exercise (30 mins, 5x/week)
            - 🥗 Healthy diet (low salt, high fiber)
            - 🚭 Quit smoking
            - 🍷 Limit alcohol consumption
            """)
        
        with col2:
            st.markdown("**Medical Considerations**")
            st.markdown("""
            - 💊 Take prescribed medications
            - 🩺 Regular health check-ups
            - 📊 Monitor blood pressure
            - 🧪 Track cholesterol levels
            """)
        
    else:
        st.info("👈 Please make a prediction first to see feature analysis")

# About tab
with tab4:
    st.markdown("""
    ## ℹ️ About This Application
    
    ### 📚 Assignment Information
    """)
    
    st.markdown("""
    <div class="assignment-box">
        <h3>🎓 IIT Jammu - Machine Learning Assignment</h3>
        <p><strong>Week:</strong> 3</p>
        <p><strong>Topic:</strong> Heart Disease Prediction using Framingham Dataset</p>
        <p><strong>Objective:</strong> Build and deploy a machine learning model for predicting 10-year risk of coronary heart disease</p>
        <p><strong>Techniques Used:</strong> Logistic Regression, Random Forest, Gradient Boosting</p>
        <p><strong>Key Features:</strong> 15 original features + 4 derived features</p>
        <p><strong>Model Performance:</strong> Accuracy: ~0.85 | ROC-AUC: ~0.88</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 👨‍💻 Developer Information
    """)
    
    # Social links section
    st.markdown("""
    <div class="social-links">
        <a href="https://github.com/asdharupu1-boop" target="_blank" class="social-btn github-btn">
            🐙 GitHub
        </a>
        <a href="https://www.linkedin.com/in/ayush-shukla-ds/" target="_blank" class="social-btn linkedin-btn">
            🔗 LinkedIn
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Or use columns for better control
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 1.1rem;">
                <b>Student:</b> [Your Name]<br>
                <b>Course:</b> Machine Learning<br>
                <b>Institution:</b> IIT Jammu<br>
                <b>Assignment:</b> Week-3<br>
                <b>Date:</b> {date}
            </p>
        </div>
        """.format(date=datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔧 Technologies Used
    
    | Category | Technologies |
    |----------|-------------|
    | **Programming** | Python 3.8+ |
    | **ML Framework** | Scikit-learn |
    | **Web Framework** | Streamlit |
    | **Visualization** | Plotly, Matplotlib, Seaborn |
    | **Data Processing** | Pandas, NumPy |
    | **Model Persistence** | Joblib |
    
    ### 📊 Dataset Information
    - **Source:** Framingham Heart Study
    - **Samples:** 4,240
    - **Features:** 15 original + 4 derived
    - **Target:** 10-year CHD risk
    
    ### 🎯 Model Performance
    - **Best Model:** {best_model}
    - **Accuracy:** {accuracy:.2%}
    - **ROC-AUC:** {roc_auc:.2%}
    """.format(
        best_model=model_summary.get('best_model', 'N/A') if model_summary else 'N/A',
        accuracy=model_summary.get('test_accuracy', 0) if model_summary else 0,
        roc_auc=model_summary.get('test_roc_auc', 0) if model_summary else 0
    ))
    
    st.markdown("---")
    
    st.markdown("""
    ### 📝 Disclaimer
    ⚠️ This application is for **educational purposes** and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.
    
    ### 📧 Contact
    For questions or feedback regarding this assignment, please reach out via:
    - GitHub: [github.com/yourusername](https://github.com/asdharupu1-boop)
    - LinkedIn: [linkedin.com/in/yourusername](https://www.linkedin.com/in/ayush-shukla-ds/)
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2026 IIT Jammu - Machine Learning Assignment Week-3 | Heart Disease Prediction Model</p>
    <p style="font-size: 0.8rem;">
        <a href="https://github.com/asdharupur1-boop/Heart_Disease_Predictor" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/ayush-shukla-ds/" target="_blank">LinkedIn</a> | 
        IIT Jammu
    </p>
</div>
""", unsafe_allow_html=True)

# Health tips (random)
tips = [
    "💡 Maintain a healthy weight to reduce heart disease risk",
    "💡 Regular physical activity strengthens your heart",
    "💡 A balanced diet rich in fruits and vegetables is heart-healthy",
    "💡 Manage stress through meditation or yoga",
    "💡 Get adequate sleep (7-8 hours) for heart health",
    "💡 Monitor your blood pressure regularly",
    "💡 Know your family history of heart disease"
]

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Health Tip:**")
st.sidebar.info(np.random.choice(tips))