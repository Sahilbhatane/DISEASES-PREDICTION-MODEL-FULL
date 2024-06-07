import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout='wide')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define the base and model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
CMODEL_DIR = os.path.join(BASE_DIR, 'common-diseases-Prediction-model-main')

# Define model paths
diabetes_model_path = os.path.join(MODEL_DIR, 'diabetes_model.sav')
heart_disease_model_path = os.path.join(MODEL_DIR, 'heart_disease_model.sav')
parkinsons_model_path = os.path.join(MODEL_DIR, 'parkinsons_model.sav')
common_model = os.path.join(CMODEL_DIR, 'decision_tree_model.sav')
symptoms_list_path = os.path.join(CMODEL_DIR, 'Symptoms.txt')
with open(symptoms_list_path, 'r') as f:
    symptoms_list = f.read().splitlines()

# Load models
try:
    with open(diabetes_model_path, 'rb') as f:
        diabetes_model = pickle.load(f)
    with open(heart_disease_model_path, 'rb') as f:
        heart_disease_model = pickle.load(f)
    with open(parkinsons_model_path, 'rb') as f:
        parkinsons_model = pickle.load(f)
    with open(common_model, 'rb') as f:
        common_model = pickle.load(f)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    st.error("Error loading models. Please check the logs.")


with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Common diseases Prediction'],
                           icons=['activity', 'heart', 'person', 'thermometer'],
                           default_index=3)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        if Pregnancies and Glucose and BloodPressure and SkinThickness and Insulin and BMI and DiabetesPedigreeFunction and Age:
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            input_data = input_data.astype(float)

            try:
                prediction = diabetes_model.predict(input_data)
                if prediction[0] == 1:
                    st.success('The person is diabetic')
                else:
                    st.success('The person is not diabetic')
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                st.error("Error during prediction. Please check the logs.")
        else:
            st.warning("Please fill in all fields.")


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex = st.number_input('Sex')
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input ('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
    

#common disease model
if selected == 'Common diseases Prediction':
    st.title('Common Disease Prediction using ML')
    
    # Create multi-select dropdowns for symptoms with a limit of 2 keys per multi-select
    selected_symptoms = []
    for i in range(5):
        selected = st.multiselect(f'Select symptoms {i+1} (Max 2)', symptoms_list, key=f'symptoms_{i}')
        if len(selected) > 2:
            st.error(f"Please select only up to 2 symptoms for 'Select symptoms {i+1}'. You selected {len(selected)}.")
        selected_symptoms.extend(selected)

    common_diagnostics = ''

    if st.button('Common Disease Prediction'):
        if any(len(st.session_state[f'symptoms_{i}']) > 2 for i in range(5)):
            st.error("Please ensure no more than 2 symptoms are selected in each dropdown.")
        else:
            # Create the input vector for the model
            ipt = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
            ipt = np.array([ipt])
            
            # Make predictions
            pred = common_model.predict(ipt)[0]
            prob = common_model.predict_proba(ipt)
            
            if any(ipt[0]):
                common_diagnostics = f"Person is predicted to have: {pred}"
            else:
                common_diagnostics = "Persons diseases cannot be classified by the model, please provide appropriate symptoms."

    st.success(common_diagnostics)

st.write("Contact a Doctor- https://www.apollo247.com/specialties")
st.write("Know your Diseases - ")
st.warning('''WARNING - A medical expert, like a doctor, is best able to help you find the information and care you need. 
               This information is medical advice or diagnosis based on prediction from more than 3000+ paitents.
               Still If you Find anything abnormal rather than searching your Diseases in any domain, You should Contact a doctor First
               anything non-serious conditions can be helpful from above or anywhere else. Also Doctors Contact is also given above''')