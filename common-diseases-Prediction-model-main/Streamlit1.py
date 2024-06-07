import pickle
import streamlit as st
import numpy as np
import json

# Load the trained model
with open('decision_tree_model.sav', 'rb') as model_file:
    common_model = pickle.load(model_file)

# Load the list of symptoms from the file
with open('Symptoms.txt', 'r') as symptoms_file:
    symptoms_list = json.load(symptoms_file)

# Set the layout of the Streamlit page
st.set_page_config(layout='wide')

# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox('Multiple Disease Prediction System',
                            ['Diabetes Prediction',
                             'Heart Disease Prediction',
                             'Parkinsons Prediction',
                             'Common diseases Prediction'])

# Common diseases prediction
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
