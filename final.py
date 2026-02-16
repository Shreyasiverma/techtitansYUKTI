import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="LifeGuard AI", page_icon="🩺", layout="wide")

# CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ LifeGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Advanced Disease Prediction & Specialist Finder</p>", unsafe_allow_html=True)
st.divider()

# Symptoms
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever',
'yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision',
'phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose',
'congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity',
'swollen_legs','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'excessive_hunger','slurred_speech','knee_pain','hip_joint_pain',
'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness',
'spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
'continuous_feel_of_urine','passage_of_gases','internal_itching',
'depression','irritability','muscle_pain','belly_pain',
'abnormal_menstruation','watering_from_eyes','increased_appetite',
'polyuria','family_history','lack_of_concentration']

# Medical Info
medical_info = {
    "Fungal infection": {"precautions": ["Keep skin dry", "Avoid tight clothes"], "doctor": "Dermatologist"},
    "Allergy": {"precautions": ["Avoid allergens", "Wear mask outdoors"], "doctor": "Allergist"},
    "GERD": {"precautions": ["Avoid spicy food", "Avoid lying down after meals"], "doctor": "Gastroenterologist"},
    "Chronic cholestasis": {"precautions": ["Avoid alcohol", "Low fat diet"], "doctor": "Hepatologist"},
    "Drug Reaction": {"precautions": ["Stop medication", "Consult doctor"], "doctor": "Dermatologist"},
    "Peptic ulcer disease": {"precautions": ["Avoid spicy food", "Regular meals"], "doctor": "Gastroenterologist"},
    "AIDS": {"precautions": ["Safe practices", "Regular follow-up"], "doctor": "Infectious Disease Specialist"},
    "Diabetes": {"precautions": ["Low sugar diet", "Daily exercise"], "doctor": "Endocrinologist"},
    "Gastroenteritis": {"precautions": ["Drink ORS", "Eat light food"], "doctor": "General Physician"},
    "Bronchial Asthma": {"precautions": ["Avoid dust", "Keep inhaler handy"], "doctor": "Pulmonologist"},
    "Hypertension": {"precautions": ["Low salt diet", "Reduce stress"], "doctor": "Cardiologist"},
    "Migraine": {"precautions": ["Dark room rest", "Avoid loud noise"], "doctor": "Neurologist"},
    "Cervical spondylosis": {"precautions": ["Neck exercises", "Proper pillow"], "doctor": "Orthopedic Surgeon"},
    "Paralysis (brain hemorrhage)": {"precautions": ["Immediate hospitalization", "Physiotherapy"], "doctor": "Neurologist"},
    "Jaundice": {"precautions": ["Rest", "Avoid oily food"], "doctor": "Hepatologist"},
    "Malaria": {"precautions": ["Mosquito net", "Stay hydrated"], "doctor": "Infectious Disease Specialist"},
    "Chicken pox": {"precautions": ["Isolation", "Oatmeal baths"], "doctor": "General Physician"},
    "Dengue": {"precautions": ["Check platelets", "Stay hydrated"], "doctor": "General Physician"},
    "Typhoid": {"precautions": ["Boiled water", "Light diet"], "doctor": "General Physician"},
    "Hepatitis A": {"precautions": ["Avoid alcohol", "Clean food"], "doctor": "Hepatologist"},
    "Tuberculosis": {"precautions": ["Masking", "Finish medicine course"], "doctor": "Pulmonologist"},
    "Common Cold": {"precautions": ["Steam inhalation", "Warm fluids"], "doctor": "General Physician"},
    "Pneumonia": {"precautions": ["Chest physiotherapy", "Rest"], "doctor": "Pulmonologist"},
    "Heartattack": {"precautions": ["Emergency call", "Chew aspirin"], "doctor": "Cardiologist"},
    "Hypothyroidism": {"precautions": ["Regular checkup", "Healthy diet"], "doctor": "Endocrinologist"},
    "Hyperthyroidism": {"precautions": ["Medication adherence", "Avoid stress"], "doctor": "Endocrinologist"},
    "Osteoarthritis": {"precautions": ["Weight management", "Light exercise"], "doctor": "Orthopedic Surgeon"},
    "Arthritis": {"precautions": ["Joint protection", "Warm compress"], "doctor": "Rheumatologist"},
    "Acne": {"precautions": ["Clean face", "Avoid oily cosmetics"], "doctor": "Dermatologist"},
    "Urinary tract infection": {"precautions": ["Drink water", "Hygiene"], "doctor": "Urologist"},
    "Psoriasis": {"precautions": ["Moisturize", "Avoid triggers"], "doctor": "Dermatologist"}
}

# Data Loading and Training
@st.cache_resource
def train_models():
    try:
        df = pd.read_csv("Training.csv")
        le = LabelEncoder()
        df['prognosis'] = le.fit_transform(df['prognosis'])
        X = df[l1]
        y = df['prognosis']
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100).fit(X, y),
        }
        return models, le
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

models, le = train_models()

# UI Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Consultance", "🏥 Nearby Hospitals", "🚨 Emergency"])

with tab1:
    st.header("Predict Disease based on Symptoms")
    if models is not None:
        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            model_option = st.selectbox("Select AI Model:", list(models.keys()))
        with col_sel2:
            user_symptoms = st.multiselect("Select your symptoms:", sorted(l1))

        if st.button("Consultance"):
            if user_symptoms:
                input_vector = np.zeros(len(l1))
                for s in user_symptoms:
                    if s in l1:
                        input_vector[l1.index(s)] = 1
                
                prediction_idx = models[model_option].predict([input_vector])[0]
                predicted_disease = le.inverse_transform([prediction_idx])[0]

                st.success(f"### Predicted Condition: **{predicted_disease}**")

                info = medical_info.get(predicted_disease, {"precautions": ["Consult a specialist"], "doctor": "General Physician"})
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📋 Recommended Precautions")
                    for p in info["precautions"]:
                        st.write(f"🔹 {p}")
                
                with c2:
                    st.subheader("👨‍⚕️ Specialist to Consult")
                    doc_type = info["doctor"]
                    st.info(f"Condition suggests visiting a: **{doc_type}**")
                    
                    # Correct Maps link
                    maps_url = f"https://www.google.com/maps/search/{doc_type.replace(' ', '+')}+near+me"
                    st.markdown(f'''
                        <a href="{maps_url}" target="_blank">
                            <button style="background-color: #28a745; color: white; padding: 12px; border: none; border-radius: 8px; cursor: pointer; width: 100%; font-size: 16px;">
                                📍 Find {doc_type} Near Me
                            </button>
                        </a>
                    ''', unsafe_allow_html=True)
            else:
                st.error("Please select at least one symptom.")

with tab2:
    st.header("Search Nearby Medical Facilities")
    facility_query = st.text_input("Search for:", "Best Hospitals")
    facility_url = f"https://www.google.com/maps/search/{facility_query.replace(' ', '+')}+near+me"
    st.markdown(f'<a href="{facility_url}" target="_blank"><button style="background-color:#17a2b8; color:white; padding:15px; width:100%; border:none; border-radius:10px;">🔍 Find "{facility_query}" Near Me</button></a>', unsafe_allow_html=True)

with tab3:
    st.header("Emergency Helplines")
    col1, col2 = st.columns(2)
    with col1:
        st.error("🚑 Ambulance: 102")
        st.markdown("[📞 Click to Call 102](tel:102)")
    with col2:
        st.error("🚑 Emergency: 108")
        st.markdown("[📞 Click to Call 108](tel:108)")

st.divider()
st.caption("Disclaimer: This AI is for informational purposes only. Always seek professional medical advice.")