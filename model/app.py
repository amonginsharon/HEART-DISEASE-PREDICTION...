import streamlit as st
import pandas as pd
import pickle

def main():
    set_page_style()

    st.title("Heart Disease Prediction")
    # User input fields
    age = st.number_input("Age", value=0, min_value=0, max_value=150, step=1)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    height = st.number_input("Height (cm)", value=0.0, min_value=0.0, step=1.0)
    weight = st.number_input("Weight (kg)", value=0.0, min_value=0.0, step=0.1)
    ap_hi = st.number_input("Systolic Blood Pressure", value=0, min_value=0, max_value=300, step=1)
    ap_lo = st.number_input("Diastolic Blood Pressure", value=0, min_value=0, max_value=300, step=1)
    cholesterol = st.selectbox("Cholesterol", options=["Normal", "Above Normal", "Well Above Normal"])
    gluc = st.selectbox("Glucose", options=["Normal", "Above Normal", "Well Above Normal"])
    smoke = st.selectbox("Smoking", options=["No", "Yes"])
    alco = st.selectbox("Alcohol Consumption", options=["No", "Yes"])
    active = st.selectbox("Physical Activity", options=["No", "Yes"])

    # Convert categorical inputs to numeric
    gender = 1 if gender == "Male" else 0
    cholesterol = convert_label(cholesterol)
    gluc = convert_label(gluc)
    smoke = 1 if smoke == "Yes" else 0
    alco = 1 if alco == "Yes" else 0
    active = 1 if active == "Yes" else 0

    # Create feature DataFrame
    data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "smoke": [smoke],
        "alco": [alco],
        "active": [active]
    })

    # Load the trained model
    model = load_model()

    # Make prediction
    if st.button("Predict"):
        result = predict(model, data)
        st.write("Prediction:", result)

def set_page_style():
    # Set page style
    page_bg_color = "#0074D9"  # Blue color (Hex: #0074D9)
    text_color = "#FFFFFF"  # White color
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {page_bg_color};
            color: {text_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def load_model():
    # Load the pre-trained model
    with open("model_pickle.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def predict(model, data):
    # Perform prediction using the loaded model
    result = model.predict(data)

    # Convert the numeric result back to descriptive label
    if result == 0:
        return "Great! You are fit. It's time to celebrate."
    elif result == 1:
        return "Oops! You are most likely to have a disease."
    elif result == 2:
        return "Oops! You are most likely to have a disease."

def convert_label(label):
    if label == "Normal":
        return 0
    elif label == "Above Normal":
        return 1
    elif label == "Well Above Normal":
        return 2

if __name__ == "__main__":
    main()
