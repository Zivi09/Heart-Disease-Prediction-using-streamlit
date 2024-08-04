import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import plotly.graph_objects as go

# Load the dataset
try:
    df = pd.read_csv('C:/Users/Ziven/Downloads/heart.csv')

    # Preprocess the data
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Model evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Streamlit app
    st.title("Heart Disease Prediction")

    # Display model evaluation metrics
    st.write("## Model Evaluation")
    st.write(f"Accuracy: {accuracy}")
    st.write("Classification Report:")
    st.text(report)

    # Initialize session state to handle reset
    if "reset" not in st.session_state:
        st.session_state.reset = False
        st.session_state.age = 0
        st.session_state.sex = ""
        st.session_state.cp = ""
        st.session_state.trestbps = 0
        st.session_state.chol = 0
        st.session_state.fbs = ""
        st.session_state.restecg = ""
        st.session_state.thalach = 0
        st.session_state.exang = ""
        st.session_state.oldpeak = 0.0
        st.session_state.slope = ""
        st.session_state.ca = 0
        st.session_state.thal = ""

    # Function to reset input fields
    def reset_fields():
        st.session_state.reset = True

    # Get user input
    st.write("## Please enter the following information:")

    with st.expander("Age: The age of the patient."):
        age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.age, key='age')

    with st.expander("Sex: The gender of the patient."):
        sex = st.selectbox("Sex", ["", "Male", "Female"], index=["", "Male", "Female"].index(st.session_state.sex), key='sex')

    with st.expander("Chest Pain Type: The type of chest pain experienced."):
        cp = st.selectbox("Chest Pain Type", ["", "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], index=["", "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(st.session_state.cp), key='cp')

    with st.expander("Resting Blood Pressure: The resting blood pressure in mm Hg."):
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=st.session_state.trestbps, key='trestbps')

    with st.expander("Cholesterol: Serum cholesterol in mg/dl."):
        chol = st.number_input("Cholesterol", min_value=0, max_value=600, value=st.session_state.chol, key='chol')

    with st.expander("Fasting Blood Sugar: Whether fasting blood sugar is greater than 120 mg/dl (True or False)."):
        fbs = st.selectbox("Fasting Blood Sugar", ["", "True", "False"], index=["", "True", "False"].index(st.session_state.fbs), key='fbs')

    with st.expander("Resting Electrocardiogram: The results of the resting electrocardiogram."):
        restecg = st.selectbox("Resting Electrocardiogram", ["", "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], index=["", "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(st.session_state.restecg), key='restecg')

    with st.expander("Maximum Heart Rate: The maximum heart rate achieved during exercise."):
        thalach = st.number_input("Maximum Heart Rate", min_value=0, max_value=220, value=st.session_state.thalach, key='thalach')

    with st.expander("Exercise Induced Angina: Whether the patient experiences angina induced by exercise (True or False)."):
        exang = st.selectbox("Exercise Induced Angina", ["", "True", "False"], index=["", "True", "False"].index(st.session_state.exang), key='exang')

    with st.expander("ST Depression: ST depression induced by exercise relative to rest."):
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=st.session_state.oldpeak, key='oldpeak')

    with st.expander("Slope of ST Segment: The slope of the peak exercise ST segment."):
        slope = st.selectbox("Slope of ST Segment", ["", "Upsloping", "Flat", "Downsloping"], index=["", "Upsloping", "Flat", "Downsloping"].index(st.session_state.slope), key='slope')

    with st.expander("Number of Major Vessels: The number of major vessels colored by fluoroscopy (0-4)."):
        ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=st.session_state.ca, key='ca')

    with st.expander("Thalassemia: A blood disorder called thalassemia."):
        thal = st.selectbox("Thalassemia", ["", "Normal", "Fixed Defect", "Reversible Defect"], index=["", "Normal", "Fixed Defect", "Reversible Defect"].index(st.session_state.thal), key='thal')

    # Check if reset button was clicked
    def reset_fields():
        st.session_state.reset = True
        st.session_state.age = 0
        st.session_state.sex = ""
        st.session_state.cp = ""
        st.session_state.trestbps = 0
        st.session_state.chol = 0
        st.session_state.fbs = ""
        st.session_state.restecg = ""
        st.session_state.thalach = 0
        st.session_state.exang = ""
        st.session_state.oldpeak = 0.0
        st.session_state.slope = ""
        st.session_state.ca = 0
        st.session_state.thal = ""
    # Convert user input to numerical values
    user_input = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "cp": [1 if cp == "Typical Angina" else 2 if cp == "Atypical Angina" else 3 if cp == "Non-Anginal Pain" else 4],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [1 if fbs == "True" else 0],
        "restecg": [1 if restecg == "Normal" else 2 if restecg == "ST-T Wave Abnormality" else 3],
        "thalach": [thalach],
        "exang": [1 if exang == "True" else 0],
        "oldpeak": [oldpeak],
        "slope": [1 if slope == "Upsloping" else 2 if slope == "Flat" else 3],
        "ca": [ca],
        "thal": [1 if thal == "Normal" else 2 if thal == "Fixed Defect" else 3]
    })

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Make a prediction
    if st.button("Predict"):
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)[0][1]  # Probability of having heart disease

        # Display the prediction
        st.write("## Prediction:")
        if prediction[0] == 1:
            st.write("You are likely to have heart disease.")
        else:
            st.write("You are unlikely to have heart disease.")

        # Display probability using a gauge chart
        st.write("## Heart Disease Risk Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            title={'text': "Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "green"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}]
            }
        ))
        st.plotly_chart(fig)

    # Reset button
    if st.button("Reset"):
        reset_fields()
        st.experimental_rerun()  # Rerun the app to update the widgets

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error(f"Exception type: {type(e).__name__}")
