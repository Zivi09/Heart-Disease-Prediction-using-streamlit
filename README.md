# Heart Disease Prediction App

## Overview

This is a Streamlit app that uses a logistic regression model to predict the likelihood of a patient having heart disease based on various clinical features. The app takes user input for 13 features, scales the data, and makes a prediction using the trained model. The app also displays the prediction probability using a gauge chart.

## Features

- **Age**: The age of the patient
- **Sex**: The gender of the patient (Male/Female)
- **Chest Pain Type**: The type of chest pain experienced (Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic)
- **Resting Blood Pressure**: The resting blood pressure in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar**: Whether fasting blood sugar is greater than 120 mg/dl (True/False)
- **Resting Electrocardiogram**: The results of the resting electrocardiogram (Normal, ST-T Wave Abnormality, Left Ventricular Hypertrophy)
- **Maximum Heart Rate**: The maximum heart rate achieved during exercise
- **Exercise Induced Angina**: Whether the patient experiences angina induced by exercise (True/False)
- **ST Depression**: ST depression induced by exercise relative to rest
- **Slope of ST Segment**: The slope of the peak exercise ST segment (Upsloping, Flat, Downsloping)
- **Number of Major Vessels**: The number of major vessels colored by fluoroscopy (0-4)
- **Thalassemia**: A blood disorder called thalassemia (Normal, Fixed Defect, Reversible Defect)

## Requirements

- Python 3.8+
- Streamlit 1.10+
- Scikit-learn 1.0+
- Pandas 1.3+
- Plotly 5.5+

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/heart-disease-prediction-app.git
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run heart disease prediction.py
    ```

4. Open a web browser and navigate to [http://localhost:8501](http://localhost:8501)

5. Enter the patient's features and click the "Predict" button to get the prediction.

6. Click the "Reset" button to reset the input fields.

## Model Evaluation

The logistic regression model is trained on the Cleveland Heart Disease dataset and evaluated using accuracy score, classification report, and confusion matrix. The model's performance is displayed in the app.

## Note

This app is for educational purposes only and should not be used for actual medical diagnosis or treatment. Always consult a healthcare professional for accurate diagnosis and treatment.
