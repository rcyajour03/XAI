import streamlit as st
import pandas as pd
import pickle
from interpret import show
import re
import io
import sys
from matplotlib import pyplot as plt
from io import BytesIO

# Capture the output of show() to extract the URL
class StreamToLogger(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = None

    def write(self, message):
        super().write(message)
        # Check if the message contains the URL pattern
        match = re.search(r'http://\S+', message)
        if match:
            self.url = match.group(0)

# Load the saved heart disease model
filename = "heart_disease_classification_tree.sav"
model = pickle.load(open(filename, "rb"))

# Streamlit UI
st.title('Heart Disease Predictions using ML')

introduction = pd.DataFrame({
    "Title": ["Age", "Sex", "cp", "Trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    "Description": ["Age", "Gender", "chest pain", "resting blood pressure", "cholesterol", "fasting blood pressure",
                    "Resting electrocardiographic results", "maximum heart rate achieved", "exercised induced angina",
                    "ST depression induced by exercise", 'The slope of the peak exercise ST segment', 
                    "Number of major vessels (0-3) colored by fluoroscopy", 
                    "0: Normal 1: Fixed 2: Reversible"]
})
st.table(introduction)

# User input fields for heart disease features
col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input('Age')
    
with col2:
    sex = st.text_input('Sex')
    
with col3:
    cp = st.text_input('Chest Pain types')
    
with col1:
    trestbps = st.text_input('Resting Blood Pressure')
    
with col2:
    chol = st.text_input('Serum Cholesterol in mg/dl')
    
with col3:
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    
with col1:
    restecg = st.text_input('Resting Electrocardiographic results')
    
with col2:
    thalach = st.text_input('Maximum Heart Rate achieved')
    
with col3:
    exang = st.text_input('Exercise Induced Angina')
    
with col1:
    oldpeak = st.text_input('ST depression induced by exercise')
    
with col2:
    slope = st.text_input('Slope of the peak exercise ST segment')
    
with col3:
    ca = st.text_input('Major vessels colored by fluoroscopy')
    
with col1:
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')


input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Prediction and result display
if st.button('Heart Disease Test Result'):
    input_data = [52.0, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, 0, 2]
    input_data = [i for i in input_data]
    # input_data = [eval(i) for i in input_data]
    
    # Make prediction
    prediction = model.predict([input_data])
    
    if prediction[0] == 1:
        st.warning('The person is having heart disease')
    else:
        st.success('The person does not have any heart disease')
    
    # Redirect stdout to capture output from show
    logger = StreamToLogger()
    sys.stdout = logger


    explainer = model.explain_global(name='Global Tree Explanation')  # Pass current input and prediction
    show(explainer) # This opens the explanation in a browser tab

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Get the extracted URL
    iframe_url = logger.url

    # Embed the generated URL in an iframe if available
    if iframe_url:
        st.components.v1.iframe(src=iframe_url, width=600, height=800)
    else:
        st.write("No URL found for the decision tree explanation.")

# Optional: Show statistics based on user input
if st.button("Show statistics"):
    avgHeart = [52.0, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, 0, 2]  # Example average values
    thisData = [eval(i) for i in input_data]
    Features = ["Age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    
    for i in range(len(avgHeart)):
        x = ["Average", "You"]
        y = [avgHeart[i], thisData[i]]
        fig = plt.figure(figsize=(3, 3))
        plt.bar(x, y, label=Features[i], color=["#902fed", "yellow"])
        plt.legend()
        
        # Display the chart in the app
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
