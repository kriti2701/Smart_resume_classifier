import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the pre-trained model
with open("resumeclassifier.joblib", "rb") as model_file:
    model = joblib.load(model_file)
tfidf_vectorizer = joblib.load('tfidf.joblib')

# Front-end code with Streamlit
st.title("Smart Resume Classifier")
st.write("Upload a CSV file and select a column to predict the candidate's domain.")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data.head())

    # Select column from CSV
    column_options = data.columns.tolist()
    selected_column = st.selectbox("Select the column with candidate data:", column_options)

    # Display a button for prediction
    if st.button("Submit"):
        # Get values from selected column
        input_data = data[[selected_column]]

        # Predict domain using the trained model
        predictions = model.predict(tfidf_vectorizer.transform(data[selected_column]))

        op_dict = { 3: 'Frontend Developer',
                    0: 'Backend Developer',
                    7: 'Python Developer',
                    2: 'Data Scientist',
                    4: 'Full Stack Developer',
                    6: 'Mobile App Developer (iOS/Android)',
                    5: 'Machine Learning Engineer',
                    1: 'Cloud Engineer'}
        

        
        data['Predicted Domain'] = predictions  # Add predictions to the DataFrame
        data['predicted_category'] = data['Predicted Domain'].map(op_dict)

        # Display the results
        st.write("Prediction Results:")
        st.write(data)
