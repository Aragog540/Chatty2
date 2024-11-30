# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:48:11 2024

@author: swaro
"""

import joblib
import pandas as pd

# Define the file path to load the model
model_path = r'C:\Users\swaro\Desktop\chatty2\disease_prediction_model.pkl'

# Load the trained model
text_clf = joblib.load(model_path)
print("Model loaded successfully.")

# Load the supplementary data for description and precautions
ds = pd.read_csv(r'symptom_Description.csv')
pr = pd.read_csv(r'symptom_precaution.csv')

# Preprocess the supplementary data
ds.index = ds['Disease']
ds = ds.drop('Disease', axis=1)

pr = pr.fillna("")  
pr['precautions'] = ""
pr['punc'] = ', '
for i in range(1, 5):
    pr['precautions'] += pr[f"Precaution_{i}"] + pr['punc']
pr = pr.drop([f"Precaution_{i}" for i in range(1, 5)], axis=1)
pr = pr.drop(['punc'], axis=1)

pr.index = pr['Disease']
pr = pr.drop('Disease', axis=1)

# Function to get disease description and precautions
def get_description_precautions(disease):
    description = ds.loc[disease, 'Description'] if disease in ds.index else "No description available."
    precautions = pr.loc[disease, 'precautions'] if disease in pr.index else "No precautions available."
    return description, precautions

# Predict using the loaded model
while True:
    symptoms = input("Enter symptoms (or 'exit' to quit): ").strip()
    if symptoms.lower() == 'exit':
        break

    # Predict the disease
    prediction = text_clf.predict([symptoms])[0]
    description, precautions = get_description_precautions(prediction)

    # Display the result
    print(f"\nPredicted Disease: {prediction}")
    print(f"Description: {description}")
    print(f"Precautions: {precautions}\n")
