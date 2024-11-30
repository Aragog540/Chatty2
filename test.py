import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import joblib  
import os



df = pd.read_csv(r'dataset.csv')
ds = pd.read_csv(r'symptom_Description.csv')
pr = pd.read_csv(r'symptom_precaution.csv')


df = df.fillna("")


df['Symptom'] = ""
for i in range(1, 18):
    df['Symptom'] += df[f"Symptom_{i}"] + " "

df = df.drop([f"Symptom_{i}" for i in range(1, 18)], axis=1)

X = df['Symptom']
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=44)

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, y_train)

model_save_path = r'C:\Users\swaro\Desktop\chatty2\disease_prediction_model.pkl'
joblib.dump(text_clf, model_save_path)

print(f"Model saved at: {model_save_path}")

predictions = text_clf.predict(X_test)
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", metrics.classification_report(y_test, predictions))
print(f"\nAccuracy: {metrics.accuracy_score(y_test, predictions) * 100:.2f}%")


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

def get_description_precautions(disease):
    description = ds.loc[disease, 'Description'] if disease in ds.index else "No description available."
    precautions = pr.loc[disease, 'precautions'] if disease in pr.index else "No precautions available."
    return description, precautions

for idx, pred in enumerate(predictions[:5]):  
    description, precautions = get_description_precautions(pred)
    print(f"\nDisease: {pred}")
    print(f"Description: {description}")
    print(f"Precautions: {precautions}")
