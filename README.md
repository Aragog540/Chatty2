

# Chatty - AI-Powered Medical Disease Diagnosis Chatbot

Chatty is an intelligent chatbot designed to assist users in diagnosing potential diseases based on user-provided symptoms. Using Natural Language Processing (NLP) and machine learning, Chatty predicts over 50 diseases and provides detailed descriptions along with precautionary advice to guide users toward better health decisions.

---

## Features
- **Accurate Disease Prediction**: Predicts over 50 diseases based on symptoms entered in natural language.
- **NLP-Powered Analysis**: Uses TF-IDF vectorization and a LinearSVC model for symptom classification.
- **Detailed Insights**: Offers descriptions and precautionary measures for each predicted disease.
- **Scalable and Customizable**: Easily extendable to include more diseases and symptoms.

---

## Installation

### Prerequisites
- Python 3.7 or above
- Required Python libraries: `numpy`, `pandas`, `scikit-learn`, `joblib`

### Clone the Repository
```bash
git clone https://github.com/Aragog540/Chatty2
cd chatty
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Model
Ensure your dataset is named `dataset.csv` and located in the project directory. Run the following script to train the model:

```bash
python train_model.py
```

This will save the trained model at the specified path.

### Predict Diseases
To use Chatty for disease prediction, run:
```bash
python predict.py
```
You can input symptoms, and Chatty will output predicted diseases, descriptions, and precautions.

---

## Dataset Structure

The model uses a dataset containing symptoms and corresponding diseases. The structure includes:
- `Symptom_1` to `Symptom_17`: Columns for symptoms.
- `Disease`: The target column for disease classification.

Additional datasets:
- `symptom_Description.csv`: Provides disease descriptions.
- `symptom_precaution.csv`: Contains precautionary advice for diseases.

---

## Model Overview

The model pipeline consists of:
1. **TF-IDF Vectorizer**: Converts symptoms into feature vectors.
2. **LinearSVC**: Classifies symptoms into disease categories.

Performance metrics such as accuracy, confusion matrix, and classification report are displayed after training.

---


## Future Enhancements
- Expand the dataset to include more diseases and symptoms.
- Integrate a user-friendly web or mobile interface.
- Incorporate multi-language support.

---


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

---

## Contact
For queries or suggestions, reach out to:
- **Email**: swaroopbhowmik7@gmail.com
```
