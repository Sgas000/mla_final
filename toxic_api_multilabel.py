from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

app = FastAPI(title="Multi-label Toxic Comment Classifier")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_text: TextInput):
    vect_text = vectorizer.transform([input_text.text])
    prediction = model.predict(vect_text)[0]

    result = {
        label: bool(pred)
        for label, pred in zip(categories, prediction)
    }

    return {
        "input": input_text.text,
        "prediction": result
    }
