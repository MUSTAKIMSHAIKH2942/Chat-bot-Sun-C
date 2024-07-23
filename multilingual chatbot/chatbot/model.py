import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

def train_model():
    # Load data
    df = pd.read_csv('data/qa.csv')
    
    # Prepare training data
    X = df['question']
    y = df['answer']
    
    # Create a pipeline with TF-IDF and Logistic Regression
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    
    # Train the model
    model.fit(X, y)
    
    # Create the model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'model/chatbot_model.pkl')

def load_model():
    return joblib.load('model/chatbot_model.pkl')

if __name__ == "__main__":
    train_model()
