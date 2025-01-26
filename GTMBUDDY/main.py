import json
import joblib
import re
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class CallSnippet(BaseModel):
    text_snippet: str

class SalesCallInferenceService:
    def __init__(self, 
                 model_path='sales_call_model.joblib', 
                 vectorizer_path='tfidf_vectorizer.joblib', 
                 mlb_path='multi_label_binarizer.joblib',
                 domain_knowledge_path='domain_knowledge.json'):
        # Load pre-trained model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.mlb = joblib.load(mlb_path)
        
        # Load domain knowledge
        with open(domain_knowledge_path, 'r') as f:
            self.domain_knowledge = json.load(f)
        
        # NLP setup
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def text_cleaner(self, text):
        """Preprocess input text for inference"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)
    
    def extract_entities(self, text):
        """Extract entities from text"""
        # Domain-specific regex patterns
        patterns = {
            'competitors': r'\b(' + '|'.join(re.escape(comp) for comp in self.domain_knowledge['competitors']) + r')\b',
            'features': r'\b(' + '|'.join(re.escape(feat) for feat in self.domain_knowledge['features']) + r')\b',
            'pricing': r'\b(' + '|'.join(re.escape(kw) for kw in self.domain_knowledge['pricing_keywords']) + r')\b',
            'security': r'\b(' + '|'.join(re.escape(sec) for sec in self.domain_knowledge['security_keywords']) + r')\b',
            'pain_points': r'\b(' + '|'.join(re.escape(point) for point in self.domain_knowledge['pain_points']) + r')\b',
            'industries': r'\b(' + '|'.join(re.escape(ind) for ind in self.domain_knowledge['industry_verticals']) + r')\b',
            'decision_makers': r'\b(' + '|'.join(re.escape(role) for role in self.domain_knowledge['decision_maker_roles']) + r')\b',
            'technical_concerns': r'\b(' + '|'.join(re.escape(concern) for concern in self.domain_knowledge['technical_concerns']) + r')\b'
        }
        
        entities = {}
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[category] = matches
        
        # SpaCy NER
        doc = self.nlp(text)
        spacy_entities = {
            'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
            'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
            'locations': [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        }
        
        # Combine and filter out empty lists
        return {**entities, **spacy_entities}
    
    def generate_summary(self, text, labels):
        """Generate a brief summary of the call snippet"""
        labels_str = ', '.join(labels)
        return f"A sales call discussing {labels_str} with key business implications."

    def predict(self, text):
        """Predict labels for a given text snippet"""
        preprocessed_text = self.text_cleaner(text)
        vectorized_text = self.vectorizer.transform([preprocessed_text])
        
        # Predict probabilities
        label_probs = self.model.predict_proba(vectorized_text)[0]
        
        # Get labels with probability > 0.5
        labels_indices = np.where(label_probs > 0.5)[0]
        labels = self.mlb.classes_[labels_indices]
        
        return list(labels)

# FastAPI setup
app = FastAPI()
inference_service = SalesCallInferenceService()

@app.get("/")
def read_root():
    return {"message": "Sales Call Inference Service"}

@app.post("/predict")
def predict_call_snippet(snippet: CallSnippet):
    text_snippet = snippet.text_snippet
    
    # Perform predictions
    labels = inference_service.predict(text_snippet)
    entities = inference_service.extract_entities(text_snippet)
    summary = inference_service.generate_summary(text_snippet, labels)
    
    return {
        'labels': labels,
        'entities': entities,
        'summary': summary
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
