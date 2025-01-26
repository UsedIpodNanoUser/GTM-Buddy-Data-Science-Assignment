FROM python:3.9-slim

WORKDIR /app

#Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Download NLTK and spaCy resources
RUN python -m nltk.downloader punkt stopwords wordnet
RUN python -m spacy download en_core_web_sm

#Copy application files AND pre-trained model files
COPY main.py .
COPY sales_call_model.joblib /app/
COPY tfidf_vectorizer.joblib /app/
COPY multi_label_binarizer.joblib /app/
COPY domain_knowledge.json /app/

#Exose port
EXPOSE 8000

#Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]