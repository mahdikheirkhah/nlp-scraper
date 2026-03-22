import nltk
import re
import string
import pickle
import os
from typing import List, Dict, Any
import spacy

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load spaCy for Entity Detection (Task 3.1)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class NewsNLPPipeline:
    def __init__(self, model_path='results/topic_classifier.pkl'):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            
            # Load the Topic Classifier (Task 3.2)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.topic_model = pickle.load(f)
            else:
                self.topic_model = None
                print(f"Warning: {model_path} not found. Topic detection will be disabled.")
        except Exception as e:
            print(f"Error initializing NLP components: {e}")
            self.stop_words = set()

    def normalize_text(self, text: str) -> str:
        """Task 2.1: Lowercasing and punctuation removal."""
        try:
            if not isinstance(text, str) or not text:
                return ""
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            print(f"Error in normalization: {e}")
            return ""

    def tokenize_and_remove_stop_words(self, text: str) -> List[str]:
        """Task 2.2: Word tokenization and stop-word filtering."""
        try:
            if not text:
                return []
            tokens = word_tokenize(text)
            return [w for w in tokens if w not in self.stop_words]
        except Exception as e:
            print(f"Error in word tokenization: {e}")
            return []

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """Task 2.3: Stemming."""
        try:
            if not tokens:
                return []
            return [self.stemmer.stem(token) for token in tokens]
        except Exception as e:
            print(f"Error in stemming: {e}")
            return tokens

    def extract_organizations(self, text: str) -> list:
        """Task 3.1: Extract ORG entities using SpaCy."""
        if not text: return []
        doc = nlp(text)
        # Extract unique Organization names
        return list({ent.text for ent in doc.ents if ent.label_ == "ORG"})

    def predict_topic(self, text: str) -> str:
        """Task 3.2: Predict topic using the trained model."""
        if self.topic_model and text:
            # We pass raw text because the TF-IDF in the pipeline handles its own cleaning
            prediction = self.topic_model.predict([text])
            return prediction[0]
        return "Unknown"

    def process_article(self, raw_body: str, headline: str = "") -> Dict[str, Any]:
        """Task 2.4 & Phase 3 Integration."""
        try:
            if not raw_body:
                return {"sentence_count": 0, "orgs": [], "topic": "Unknown", "processed_text": ""}

            # Full text for intelligence tasks
            full_content = f"{headline} {raw_body}".strip()

            # 1. Base Preprocessing (Phase 2)
            sentences = sent_tokenize(raw_body)
            normalized = self.normalize_text(raw_body)
            tokens = self.tokenize_and_remove_stop_words(normalized)
            stemmed_tokens = self.apply_stemming(tokens)
            
            # 2. Information Extraction (Phase 3)
            orgs = self.extract_organizations(full_content)
            topic = self.predict_topic(full_content)
            
            return {
                "sentence_count": len(sentences),
                "clean_tokens": tokens,
                "stemmed_tokens": stemmed_tokens,
                "processed_text": " ".join(stemmed_tokens),
                "orgs": orgs,
                "topic": topic
            }
        except Exception as e:
            print(f"General error in NLP pipeline: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Ensure you have results/topic_classifier.pkl before running
    pipeline = NewsNLPPipeline()
    
    sample_headline = "Apple reaches record stock price"
    sample_body = "The company Apple Inc. reported massive profits today in Cupertino. Tech analysts are impressed."
    
    result = pipeline.process_article(sample_body, sample_headline)
    print(f"--- Analysis Result ---")
    print(f"Topic: {result['topic']}")
    print(f"Organizations: {result['orgs']}")
    print(f"Stemmed Preview: {result['processed_text'][:50]}...")