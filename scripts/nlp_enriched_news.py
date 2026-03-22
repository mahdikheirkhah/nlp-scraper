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
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.sia = SentimentIntensityAnalyzer()
        
        # Task 4.2: Define Environmental Scandal Keywords
        self.scandal_keywords = [
            nlp("environmental disaster"),
            nlp("oil spill"),
            nlp("toxic waste leakage"),
            nlp("pollution lawsuit"),
            nlp("illegal dumping"),
            nlp("carbon emission fraud")
        ]

        # Load Topic Model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.topic_model = pickle.load(f)
        else:
            self.topic_model = None

    def analyze_sentiment(self, text: str) -> float:
        """Task 4.1: Get compound sentiment score (-1 to 1)."""
        return self.sia.polarity_scores(text)['compound']

    def calculate_scandal_score(self, sentences: List[str], orgs: List[str]) -> float:
        """Task 4.3 & 4.4: Calculate distance to disaster keywords."""
        if not orgs or not sentences:
            return 0.0
        
        max_similarity = 0.0
        # Find sentences that actually mention one of our detected companies
        for sent in sentences:
            if any(org.lower() in sent.lower() for org in orgs):
                sent_doc = nlp(sent)
                # Compare sentence vector to each risk keyword vector
                for keyword in self.scandal_keywords:
                    sim = sent_doc.similarity(keyword)
                    if sim > max_similarity:
                        max_similarity = sim
        return round(max_similarity, 4)

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
        """Full Phase 4 Intelligence Pipeline."""
        clean_headline = (headline or "").strip()
        if clean_headline and not clean_headline.endswith(('.', '!', '?')):
            clean_headline += "."
        
        full_content = f"{clean_headline} {(raw_body or '').strip()}".strip()
        sentences = sent_tokenize(full_content)
        
        # 1. Base Logic
        normalized = self.normalize_text(full_content)
        # 2. Topic & NER
        orgs = self.extract_organizations(full_content)
        topic = self.predict_topic(full_content)
        
        # 3. Phase 4 Intelligence
        sentiment = self.analyze_sentiment(full_content)
        scandal_score = self.calculate_scandal_score(sentences, orgs)
        
        # Risk Logic: High Scandal + Low Sentiment = High Risk
        is_scandal = True if (scandal_score > 0.7 and sentiment < 0) else False

        return {
            "topic": topic,
            "orgs": orgs,
            "sentiment": sentiment,
            "scandal_distance": scandal_score,
            "is_flagged": is_scandal,
            "sentence_count": len(sentences),
            "processed_text": " ".join(self.apply_stemming(self.tokenize_and_remove_stop_words(normalized)))
        }


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