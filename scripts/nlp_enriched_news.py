from pydoc import text

import nltk
import re
import string
import sys
from typing import List, Dict, Any
import spacy

# Ensure resources are available
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

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
class NewsNLPPipeline:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        except Exception as e:
            print(f"Error initializing NLTK components: {e}")
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
        # Extract only unique Organization names
        orgs = list({ent.text for ent in doc.ents if ent.label_ == "ORG"})
        return orgs
    def process_article(self, raw_body: str) -> Dict[str, Any]:
        """Task 2.4: Final preprocessing pipeline function."""
        try:
            if not raw_body:
                return {"sentence_count": 0, "clean_tokens": [], "stemmed_tokens": [], "processed_text": ""}

            # Sentence Tokenization
            sentences = sent_tokenize(raw_body)
            
            # Normalization
            normalized = self.normalize_text(raw_body)
            
            # Word Tokenization & Stop-word removal
            tokens = self.tokenize_and_remove_stop_words(normalized)
            
            # Stemming
            stemmed_tokens = self.apply_stemming(tokens)
            
            return {
                "sentence_count": len(sentences),
                "clean_tokens": tokens,
                "stemmed_tokens": stemmed_tokens,
                "processed_text": " ".join(stemmed_tokens)
            }
        except Exception as e:
            print(f"General error in NLP pipeline for article: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    pipeline = NewsNLPPipeline()
    sample = "The Federal Reserve is increasing interest rates. Inflation is running high!"
    result = pipeline.process_article(sample)
    print(f"Stemmed Output: {result.get('processed_text')}")