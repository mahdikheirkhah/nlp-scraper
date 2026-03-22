import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

import re
import string
from typing import List
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class NewsNLPPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def normalize_text(self, text: str) -> str:
        """Task 2.1: Lowercasing and punctuation removal."""
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation using regex
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace/newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_remove_stop_words(self, text: str) -> List[str]:
        """Task 2.2: Word tokenization and stop-word filtering."""
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        return filtered_tokens

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """Task 2.3: Stemming to reduce words to their roots (e.g., 'running' -> 'run')."""
        return [self.stemmer.stem(token) for token in tokens]

    def process_article(self, raw_body: str) -> dict:
        """Task 2.4: Final preprocessing pipeline function."""
        # 1. Sentence Tokenization (keep original for context if needed)
        sentences = sent_tokenize(raw_body)
        
        # 2. Normalization
        normalized = self.normalize_text(raw_body)
        
        # 3. Word Tokenization & Stop-word removal
        tokens = self.tokenize_and_remove_stop_words(normalized)
        
        # 4. Stemming
        stemmed_tokens = self.apply_stemming(tokens)
        
        return {
            "sentence_count": len(sentences),
            "clean_tokens": tokens,
            "stemmed_tokens": stemmed_tokens,
            "processed_text": " ".join(stemmed_tokens)
        }

# Quick Test logic
if __name__ == "__main__":
    pipeline = NewsNLPPipeline()
    sample = "The Federal Reserve is increasing interest rates. Inflation is running high!"
    result = pipeline.process_article(sample)
    print(f"Original: {sample}")
    print(f"Stemmed: {result['stemmed_tokens']}")