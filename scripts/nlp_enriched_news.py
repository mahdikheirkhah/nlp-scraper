import nltk
import re
import string
import pickle
import os
import spacy
from typing import List, Dict, Any
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure resources
try:
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer




class NewsNLPPipeline:
    def __init__(self, model_path='results/topic_classifier.pkl'):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.sia = SentimentIntensityAnalyzer()
        
        self.SIMILARITY_THRESHOLD = 0.85  # Very High. Requires the sentence to be almost a direct match to a scandal.
        self.SENTIMENT_THRESHOLD = -0.6   # Very Negative. Filters out "bad news" and keeps "disastrous news."
        
        # Task 4.1: TRIPWIRE_WORDS (The "Hard" Check)
        # 30 high-risk words per 8 categories = 240 words
        self.TRIPWIRE_WORDS = {
            # Business & Finance
            "embezzlement", "liquidation", "insolvency", "bankrupt", "monopoly", "antitrust", "laundering", "fraud", "scam", "insider",
            "offshore", "tax-evasion", "foreclosure", "default", "bailout", "audit", "deficit", "nepotism", "kickback", "bribery",
            "malfeasance", "ponzi", "collusion", "price-fixing", "sweatshop", "boycott", "sanctions", "layoff", "severance", "arbitration",
            # Technology
            "breach", "hack", "cyberattack", "ransomware", "spyware", "malware", "backdoor", "exploit", "zero-day", "leak",
            "deepfake", "dark-web", "phishing", "botnet", "surveillance", "encryption-break", "data-harvesting", "identity-theft", "troll-farm", "algorithm-bias",
            "censorship", "shadow-ban", "deplatform", "outage", "vulnerability", "piracy", "scraping", "impersonation", "spy", "tracking",
            # Politics & Government
            "corruption", "impeachment", "insurrection", "coup", "treason", "filibuster", "gerrymander", "lobbyist", "scandal", "misconduct",
            "propaganda", "extortion", "indictment", "subpoena", "perjury", "wiretap", "whistleblower", "unconstitutional", "authoritarian", "autocracy",
            "tyranny", "protest", "riot", "anarchy", "regime", "militia", "espionage", "diplomatic-rift", "veto", "controversy",
            # Health & Science
            "malpractice", "outbreak", "epidemic", "pandemic", "contamination", "opioid", "overdose", "falsification", "plagiarism", "unethical",
            "biohazard", "radiation", "carcinogen", "toxic", "recall", "counterfeit", "misdiagnosis", "clinical-fail", "side-effect", "mutation",
            "quackery", "anti-vax", "pathogen", "superbug", "infection", "fatality", "mortality", "negligence", "unlicensed", "asbestos",
            # Environment
            "spill", "deforestation", "poaching", "extinction", "fracking", "smog", "emissions", "global-warming", "drought", "flood",
            "wildfire", "radioactive", "pesticide", "landfill", "sewage", "dumping", "erosion", "tsunami", "earthquake", "melting",
            "methane", "drilling", "poaching", "illegal-logging", "pipeline-leak", "microplastics", "acid-rain", "overfishing", "desertification", "habitat-loss",
            # Education & Arts (Scandals)
            "cheating", "hazing", "plagiarism", "admission-scandal", "defunded", "tenure-revoked", "strike", "expulsion", "harassment", "assault",
            "forgery", "theft", "vandalism", "censored", "royalty-dispute", "cancel-culture", "blackface", "appropriation", "propaganda", "misogyny",
            "racism", "homophobia", "abuse", "overdose", "rehab", "paparazzi", "lawsuit", "allegation", "divorce", "bankruptcy"
        }

        # Task 4.2: RISK THEMES (The "Semantic" Check)
        # Combined 15 thematic "risk sentences" per topic
        risk_themes_raw = [
            "Corporate embezzlement and financial fraud investigation", "Antitrust monopoly lawsuit and price fixing",
            "Massive data breach leaking sensitive user passwords", "Government corruption and bribery of public officials",
            "Environmental disaster involving toxic chemical spills", "Academic plagiarism and college admission scandals",
            "Medical malpractice resulting in patient fatalities", "Labor strikes and human rights violations in factories",
            "Cyberattack on infrastructure using sophisticated ransomware", "Illegal dumping of hazardous waste in protected areas",
            "Political insurrection and attempts to overthrow democracy", "Retraction of scientific papers due to data falsification",
            "Celebrity allegations of sexual harassment and assault", "Public health crisis involving counterfeit medication",
            "Tax evasion through offshore shell companies", "Systemic racism and discrimination in corporate hiring",
            "Privacy violations through illegal mass surveillance", "Bankruptcy and collapse of major global banks",
            "Product recalls due to life threatening defects", "Ethical misconduct in high level clinical trials"
        ]
        self.scandal_keywords = [nlp(theme) for theme in risk_themes_raw]

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.topic_model = pickle.load(f)
        else:
            self.topic_model = None

    def analyze_sentiment(self, text: str) -> float:
        return self.sia.polarity_scores(text)['compound']
    
    def calculate_scandal_score(self, sentences: List[str], orgs: List[str]) -> Dict[str, Any]:
        # 💡 Filter out "Noise ORGs" that aren't specific companies
        org_blacklist = {"government", "the government", "congress", "senate", "parliament", "police", "court", "the city"}
        valid_orgs = [o for o in orgs if o.lower() not in org_blacklist]
        
        if not valid_orgs or not sentences:
            return {"score": 0.0, "trigger": "N/A"}
        
        max_score = 0.0
        best_trigger = "None"
        
        for sent in sentences:
            sent_lower = sent.lower()
            # Only check if a VALID company is in the sentence
            if any(org.lower() in sent_lower for org in valid_orgs):
                sent_doc = nlp(sent)
                
                # A. Base Similarity
                sim_score = max([sent_doc.similarity(k) for k in self.scandal_keywords])
                
                # B. Tripwire Boost (Reduced even further)
                clean_words = set(re.findall(r'\w+', sent_lower))
                match_count = len(clean_words.intersection(self.TRIPWIRE_WORDS))
                tripwire_boost = min(match_count * 0.02, 0.08) 
                
                # C. Length Requirement (Scandals need detail)
                length_multiplier = 1.0 if len(clean_words) >= 12 else 0.5
                
                total_sent_score = round(min((sim_score + tripwire_boost) * length_multiplier, 1.0), 4)
                
                if total_sent_score > max_score:
                    max_score = total_sent_score
                    best_trigger = sent.strip()
                    
        return {"score": max_score, "trigger": best_trigger}
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
        """Sharpened Phase 4 Pipeline."""
        clean_headline = (headline or "").strip()
        if clean_headline and not clean_headline.endswith(('.', '!', '?')):
            clean_headline += "."
        
        full_content = f"{clean_headline} {(raw_body or '').strip()}".strip()
        sentences = sent_tokenize(full_content)
        
        # 1. Intelligence (Run on raw text for better accuracy)
        orgs = self.extract_organizations(full_content)
        topic = self.predict_topic(full_content)
        
        # 2. Scandal & Local Sentiment
        scandal_data = self.calculate_scandal_score(sentences, orgs)
        
        # 💡 FIX: Check sentiment of the trigger sentence, not just the whole article
        trigger_sent = scandal_data['trigger']
        if trigger_sent != "N/A":
            trigger_sentiment = self.analyze_sentiment(trigger_sent)
        else:
            trigger_sentiment = self.analyze_sentiment(full_content)
            
        # 3. Flagging Logic
        # We flag if the score is high AND the specific trigger sentence is negative
        is_scandal = (scandal_data['score'] >= self.SIMILARITY_THRESHOLD and 
                      trigger_sentiment <= self.SENTIMENT_THRESHOLD)

        # 4. Final Processing (Only for storage/Task 2 requirements)
        normalized = self.normalize_text(full_content)
        stemmed_preview = " ".join(self.apply_stemming(self.tokenize_and_remove_stop_words(normalized)))

        return {
            "topic": topic,
            "orgs": orgs,
            "sentiment": trigger_sentiment, # Return the sentiment that mattered
            "scandal_distance": scandal_data['score'],
            "trigger_sentence": trigger_sent,
            "is_flagged": is_scandal,
            "sentence_count": len(sentences),
            "processed_text": stemmed_preview
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