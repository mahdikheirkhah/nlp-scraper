import pytest
import os
from scripts.nlp_enriched_news import NewsNLPPipeline

@pytest.fixture
def pipeline():
    # Ensure the model exists or the tests will use "Unknown"
    return NewsNLPPipeline()


def test_normalization(pipeline):
    """Test Task 2.1: Lowercase and punctuation."""
    raw = "Hello, World! This is a TEST."
    expected = "hello world this is a test"
    assert pipeline.normalize_text(raw) == expected

def test_tokenization_and_stopwords(pipeline):
    """Test Task 2.2: Word splitting and stop-word removal."""
    text = "this is a test the news"
    tokens = pipeline.tokenize_and_remove_stop_words(text)
    assert "test" in tokens
    assert "news" in tokens
    assert "is" not in tokens

def test_stemming(pipeline):
    """Test Task 2.3: Stemming roots."""
    tokens = ["increasing", "interest", "rates"]
    stemmed = pipeline.apply_stemming(tokens)
    assert "increas" in stemmed
    assert "rate" in stemmed


def test_extract_organizations(pipeline):
    """Task 3.1: Verify SpaCy can find multiple unique ORGs."""
    text = "Apple and Microsoft are competing in the cloud market with Amazon."
    orgs = pipeline.extract_organizations(text)
    assert "Apple" in orgs
    assert "Microsoft" in orgs
    assert "Amazon" in orgs
    # Verify uniqueness (set logic)
    text_with_duplicates = "Google is better than Google."
    assert len(pipeline.extract_organizations(text_with_duplicates)) == 1

def test_topic_prediction_logic(pipeline):
    """Task 3.2: Verify topic classification (if model exists)."""
    if pipeline.topic_model is None:
        pytest.skip("Topic classifier model not found in results/. Skipping prediction test.")
    
    # Simple tech scenario
    tech_text = "The new iPhone features a faster processor and better battery life."
    assert pipeline.predict_topic(tech_text) == "tech"
    
    # Simple business scenario
    biz_text = "The stock market crashed after the central bank raised interest rates."
    assert pipeline.predict_topic(biz_text) == "business"

def test_full_pipeline_with_headline(pipeline):
    """Task 2.4 & Phase 3: Verify headline + body integration."""
    # Use 'Tesla Inc' to make it easier for the small SpaCy model to be 100% sure
    headline = "Tesla Inc stock rises"
    body = "Elon Musk announced new profits for the car company."
    
    result = pipeline.process_article(body, headline)
    
    assert "Tesla Inc" in result["orgs"]
    assert result["sentence_count"] > 0
    assert result["topic"] in ["business", "tech", "Unknown"]

def test_multiple_orgs_scenario(pipeline):
    """Scenario: Multiple companies in a single piece of text."""
    text = "Google and Meta are facing new regulations from the EU."
    result = pipeline.process_article(text)
    
    assert "Google" in result["orgs"]
    assert "Meta" in result["orgs"]

def test_mixed_casing_resilience(pipeline):
    """Scenario: The pipeline should handle weird casing for normalization but keep it for NER."""
    headline = "APPLE IS BOOMING"
    body = "the company based in Cupertino."
    
    result = pipeline.process_article(body, headline)
    
    # Preprocessing should be lowercase
    assert "appl" in result["processed_text"] 
    # NER should find the Org despite the headline being ALL CAPS
    assert "APPLE" in result["orgs"]

def test_noise_and_short_text(pipeline):
    """Scenario: Handling very short or noisy text."""
    text = "Just a quick update."
    result = pipeline.process_article(text)
    
    assert result["sentence_count"] == 1
    assert result["orgs"] == [] # No orgs here
    assert isinstance(result["topic"], str)

def test_duplicate_orgs(pipeline):
    """Scenario: Ensure organizations are not duplicated in the results."""
    text = "Microsoft is hiring. Microsoft is a great place to work."
    result = pipeline.process_article(text)
    
    # The count of 'Microsoft' in the list should be exactly 1
    assert result["orgs"].count("Microsoft") == 1

def test_weird_formatting(pipeline):
    """Test handling of multiple newlines and spaces."""
    raw = "Multiple    spaces\n\nand\nnewlines."
    normalized = pipeline.normalize_text(raw)
    assert "  " not in normalized # Should be collapsed to single space
    assert "\n" not in normalized

def test_empty_inputs(pipeline):
    """Verify the pipeline doesn't crash on empty strings."""
    result = pipeline.process_article("", "")
    assert result["sentence_count"] == 0
    assert result["orgs"] == []
    assert result["topic"] == "Unknown"

def test_none_input(pipeline):
    """Ensure None doesn't crash the script."""
    # normalize_text returns empty string for None
    assert pipeline.normalize_text(None) == ""
    # predict_topic returns Unknown for None
    assert pipeline.predict_topic(None) == "Unknown"