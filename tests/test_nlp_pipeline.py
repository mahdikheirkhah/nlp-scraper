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
    
    
def test_scandal_flagging_positive(pipeline):
    """Scenario: A clear high-risk scandal should be flagged with strict sentiment."""
    # We use "Fraud", "Criminal", and "Disaster" because VADER knows these are very negative.
    headline = "Microsoft Corp Criminal Fraud Investigation"
    body = (
        "Authorities have launched a criminal investigation into Microsoft Corp for massive "
        "financial fraud and illegal embezzlement. The company is facing a catastrophic disaster "
        "after billions of dollars were stolen in a horrific corruption scandal."
    )
    
    result = pipeline.process_article(body, headline)
    
    # Debugging help
    print(f"\n--- Scandal Test Results ---")
    print(f"Similarity Score: {result['scandal_distance']}")
    print(f"Sentiment Score: {result['sentiment']}")
    print(f"Is Flagged: {result['is_flagged']}")

    # Check if the logic caught it
    assert result["is_flagged"] is True
    assert result["scandal_distance"] >= pipeline.SIMILARITY_THRESHOLD
    assert result["sentiment"] <= pipeline.SENTIMENT_THRESHOLD

def test_scandal_flagging_negative_news_but_no_scandal(pipeline):
    """Scenario: Sad news that isn't a 'scandal' should NOT be flagged."""
    # Sad news about stock, but not a crime/disaster
    text = "Apple shares dropped slightly today as investors expressed concern over global sales."
    result = pipeline.process_article(text)
    
    # Should be clean because similarity to "scandal themes" should be low
    assert result["is_flagged"] is False

def test_org_blacklist_logic(pipeline):
    """Scenario: Political noise like 'The Government' should be ignored."""
    # A sentence that looks like a scandal but involves a blacklisted ORG
    text = "The Government was accused of bribery and corruption in the new report."
    result = pipeline.process_article(text)
    
    # Since 'The Government' is blacklisted, scandal_score should be 0.0
    assert result["scandal_distance"] == 0.0
    assert result["is_flagged"] is False

def test_short_sentence_penalty(pipeline):
    """Scenario: Very short sentences should be penalized even if they contain bad words."""
    # "Apple fraud." is very short.
    text = "Apple fraud." 
    result = pipeline.process_article(text)
    
    # The length_multiplier 0.5 should keep this score low
    assert result["scandal_distance"] < 1.0
    assert result["is_flagged"] is False

def test_trigger_sentence_sentiment(pipeline):
    """Verify that sentiment is calculated on the trigger, not the whole article."""
    headline = "Tesla opens new factory." # Very positive
    body = "The company is growing. However, Tesla was sued for massive financial fraud in court." # Very negative
    
    result = pipeline.process_article(body, headline)
    
    # The 'trigger' is the fraud sentence, so sentiment should be negative
    # even though the headline was positive.
    assert result["sentiment"] < 0
    assert "sued" in result["trigger_sentence"]

def test_tripwire_word_boost(pipeline):
    """Verify that specific words from TRIPWIRE_WORDS increase the score."""
    text_normal = "Facebook is discussing new data policies."
    text_risky = "Facebook is facing a massive data breach." # 'breach' is a tripwire
    
    res_normal = pipeline.process_article(text_normal)
    res_risky = pipeline.process_article(text_risky)
    
    assert res_risky["scandal_distance"] > res_normal["scandal_distance"]

def test_no_org_no_scandal(pipeline):
    """If no organization is found, no scandal can be flagged."""
    text = "A massive data breach happened somewhere in the world involving fraud."
    result = pipeline.process_article(text)
    
    assert result["scandal_distance"] == 0.0
    assert result["is_flagged"] is False

def test_high_similarity_neutral_sentiment(pipeline):
    """Scenario: High similarity but neutral/positive sentiment shouldn't flag."""
    # A training exercise or a book title about scandals
    text = "Google released a documentary about how to prevent Corporate Embezzlement."
    result = pipeline.process_article(text)
    
    # High similarity to 'Embezzlement' theme, but sentiment is likely not <= -0.6
    if result["sentiment"] > pipeline.SENTIMENT_THRESHOLD:
        assert result["is_flagged"] is False