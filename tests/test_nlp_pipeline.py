import pytest
from scripts.nlp_enriched_news import NewsNLPPipeline

@pytest.fixture
def pipeline():
    return NewsNLPPipeline()

def test_normalization(pipeline):
    """Test Task 2.1: Lowercase and punctuation."""
    raw = "Hello, World! This is a TEST."
    expected = "hello world this is a test"
    assert pipeline.normalize_text(raw) == expected

def test_tokenization_and_stopwords(pipeline):
    """Test Task 2.2: Word splitting and stop-word removal."""
    # 'is', 'a', 'the' should be removed
    text = "this is a test the news"
    tokens = pipeline.tokenize_and_remove_stop_words(text)
    assert "test" in tokens
    assert "news" in tokens
    assert "is" not in tokens

def test_stemming(pipeline):
    """Test Task 2.3: Stemming roots."""
    tokens = ["increasing", "interest", "rates"]
    stemmed = pipeline.apply_stemming(tokens)
    # Porter Stemmer usually turns 'increasing' to 'increas'
    assert "increas" in stemmed
    assert "rate" in stemmed

def test_empty_pipeline(pipeline):
    """Test Task 2.4: Handling of empty or None inputs."""
    result = pipeline.process_article(None)
    assert result["sentence_count"] == 0
    assert result["processed_text"] == ""

def test_full_pipeline_flow(pipeline):
    """Verify the entire flow from string to processed text."""
    sample = "Banks are lending money."
    result = pipeline.process_article(sample)
    # bank (stemmed from banks), lend (stemmed from lending), money
    assert "bank" in result["processed_text"]
    assert "lend" in result["processed_text"]