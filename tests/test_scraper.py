import pytest
import requests
from unittest.mock import patch, MagicMock
from scripts.scraper_news import fetch_html, parse_article_content
from scripts.run_snowflake import get_snowflake_engine


def test_get_snowflake_engine_success():
    """
    Tests successful creation of the Snowflake engine.
    Mocks the RSA key loading and SQLAlchemy engine creation.
    """
    try:
        with patch('os.getenv') as mock_env, \
             patch('scripts.scraper_news.serialization.load_pem_private_key') as mock_load_key, \
             patch('scripts.scraper_news.create_engine') as mock_create_engine:
            
            # Setup mocks
            mock_env.side_effect = lambda k: {
                "SNOWFLAKE_PRIVATE_KEY": "fake_key",
                "SNOWFLAKE_USER": "test_user",
                "SNOWFLAKE_ACCOUNT": "test_account"
            }.get(k)
            
            # Mock the key object and its bytes representation
            mock_key_obj = MagicMock()
            mock_load_key.return_value = mock_key_obj
            mock_key_obj.private_bytes.return_value = b"fake_der_bytes"
            
            # Mock the returned engine
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            result = get_snowflake_engine()
            
            assert result == mock_engine
            mock_create_engine.assert_called_once()
    except Exception as e:
        pytest.fail(f"get_snowflake_engine_success failed unexpectedly: {e}")

def test_get_snowflake_engine_invalid_key():
    """
    Tests that the function returns None if the RSA key is malformed.
    Validates the internal try-except block for cryptographic errors.
    """
    with patch('os.getenv', return_value="invalid_key"), \
         patch('scripts.scraper_news.serialization.load_pem_private_key') as mock_load_key:
        
        # Simulate a cryptography error
        mock_load_key.side_effect = ValueError("Could not deserialize key data")
        
        result = get_snowflake_engine()
        
        # The try-except should catch the ValueError and return None
        assert result is None

def test_get_snowflake_engine_missing_env():
    """
    Tests that the function handles missing environment variables gracefully.
    """
    with patch('os.getenv', return_value=None):
        # This will cause an AttributeError when .replace() is called on None
        result = get_snowflake_engine()
        assert result is None

def test_fetch_html_success():
    """Tests successful HTML retrieval with try-except safety."""
    try:
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = "<html><body><h1>Test</h1></body></html>"
            
            result = fetch_html("https://fake-url.com")
            assert "Test" in result
    except Exception as e:
        pytest.fail(f"Test failed due to unexpected error: {e}")

def test_fetch_html_failure():
    """
    Tests that fetch_html returns None when a requests error occurs.
    This validates the internal try-except block in scraper_news.py.
    """
    with patch('requests.get') as mock_get:
        # We use RequestException to match the 'except' block in the source code
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.RequestException("404 Error")
        
        result = fetch_html("https://invalid-url.com")
        assert result is None

def test_parse_article_content_valid():
    """Tests extraction of h1 and p tags from valid HTML."""
    html = "<html><body><h1>The Headline</h1><p>The body text.</p></body></html>"
    result = parse_article_content(html)
    
    assert result['headline'] == "The Headline"
    assert "The body text." in result['body']

def test_parse_article_content_missing_elements():
    """Tests fallback values when HTML is missing headline or body."""
    html = "<html><body></body></html>"
    result = parse_article_content(html)
    
    assert result['headline'] == "No Headline"
    assert result['body'] == ""

def test_parse_article_content_error_handling():
    """Tests that the parser handles malformed HTML via its try-except block."""
    try:
        # Passing None to trigger the internal try-except in parse_article_content
        result = parse_article_content(None) 
        assert result['headline'] == "Error"
    except Exception as e:
        pytest.fail(f"Parser try-except did not catch the error: {e}")