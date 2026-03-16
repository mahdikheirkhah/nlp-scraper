import pytest
import requests
from unittest.mock import patch, MagicMock
from scripts.scraper_news import fetch_html, parse_article_content

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