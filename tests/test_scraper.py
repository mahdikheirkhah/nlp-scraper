import pytest
import requests
import os
from unittest.mock import patch, MagicMock
from scripts.scraper_news import fetch_html, parse_article_content, save_articles_to_snowflake, discover_urls_from_rss
# --- Scraper Logic Tests ---

def test_fetch_html_success():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html><body><h1>Test</h1></body></html>"
        result = fetch_html("https://fake.com")
        assert "Test" in result

def test_fetch_html_failure():
    with patch('requests.get') as mock_get:
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.RequestException()
        assert fetch_html("https://fake.com") is None

def test_parse_article_content_valid():
    html = "<html><body><h1>Headline</h1><p>Body</p></body></html>"
    result = parse_article_content(html)
    assert result['headline'] == "Headline"
    assert "Body" in result['body']

def test_parse_article_content_error_handling():
    # Pass invalid type to trigger internal try-except
    result = parse_article_content(None) 
    assert result['headline'] == "Error"
    
    
def test_discover_urls_from_rss_success():
    """Tests if URLs are correctly extracted from a mock RSS string."""
    mock_rss = """<rss version="2.0">
        <channel>
            <item><link>https://news.com/art1</link></item>
            <item><link>https://news.com/art2</link></item>
        </channel>
    </rss>"""
    
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_rss.encode('utf-8')
        
        result = discover_urls_from_rss("https://fake-rss.com")
        assert len(result) == 2
        assert "https://news.com/art1" in result

# --- Snowflake MERGE Tests ---

def test_save_articles_to_snowflake_empty():
    """Ensure we return False early if no articles are provided."""
    assert save_articles_to_snowflake([]) is False

@patch('scripts.scraper_news.load_sql_file')
@patch('scripts.scraper_news.get_snowflake_engine')
def test_save_articles_to_snowflake_success(mock_engine_func, mock_load_sql):
    """Tests the full flow of loading SQL and executing the MERGE."""
    # Mock the SQL file content
    mock_load_sql.return_value = "MERGE INTO TABLE..."
    
    # Mock the Engine and Connection
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine_func.return_value = mock_engine
    
    # This simulates the 'with engine.begin() as conn' context manager
    mock_engine.begin.return_value.__enter__.return_value = mock_conn
    
    test_data = [{"uuid": "123", "url": "url", "headline": "h", "body": "b"}]
    
    result = save_articles_to_snowflake(test_data)
    
    assert result is True
    # Verify that the SQL was actually 'executed' once
    assert mock_conn.execute.called