import pytest
import requests
import os
from unittest.mock import patch, MagicMock
from scripts.scraper_news import fetch_html, parse_article_content
# Ensure this matches your file name exactly
from scripts.run_snowflake import get_snowflake_engine 

# --- Snowflake Engine Tests ---

def test_get_snowflake_engine_success():
    """Tests successful creation of the Snowflake engine."""
    try:
        # NOTE: We patch the libraries AS THEY ARE USED in scripts.run_snowflake
        with patch('os.getenv') as mock_env, \
             patch('scripts.run_snowflake.serialization.load_pem_private_key') as mock_load_key, \
             patch('scripts.run_snowflake.create_engine') as mock_create_engine, \
             patch('scripts.run_snowflake.URL.create') as mock_url:
            
            # Mock Environment
            mock_env.side_effect = lambda k, default=None: {
                "SNOWFLAKE_PRIVATE_KEY": "fake_key",
                "SNOWFLAKE_USER": "test_user",
                "SNOWFLAKE_ACCOUNT": "test_account"
            }.get(k, default)
            
            # Mock Cryptography
            mock_key_obj = MagicMock()
            mock_load_key.return_value = mock_key_obj
            mock_key_obj.private_bytes.return_value = b"fake_der_bytes"
            
            # Mock SQLAlchemy
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_url.return_value = "snowflake://fake_url"

            result = get_snowflake_engine()
            
            assert result == mock_engine
    except Exception as e:
        pytest.fail(f"get_snowflake_engine_success failed: {e}")

def test_get_snowflake_engine_invalid_key():
    """Tests failure when the RSA key is malformed."""
    with patch('os.getenv', return_value="invalid_key"), \
         patch('scripts.run_snowflake.serialization.load_pem_private_key') as mock_load_key:
        
        mock_load_key.side_effect = ValueError("Invalid Key")
        
        result = get_snowflake_engine()
        assert result is None

def test_get_snowflake_engine_missing_env():
    """Tests failure when env variables are missing."""
    with patch('os.getenv', return_value=None):
        result = get_snowflake_engine()
        assert result is None

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