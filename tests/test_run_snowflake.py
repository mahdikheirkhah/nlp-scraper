import pytest
from unittest.mock import patch, MagicMock
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
