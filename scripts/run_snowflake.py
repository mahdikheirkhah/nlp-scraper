import os
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.engine.url import URL
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

load_dotenv()

def get_snowflake_engine() -> Engine:
    """
    Creates a Snowflake SQLAlchemy engine using RSA Private Key authentication.
    
    Returns:
        Engine: SQLAlchemy engine configured for Snowflake.
    """
    # Prepare the private key bytes from .env string
    p_key_raw = os.getenv("SNOWFLAKE_PRIVATE_KEY").replace("\\n", "\n").encode()
    
    p_key_obj = serialization.load_pem_private_key(
        p_key_raw,
        password=None,
        backend=default_backend()
    )

    p_key_der = p_key_obj.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Build the connection URL
    # Snowflake uses the 'database' and 'schema' as query parameters in SQLAlchemy
    connection_url = URL.create(
        drivername="snowflake",
        username=os.getenv("SNOWFLAKE_USER"),
        host=os.getenv("SNOWFLAKE_ACCOUNT"),
        query={
            "database": "NEWS_DB",
            "schema": "RAW",
            "warehouse": "NEWS_WH" # Added this to ensure compute is available
        }
    )

    return create_engine(
        connection_url,
        connect_args={'private_key': p_key_der}
    )
# Usage
engine = get_snowflake_engine()
with engine.connect() as conn:
    # Use text() to wrap the raw SQL string
    result = conn.execute(text("SELECT current_version()")).fetchone()
    print(result)