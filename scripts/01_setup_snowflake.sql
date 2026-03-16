-- 1. Create the environment
CREATE DATABASE IF NOT EXISTS NEWS_DB;
CREATE SCHEMA IF NOT EXISTS NEWS_DB.RAW;

-- 2. Create the Warehouse with aggressive auto-suspend to save credits
CREATE WAREHOUSE IF NOT EXISTS NEWS_WH 
    WITH WAREHOUSE_SIZE = 'XSMALL' 
    AUTO_SUSPEND = 60    -- Shuts down after 60 seconds of inactivity
    AUTO_RESUME = TRUE   -- Automatically turns back on when a query hits it
    COMMENT = 'Warehouse for news scraping and NLP processing';

-- 3. Create the table using professional data types
-- Using TRANSIENT to save on storage costs for raw data
CREATE OR REPLACE TRANSIENT TABLE NEWS_DB.RAW.ARTICLES (
    UUID STRING NOT NULL COMMENT 'Unique identifier for the article',
    URL STRING NOT NULL,
    DATE_SCRAPED TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    HEADLINE STRING,
    BODY STRING,
    CONSTRAINT pk_uuid PRIMARY KEY (UUID) -- Informational only in Snowflake
) COMMENT = 'Landing table for raw news scraper data';
