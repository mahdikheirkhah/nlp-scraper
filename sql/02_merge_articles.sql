-- sql/02_merge_articles.sql
-- Upsert logic to handle deduplication at the database level
MERGE INTO NEWS_DB.RAW.ARTICLES AS target
USING (
    SELECT 
        :uuid AS uuid, 
        :url AS url, 
        :headline AS headline, 
        :body AS body
) AS source
ON target.URL = source.url
WHEN NOT MATCHED THEN
    INSERT (UUID, URL, HEADLINE, BODY, DATE_SCRAPED)
    VALUES (source.uuid, source.url, source.headline, source.body, CURRENT_TIMESTAMP());