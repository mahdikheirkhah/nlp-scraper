import sys
import os
import pandas as pd
from sqlalchemy import text
from run_snowflake import get_snowflake_engine
from nlp_enriched_news import NewsNLPPipeline

def enrich_scraped_data():
    engine = get_snowflake_engine()
    pipeline = NewsNLPPipeline()
    
    if not engine:
        print("Failed to connect to Snowflake.")
        return

    # 1. Fetch data from Snowflake
    query = "SELECT ID, HEADLINE, BODY, URL FROM NEWS_DB.RAW.ARTICLES LIMIT 10"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    print(f"Enriching {len(df)} articles...")

    results = []
    for index, row in df.iterrows():
        # 2. Run through NLP Pipeline
        analysis = pipeline.process_article(row['BODY'], row['HEADLINE'])
        
        # Combine original data with analysis
        results.append({
            "Headline": row['HEADLINE'][:50] + "...",
            "Topic": analysis['topic'],
            "Companies": ", ".join(analysis['orgs']),
            "Sentiment": analysis['sentiment'],
            "Scandal Score": analysis['scandal_distance'],
            "Flagged?": "⚠️ YES" if analysis['is_flagged'] else "✅ NO"
        })

    # 3. Display Results
    enriched_df = pd.DataFrame(results)
    print(enriched_df.to_string())

if __name__ == "__main__":
    enrich_scraped_data()
