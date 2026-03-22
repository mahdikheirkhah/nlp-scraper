import os
import pandas as pd
from sqlalchemy import text
from tabulate import tabulate
from run_snowflake import get_snowflake_engine
from nlp_enriched_news import NewsNLPPipeline

SQL_FILE_NAME = "03_read_articles.sql"

def read_sql_file(file_name: str) -> str:
    """Reads a SQL query from a file in the 'sql' directory."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, 'sql', file_name)
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: SQL file {file_name} not found at {file_path}")
        return ""

def enrich_scraped_data():
    try:
        engine = get_snowflake_engine()
        pipeline = NewsNLPPipeline()
        
        if not engine:
            print("❌ Failed to connect to Snowflake engine.")
            return

        query_raw = read_sql_file(SQL_FILE_NAME)
        if not query_raw:
            return

        batch_size = 50
        all_results = []
        
        print(f"📡 Connecting to Snowflake... (Batch Size: {batch_size})")
        
        with engine.connect() as conn:
            for chunk_idx, df_chunk in enumerate(pd.read_sql(text(query_raw), conn, chunksize=batch_size)):
                df_chunk.columns = [c.lower() for c in df_chunk.columns]
                current_batch_num = chunk_idx + 1
                print(f"📦 Batch {current_batch_num}: Analyzing {len(df_chunk)} articles...")

                for index, row in df_chunk.iterrows():
                    try:
                        body = row.get('body') or ""
                        headline = row.get('headline') or ""
                        url = row.get('url') or "No URL"
                        
                        if not body and not headline:
                            continue

                        # 1. Run the Intelligence Pipeline
                        analysis = pipeline.process_article(body, headline)
                        
                        # 2. Extract specific trigger and truncate for display
                        raw_trigger = analysis.get('trigger_sentence', 'N/A')
                        display_trigger = (raw_trigger[:77] + "...") if len(raw_trigger) > 80 else raw_trigger

                        # 3. Store results
                        all_results.append({
                            "Headline": (headline[:40] + "..") if headline else "N/A",
                            "Topic": analysis.get('topic', 'Unknown'),
                            "Entities": ", ".join(analysis.get('orgs', [])[:2]),
                            "Sentim.": f"{analysis.get('sentiment', 0):.2f}",
                            "Score": f"{analysis.get('scandal_distance', 0):.2f}",
                            "Status": "🚩 SCANDAL" if analysis.get('is_flagged') else "✅ Clean",
                            "Trigger Sentence": display_trigger,
                            "URL": url
                        })

                    except Exception as e:
                        print(f"  ⚠️ Error in Batch {current_batch_num}, row {index}: {e}")
                        continue

        # 4. FINAL REPORTING DASHBOARD
        if all_results:
            full_df = pd.DataFrame(all_results)
            scandals = full_df[full_df['Status'] == "🚩 SCANDAL"]

            print("\n" + "═"*100)
            print(" " * 35 + "🕵️ RISK INTELLIGENCE REPORT")
            print("═"*100)

            if not scandals.empty:
                print(f"\n🚨 {len(scandals)} POTENTIAL SCANDALS IDENTIFIED")
                # We show the Trigger Sentence here so you can see 'Why' it flagged
                print(tabulate(
                    scandals[['Headline', 'Topic', 'Entities', 'Score', 'Trigger Sentence']], 
                    headers=['Headline', 'Topic', 'Companies', 'Risk Score', 'Evidence (Trigger Sentence)'], 
                    tablefmt='fancy_grid', 
                    showindex=False
                ))
                
                print("\n🔗 ACTION REQUIRED: Review these URLs immediately:")
                for i, row in scandals.iterrows():
                    print(f"  [{i+1}] {row['Headline']} -> {row['URL']}")
            else:
                print("\n✅ SYSTEM STATUS: No significant risks detected.")

            print("\n📝 RECENT CLEAN ARTICLES (PREVIEW):")
            clean_sample = full_df[full_df['Status'] == "✅ Clean"].tail(5)
            print(tabulate(
                clean_sample[['Headline', 'Topic', 'Entities', 'Sentim.']], 
                headers=['Headline', 'Topic', 'Companies', 'Sent.'], 
                tablefmt='simple', 
                showindex=False
            ))
            
            print(f"\n✨ Processed {len(all_results)} articles successfully.")
        else:
            print("No data was retrieved from Snowflake.")

    except Exception as e:
        print(f"❌ Critical error in the enrichment process: {e}")

if __name__ == "__main__":
    enrich_scraped_data()