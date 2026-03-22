import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text
from tabulate import tabulate
from run_snowflake import get_snowflake_engine
from nlp_enriched_news import NewsNLPPipeline

# Constants
SQL_FILE_NAME = "03_read_articles.sql"
OUTPUT_CSV = "results/enhanced_news.csv"

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

def generate_plots(df):
    """Task 5.4: Create visual analysis of the news feed."""
    plt.style.use('ggplot')
    
    # 1. Topic Distribution
    plt.figure(figsize=(10, 5))
    df['Topic'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Articles by Topic Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/topic_distribution.png")
    
    # 2. Risk Score Histogram
    plt.figure(figsize=(10, 5))
    df['Score'].astype(float).plot(kind='hist', bins=20, color='salmon', edgecolor='black')
    plt.title("Distribution of Scandal Risk Scores")
    plt.xlabel("Score (0.0 - 1.0)")
    plt.savefig("results/risk_distribution.png")
    
    print("\n📊 Visualizations saved: 'topic_distribution.png' and 'risk_distribution.png'")

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
            # Using chunksize for memory efficiency
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
                        
                        # 2. Extract specific trigger for reporting
                        raw_trigger = analysis.get('trigger_sentence', 'N/A')
                        display_trigger = (raw_trigger[:77] + "...") if len(raw_trigger) > 80 else raw_trigger

                        # 3. Store results with all required Phase 5 columns
                        all_results.append({
                            "Headline": headline,
                            "Topic": analysis.get('topic', 'Unknown'),
                            "Entities": ", ".join(analysis.get('orgs', [])),
                            "Sentiment": analysis.get('sentiment', 0),
                            "Score": analysis.get('scandal_distance', 0),
                            "Is_Scandal": analysis.get('is_flagged'),
                            "Status": "🚩 SCANDAL" if analysis.get('is_flagged') else "✅ Clean",
                            "Evidence": display_trigger,
                            "URL": url,
                            "Processed_Text": analysis.get('processed_text', '')
                        })

                    except Exception as e:
                        print(f"  ⚠️ Error in Batch {current_batch_num}, row {index}: {e}")
                        continue

        if not all_results:
            print("No data was retrieved from Snowflake.")
            return

        # 4. FINAL DATAFRAME & EXPORT (Task 5.2)
        full_df = pd.DataFrame(all_results)
        full_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Task 5.2: Exported {len(full_df)} articles to {OUTPUT_CSV}")

        # 5. RISK INTELLIGENCE DASHBOARD (Task 5.1 & 5.3)
        print("\n" + "═"*110)
        print(" " * 40 + "🕵️ FINAL RISK INTELLIGENCE REPORT")
        print("═"*110)

        # Show Top 10 by Risk Score (Task 5.3)
        top_10 = full_df.sort_values(by="Score", ascending=False).head(10)
        print(f"\n🏆 TOP 10 HIGHEST RISK EXPOSURES IDENTIFIED:")
        print(tabulate(
            top_10[['Headline', 'Topic', 'Score', 'Status']], 
            headers=['Headline', 'Topic', 'Risk Score', 'Result'], 
            tablefmt='fancy_grid', 
            showindex=False
        ))

        # Show detailed evidence for flagged items
        scandals = full_df[full_df['Is_Scandal'] == True]
        if not scandals.empty:
            print(f"\n🚨 CRITICAL EVIDENCE FOR {len(scandals)} FLAGGED SCANDALS:")
            for i, row in scandals.iterrows():
                print(f"[{i+1}] {row['Headline'][:80]}")
                print(f"    Evidence: \"{row['Evidence']}\"")
                print(f"    Link: {row['URL']}\n")
        else:
            print("\n✅ SYSTEM STATUS: No articles crossed both Similarity and Sentiment thresholds.")

        # 6. VISUALIZATION (Task 5.4)
        generate_plots(full_df)

    except Exception as e:
        print(f"❌ Critical error in the enrichment process: {e}")

if __name__ == "__main__":
    enrich_scraped_data()