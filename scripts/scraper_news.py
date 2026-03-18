import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List
import time
from sqlalchemy import text
import xml.etree.ElementTree as ET
from scripts.run_snowflake import get_snowflake_engine

def fetch_html(url: str) -> Optional[str]:
    """
    Performs an HTTP GET request to retrieve the HTML content of a URL.
    
    Args:
        url (str): The target URL to scrape.
        
    Returns:
        Optional[str]: The raw HTML text if successful, None otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        # Explaining why: Simple logging to console to track network failures 
        # without stopping the entire scraper loop.
        print(f"Failed to fetch {url}: {e}")
        return None

def parse_article_content(html: str) -> Dict[str, str]:
    """
    Parses raw HTML to extract the headline and body text of an article.
    
    Args:
        html (str): Raw HTML content from a news page.
        
    Returns:
        Dict[str, str]: A dictionary containing 'headline' and 'body'.
    }
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        headline = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No Headline"
        paragraphs = soup.find_all('p')
        body = " ".join([p.get_text(strip=True) for p in paragraphs])
        return {"headline": headline, "body": body}
    except Exception as e:
        # Explaining why: Malformed HTML shouldn't stop the scraper from moving 
        # to the next article in the list.
        print(f"Error parsing HTML content: {e}")
        return {"headline": "Error", "body": ""}


def load_sql_file(file_path: str):
    """Reads a .sql file and returns the query as a string."""
    with open(file_path, 'r') as f:
        return f.read()

def save_articles_to_snowflake(articles: List[Dict]):
    if not articles:
        return False

    engine = get_snowflake_engine()
    if not engine:
        return False

    # Load the query from the external file
    query_text = load_sql_file('sql/02_merge_articles.sql')
    merge_query = text(query_text)

    try:
        with engine.begin() as conn:
            # Snowflake handles the looping over 'articles' automatically
            conn.execute(merge_query, articles)
            print(f"Successfully processed {len(articles)} articles through MERGE.")
            return True
    except Exception as e:
        print(f"Error during Snowflake merge: {e}")
        return False

def discover_urls_from_rss(rss_url: str, target_count: int = 300) -> List[str]:
    """
    Parses an RSS feed to extract article URLs.
    
    Arguments:
        rss_url (str): The BBC RSS feed URL.
        target_count (int): Maximum number of URLs to collect.
        
    Returns:
        List[str]: A list of unique article URLs found in the feed.
    """
    article_urls = []
    try:
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        
        # Parse the XML content
        root = ET.fromstring(response.content)
        
        # In RSS, each article is inside an <item> tag
        for item in root.findall('.//item'):
            link = item.find('link').text
            if link and link not in article_urls:
                article_urls.append(link)
            
            if len(article_urls) >= target_count:
                break
                
        print(f"Found {len(article_urls)} articles in the RSS feed.")
    except Exception as e:
        print(f"Error parsing RSS feed: {e}")
        
    return article_urls

def run_ingestion_pipeline():
    """
    Orchestrates ingestion from multiple feeds to reach the 300-article goal.
    """
    feeds = [
            "http://feeds.bbci.co.uk/news/business/rss.xml",
            "http://feeds.bbci.co.uk/news/technology/rss.xml",
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "http://feeds.bbci.co.uk/news/politics/rss.xml", 
            "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
            "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "http://feeds.bbci.co.uk/news/education/rss.xml",
            "https://moxie.foxnews.com/google-publisher/latest.xml",
            "https://moxie.foxnews.com/google-publisher/politics.xml",
            "https://moxie.foxnews.com/google-publisher/health.xml",
            "https://moxie.foxnews.com/google-publisher/science.xml",
            "https://moxie.foxnews.com/google-publisher/tech.xml"
    ]
    
    all_urls = []
    for feed in feeds:
        urls = discover_urls_from_rss(feed, target_count=150) # Take 150 from each
        all_urls.extend(urls)
    
    # Remove duplicates between different feeds
    unique_urls = list(set(all_urls))[:350] 
    print(f"Total unique URLs to scrape: {len(unique_urls)}")

    scraped_data = []
    for i, url in enumerate(unique_urls):
        print(f"Scraping [{i+1}/{len(unique_urls)}]: {url}")
        html = fetch_html(url)
        if html:
            content = parse_article_content(html)
            content['url'] = url
            # ADD THIS LINE: Generate a unique ID for every article here
            content['uuid'] = str(uuid.uuid4()) 
            scraped_data.append(content)
        time.sleep(1.0)

    save_articles_to_snowflake(scraped_data)

if __name__ == "__main__":
    run_ingestion_pipeline()