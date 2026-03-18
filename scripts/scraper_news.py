import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List
import time
from sqlalchemy import text
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



def discover_urls(archive_url: str, target_count: int = 300) -> List[str]:
    """
    Crawls an index/archive page to collect a list of unique article URLs.
    
    Arguments:
        archive_url (str): The starting URL (e.g., a 'Latest News' or 'Archive' page).
        target_count (int): The number of unique URLs to collect. Defaults to 300.

    Returns:
        List[str]: A list of unique strings representing article URLs.
    """
    found_urls = set()
    page = 1
    
    try:
        while len(found_urls) < target_count:
            # Handle pagination (example: ?page=1)
            current_url = f"{archive_url}?page={page}"
            print(f"Searching for links on: {current_url}")
            
            html = fetch_html(current_url)
            if not html:
                break
                
            soup = BeautifulSoup(html, 'html.parser')
            # Look for links that contain news patterns
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Filtering logic: ensures it's an article and not a category link
                if "/articles/" in href or "/story/" in href:
                    # Construct full URL if relative
                    full_path = href if href.startswith('http') else f"https://news-site.com{href}"
                    found_urls.add(full_path)
                
                if len(found_urls) >= target_count:
                    break
            
            page += 1
            if page > 20: # Safety break to prevent infinite loops
                break
    except Exception as e:
        print(f"Error during URL discovery: {e}")
        
    return list(found_urls)

def save_articles_to_snowflake(articles: List[Dict]):
    """
    Performs a bulk insert of scraped data into the Snowflake RAW table.
    
    Arguments:
        articles (List[Dict]): A list of dictionaries, where each dict contains 
                               'url', 'headline', and 'body'.

    Returns:
        bool: True if insertion was successful, False otherwise.
    """
    if not articles:
        print("No articles to save.")
        return False

    engine = get_snowflake_engine()
    if not engine:
        return False

    query = text("""
        INSERT INTO NEWS_DB.RAW.ARTICLES (URL, HEADLINE, BODY, SCRAPED_AT)
        VALUES (:url, :headline, :body, CURRENT_TIMESTAMP())
    """)

    try:
        with engine.begin() as conn:
            # executemany is efficient for 300+ records
            conn.execute(query, articles)
            print(f"Successfully ingested {len(articles)} articles into Snowflake.")
            return True
    except Exception as e:
        print(f"Failed to save to Snowflake: {e}")
        return False

def run_ingestion_pipeline():
    """
    Orchestrates the discovery, scraping, and storage of 300 articles.
    """
    print("Starting Phase 1 Ingestion...")
    
    # 1. Discover URLs
    base_archive = "https://news-site.com/archive" # Replace with real target
    urls = discover_urls(base_archive, target_count=300)
    
    scraped_data = []
    
    # 2. Scrape individual content
    for i, url in enumerate(urls):
        print(f"Scraping [{i+1}/300]: {url}")
        html = fetch_html(url)
        if html:
            content = parse_article_content(html)
            content['url'] = url # Add URL to the dict for the DB insert
            scraped_data.append(content)
        
        # Be a polite scraper!
        time.sleep(1) 

    # 3. Save to Snowflake
    save_articles_to_snowflake(scraped_data)

if __name__ == "__main__":
    run_ingestion_pipeline()