import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict

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
