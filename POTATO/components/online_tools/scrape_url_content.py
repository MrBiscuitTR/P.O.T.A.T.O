from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import re

def scrape_and_clean_url_content(url: str) -> dict:
    """
    Scrape a URL and clean the content to keep only relevant elements.
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        dict: A dictionary containing the cleaned content and title
    """
    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else 'No title found'
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Keep only specific tags and clean them
        # Remove all tags except h1, h2, h3, p, li, blockquote, table, a, th, td, tr, div
        for tag in soup.find_all():
            if tag.name not in ['h1', 'h2', 'h3', 'p', 'li', 'blockquote', 'table', 'a', 'th', 'td', 'tr', 'div']:
                tag.decompose()
        
        # Clean <a> tags - keep only href attribute
        for a_tag in soup.find_all('a'):
            if a_tag.get('href'):
                a_tag.attrs = {'href': a_tag['href']}
            else:
                a_tag.attrs = {}
        
        # Clean <p> tags - remove all attributes
        for p_tag in soup.find_all('p'):
            p_tag.attrs = {}
        
        # Clean <div> tags - remove all attributes
        for div_tag in soup.find_all('div'):
            div_tag.attrs = {}
        
        # Clean table-related tags
        for table_tag in soup.find_all('table'):
            table_tag.attrs = {}
        
        for th_tag in soup.find_all('th'):
            th_tag.attrs = {}
        
        for td_tag in soup.find_all('td'):
            td_tag.attrs = {}
        
        for tr_tag in soup.find_all('tr'):
            tr_tag.attrs = {}
        
        # Extract meaningful text content
        text_content = ""
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'blockquote', 'table', 'div']):
            if element.name in ['h1', 'h2', 'h3', 'p', 'li', 'blockquote']:
                text_content += element.get_text(strip=True) + "\n"
            elif element.name == 'table':
                # For tables, extract the text content
                table_text = ""
                for row in element.find_all('tr'):
                    row_text = ""
                    for cell in row.find_all(['td', 'th']):
                        row_text += cell.get_text(strip=True) + " "
                    table_text += row_text.strip() + "\n"
                text_content += table_text + "\n"
            elif element.name == 'div':
                # For divs, extract text but skip empty ones
                div_text = element.get_text(strip=True)
                if div_text:
                    text_content += div_text + "\n"
        
        # Clean up the text content
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        return {
            'title': title,
            'content': text_content
        }
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {
            'title': 'Error',
            'content': f'Error scraping content: {str(e)}'
        }