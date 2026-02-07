from fastmcp import FastMCP  # ← no need to import 'tools' anymore
import requests
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup, NavigableString, Tag
from POTATO.components.visual_tools.extract_text import fast_extract_pdf_text
SEARX_URL = "http://localhost:8080/search"  # Change to your SearXNG instance URL, probably the same. you can also use a public instance but be aware of privacy implications.
MAX_RESULTS = 5

# Create the server instance (this is what you use for decorators)
mcp = FastMCP("Searx Web Search")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)          # strip HTML
    text = re.sub(r'\s+', ' ', text)             # normalize whitespace
    return text.strip()

# Define raw functions first (for direct imports)
def _potatool_web_search_urls_impl(
    query: str,
    num_results: int = MAX_RESULTS,
    page: int = 1
) -> dict:
    """
    Search the web using a private SearxNG instance.
    Returns clean titles, URLs, domains & snippets.
    
    Preferred tool for general/current information needs.
    Can search multiple pages for more results.
    
    Args:
        query: Search query string
        num_results: Number of results per page (1-10)
        page: Page number (starts at 1)
    """
    try:
        num_results = min(max(1, num_results), 10)
        page = max(1, page)  # Ensure page is at least 1
        
        params = {
            'q': query,
            'format': 'json',
            'language': 'all',
            'category': 'general',
            'safesearch':'0',
            'pageno': page
        }
        
        resp = requests.get(SEARX_URL, params=params, timeout=12)
        resp.raise_for_status()
        
        data = resp.json()
        raw_results = data.get("results", [])[:num_results]
        
        formatted = []
        for item in raw_results:
            formatted.append({
                "title": clean_text(item.get("title", "")),
                "url": item.get("url", ""),
                "domain": urlparse(item.get("url", "")).netloc,
                "snippet": clean_text(item.get("content", ""))
            })
        
        return {
            "query": query,
            "page": page,
            "results": formatted,
            "count": len(formatted),
            "error": None
        }
    
    except Exception as e:
        return {
            "query": query,
            "page": page,
            "results": [],
            "count": 0,
            "error": str(e)
        }

def _potatool_extract_content_impl(
    url: str
) -> dict:
    """
    Extract human-readable content from a web page URL.
    Use with the relevant URLs retrieved after calling "potatool_web_search_urls" tool.
    Preserves <a href="...">text</a> links exactly as-is.
    Preserves tables in simplified HTML (<table><tr><th><td> only, no attributes).
    Flattens all other elements (including headings, paragraphs, lists, code blocks, divs, spans, etc.) to plain text.
    
    Useful for AI agents to understand full page content without full HTML clutter.
    """
    try:
        # Use realistic browser headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        # Try with longer timeout and follow redirects
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        resp.raise_for_status()
        
        # Check if we got content
        if not resp.text or len(resp.text) < 100:
            return {
                "url": url,
                "content": "",
                "error": f"Page returned very little content ({len(resp.text)} chars). Site may require JavaScript or block scrapers."
            }
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove unwanted sections/tags
        for tag in soup.find_all(['script', 'style', 'head', 'iframe', 'noscript', 'footer', 'nav', 'aside', 'header', 'form', 'button']):
            tag.decompose()
        
        # Recursive function to process elements
        def process_element(element):
            if isinstance(element, NavigableString):
                return clean_text(element)
            
            if not isinstance(element, Tag):
                return ''
            
            if element.name == 'a' and element.get('href'):
                href = element.get('href', '')
                inner = ''.join(process_element(child) for child in element.children)
                return f'<a href="{href}">{inner}</a>'
            
            if element.name == 'table':
                simplified = '<table>'
                for row in element.find_all('tr', recursive=False):
                    simplified += '<tr>'
                    for cell in row.find_all(['th', 'td'], recursive=False):
                        tag_name = 'th' if cell.name == 'th' else 'td'
                        cell_inner = ''.join(process_element(child) for child in cell.children)
                        simplified += f'<{tag_name}>{cell_inner}</{tag_name}>'
                    simplified += '</tr>'
                simplified += '</table>'
                return simplified
            
            # For all other tags: recurse children and join appropriately
            children_content = ''.join(process_element(child) for child in element.children)
            
            # Add newlines for block-level elements to improve readability
            if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'section', 'article', 'li', 'pre', 'code', 'blockquote']:
                return children_content + '\n\n'
            elif element.name == 'br':
                return '\n'
            else:
                # Inline elements like span, strong, em, etc. – just concatenate
                return children_content
            
            return ''
        
        # Start processing from body (or whole soup if no body)
        main_content = soup.body if soup.body else soup
        extracted = process_element(main_content).strip()
        
        # Check if we actually extracted meaningful content
        if not extracted or len(extracted) < 50:
            return {
                "url": url,
                "content": extracted,
                "error": f"Extracted very little content ({len(extracted)} chars). Site may be blocked, require login, or need JavaScript rendering. Try a different URL."
            }
        return {
            "url": url,
            "content": extracted,
            "error": None
        }
    
    except Exception as e:
        return {
            "url": url,
            "content": "",
            "error": str(e)
        }

# Register with MCP decorators (wraps the _impl functions)
@mcp.tool
def potatool_web_search_urls(query: str, num_results: int = MAX_RESULTS) -> dict:
    """
    Search the web using a private SearxNG instance.
    Returns clean titles, URLs, domains & snippets.
    """
    return _potatool_web_search_urls_impl(query, num_results)

@mcp.tool
def potatool_extract_content(url: str) -> dict:
    """
    Extract human-readable content from a web page URL.
    """
    return _potatool_extract_content_impl(url)

if __name__ == "__main__":
    # Default = stdio transport → perfect for Continue.dev MCP
    mcp.run()