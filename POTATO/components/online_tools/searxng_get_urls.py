from ollama import ChatResponse, chat
import requests
import json

def searx_get_urls_topic_page(topic: str, pageno: int = 1) -> list:
    """
    Use Ollama to generate search queries and fetch URLs from SearXNG.
    
    Args:
        topic (str): The search topic (should be brief for meaningful results)
        pageno (int): Page number for pagination (1-5)
    
    Returns:
        list: List of dictionaries containing URL, title, content preview, and related URLs
    """
    # Generate search query using Ollama
    response = chat(
        model='llama3.1:8b',
        messages=[
            {
                'role': 'user',
                'content': f"""Generate a brief search topic (max 5 words) to get information on this topic: "{topic}". Only return the topic. 
                Do not add any additional words or phrases. If the topic is a name, return just the name without any additions like "biography", "life", etc.
                For example, if the input is "cagan efe calidag biography", return only "cagan efe calidag". If the input is "machine learning", return "machine learning".
                Always generate one single topic. Do not hallucinate. Do not add unnecessary words."""
            }
        ]
    )
    
    search_topic = response['message']['content'].strip()
    print(f"Search topic (searxng_get_urls.py): {search_topic}")

    
    # Construct the SearXNG URL
    url = f"https://searx.cagancalidag.com/search?q={search_topic}&format=json&pageno={pageno}&language=all&time_range=&safesearch=0&categories=general"
    
    try:
        # Make GET request to SearXNG
        response = requests.get(url)
        # print(f"Response : {response.content}")
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Extract results
        results = []
        for result in data.get('results', []):
            results.append({
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'publishedDate': result.get('publishedDate', None),
                'engine': result.get('engine', ''),
                'template': result.get('template', ''),
                # 'img_src': result.get('img_src', ''),
                # 'thumbnail': result.get('thumbnail', ''),
                # 'priority': result.get('priority', ''),
                'engines': result.get('engines', []),
                'positions': result.get('positions', []),
                # 'score': result.get('score', 0),
                'parsed_url': result.get('parsed_url', []),
            })
        
        return results
    
    except requests.RequestException as e:
        print(f"Error fetching from SearXNG: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Example with a topic
    results = searx_get_urls_topic_page("cagan efe calidag", 1)
    for result in results:
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content']}")
        print("---")