# POTATO/components/utilities/combine_context.py
def combine_context(results: list) -> str:
    """
    Combine search results into a single context string.
    
    Args:
        results (list): List of search result dictionaries
        
    Returns:
        str: Combined context string
    """
    combined = ""
    for i, result in enumerate(results):
        combined += f"Result {i+1}:\n"
        combined += f"Title: {result.get('title', 'No title')}\n"
        combined += f"URL: {result.get('url', 'No URL')}\n"
        combined += f"Content: {result.get('content', 'No content')}\n"
        combined += "\n" + "="*50 + "\n\n"
    
    return combined