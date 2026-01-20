from ollama import ChatResponse, chat
import requests
import json
from POTATO.components.utilities.summarize import summarize_content

def check_preview_relevance(topic: str, title: str, url: str, content: str) -> bool:
    """
    Check if a search result is relevant to the original topic based on title, URL, and content.
    
    Args:
        topic (str): The original search topic
        title (str): The title of the search result
        url (str): The URL of the search result
        content (str): The content preview of the search result
    
    Returns:
        bool: True if the result is relevant, False otherwise
    """
    # Combine all text for analysis
    combined_text = f"{title} {url} {content}"
    
    # Use Ollama to determine relevance
    response = chat(
        model='llama3.1:8b',
        messages=[
            {
                'role': 'user',
                'content': f"""Analyze if the following search result is relevant to the topic "{topic}".
                Title: {title}
                URL: {url}
                Content: {content}
                
                Respond with only 'true' if relevant, or 'false' if not relevant."""
            }
        ]
    )
    
    result = response['message']['content'].strip().lower()
    return result == 'true'

def check_summary_relevance(topic: str, summary: str) -> str:
    """
    Check if a summary is relevant to the original topic. If not relevant, resummarize it.
    
    Args:
        topic (str): The original search topic
        summary (str): The summary to check
        
    Returns:
        str: The final summary (either the original if relevant, or a resummarized version)
    """
    # Use Ollama to determine if the summary is relevant to the topic
    response = chat(
        model='llama3.1:8b',
        messages=[
            {
                'role': 'user',
                'content': f"""Analyze if the following summary is relevant to the topic "{topic}".
                Summary: {summary}
                
                Respond with only 'true' if relevant, or 'false' if not relevant."""
            }
        ]
    )
    
    result = response['message']['content'].strip().lower()
    
    if result == 'true':
        return summary
    else:
        # If not relevant, resummarize the content
        return summarize_content(summary, "Resummarized Content", topic)


if __name__ == "__main__":
    topic = "Python programming... which level is it? low high or what really"
    content = """
    Python is a high-level, interpreted programming language that focuses on code readability and simplicity. It was created by Guido van Rossum and first released in 1991. Since then, it has become one of the most popular languages for web development, data analysis, artificial intelligence, machine learning, and more.

    Python's syntax is clean and easy to learn, making it a great choice for beginners. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. This versatility makes Python suitable for a wide range of applications.

    One of the key features of Python is its extensive standard library, which includes modules for handling various tasks such as file I/O, network communication, threading, and more. Additionally, Python has a large ecosystem of third-party libraries that can be easily installed using pip, making it even more powerful.

    Python's simplicity also extends to error handling. Unlike some other programming languages, Python provides clear and helpful error messages when something goes wrong. This makes debugging easier for beginners and experienced developers alike.

    In terms of performance, Python is an interpreted language, which means it may not be as fast as compiled languages like C or Java. However, for many applications, the tradeoff between speed and ease of development is worth it. For tasks that require more performance, Python can be combined with Cython or PyPy to achieve faster execution times.

    Python's popularity is evident in its large and active community. There are countless resources available online for learning Python, including tutorials, documentation, and forums where developers can ask questions and get help. Additionally, Python conferences and meetups are held worldwide, providing opportunities for networking and collaboration.

    In summary, Python is a versatile, easy-to-learn programming language that is widely used in various domains. Its clean syntax, extensive standard library, and large community make it an excellent choice for beginners and experienced developers alike.
    """
    title = "Introduction to Python Programming"
    summary = summarize_content(content, title, topic)
    print(summary)
