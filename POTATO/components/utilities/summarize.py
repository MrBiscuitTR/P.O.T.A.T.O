from ollama import chat

def summarize_content(content: str, title: str, topic: str) -> str:
    """
    Summarize the scraped content using Ollama.
    
    Args:
        content (str): The cleaned content to summarize
        title (str): The title of the webpage
        topic (str): The original search topic
        
    Returns:
        str: The summarized content
    """
    # System prompt to guide the summarization process
    system_prompt = f"""
You are an expert content summarizer. Your task is to create a comprehensive summary of the provided webpage content.
    
Key instructions:
1. Analyze the content thoroughly and understand the main topic
2. Extract and preserve all key information, including technical details, mathematical formulas, scientific concepts, and important facts
3. If tables are present, understand their content and incorporate the key data into your summary rather than keeping the table format
4. Make sure to explain complex concepts clearly and concisely
5. The summary should be informative and cover all essential points
6. Keep the summary below 500 words when possible, but extend if key information requires more detail
7. Focus on the relevance to the original topic: "{topic}"
8. If content is fragmented or not fully readable, piece together information logically
9. Maintain accuracy and avoid adding information not present in the source

Format your response as a clear, well-structured summary without any additional formatting or markdown.
"""
    
    # User prompt with the content
    user_prompt = f"""
Summarize the following webpage content. The title is "{title}" and the content is:

{content}

Make sure to:
- Include all key technical details, scientific concepts, and important facts
- Explain complex concepts clearly
- Incorporate table data into the summary text rather than keeping table format
- Maintain relevance to the topic "{topic}"
- Keep the summary informative but concise (preferably under 500 words)
"""
    
    # Call Ollama for summarization
    response = chat(
        model='llama3.1:8b',
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
    )
    
    return response['message']['content'].strip()