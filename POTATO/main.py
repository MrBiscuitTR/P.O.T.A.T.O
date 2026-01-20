# POTATO/main.py
# The core of P.O.T.A.T.O -- Practical Omnipurpose Technical AI Tool Operator
# Uses gpt-oss:20b as main model.
## Uses ollama module to interface with local LLMs.
## Uses langchain to manage chains, agents, and memory.(?)
## Uses a modular approach to add tools , utilities and functionalities. 
###(- ./components/vocal_tools/ , - ./components/visual_tools/ , - ./components/online_tools/ , - ./components/local_tools/ , - ./components/dangerous_tools/ , - ./components/n8n/ , - ./components/utilities/ )
### Uses a webui (./webui/) to provide an interface for users to interact with P.O.T.A.T.O. shows status, logs, system usage, prompts, desktop/browser, other useful parameters.

# POTATO/main.py
"""
Main entry point for the POTATO assistant system.
Handles the boot phase, event loop, and safety gates as described in project-notes.txt.

Architecture Overview:
1. Boot Phase: Load config, detect GPU/VRAM, load models, start background workers
2. Event Loop: Always listening (STT stream), push utterances to queue, agent decides → tool calls
3. Safety Gate: Any dangerous_tools call must pass intent check and explicit human confirmation, not vocal.

Key Design Principles:
- STT runs in its own thread/process, always on.
- TTS runs in its own thread/process. do not use piper. use something more realistic and with more emotion even if its more resource heavy.
- LLM reasoning never blocks audio I/O
- Streaming responses for natural conversation flow
- certain block words like ok stop, enough, thank you... will stop generation and tts.
"""

import os
import json
import logging
from ollama import (
    ChatResponse,
    EmbeddingsResponse,
    EmbedResponse,
    GenerateResponse,
    Image,
    ListResponse,
    Message,
    Options,
    ProcessResponse,
    ProgressResponse,
    RequestError,
    ResponseError,
    ShowResponse,
    StatusResponse,
    Tool,
    WebFetchResponse,
    WebSearchResponse,
    chat
)
from POTATO.components.utilities.get_system_info import json_get_instant_system_info
from POTATO.components.online_tools.searxng_get_urls import searx_get_urls_topic_page
from POTATO.components.visual_tools.image_gen import generate_image
from POTATO.components.utilities.summarize import summarize_content
from POTATO.components.utilities.combine_context import combine_context
from POTATO.components.utilities.check_relevance import check_preview_relevance, check_summary_relevance
from POTATO.components.online_tools.scrape_url_content import scrape_and_clean_url_content

initial_system_status = json_get_instant_system_info()

# Load config.json
def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("config.json not found, using default settings")
        return {}

# disable all telemetry/logging from third party libraries
os.environ["LANGCHAIN_HANDLER"] = "langchain_core.handlers.noop_handler.NoOpHandler"
os.environ["OPENAI_LOG"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFF"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["OLLAMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_METADATA"]   = "1" # block extra hub metadata requests
os.environ["HF_HUB_DISABLE_TOKENS"]     = "1" # prevent token usage telemetry

def generate_topics_from_prompt(prompt: str) -> list:
    """
    Generate multiple topics from a user prompt using Ollama.
    
    Args:
        prompt (str): The user's input prompt
        
    Returns:
        list: List of generated topics
    """
    system_prompt = """
    You are an expert at analyzing prompts and breaking them down into specific, focused search topics.
    Given a user's prompt, generate 3-5 specific, relevant topics that would help answer the question comprehensively.
    Each topic should be a brief, clear phrase (max 5 words) that focuses on a specific aspect of the prompt.
    Do not add any explanations or formatting - just return the topics as a list.
    """
    
    user_prompt = f"""
    Analyze the following prompt and generate 3-5 specific search topics:
    "{prompt}"
    
    Each topic should be a brief, clear phrase (max 5 words) that focuses on a specific aspect of the prompt.
    Return only the topics, one per line, without any additional formatting or explanations.
    """
    
    response = chat(
        model='gpt-oss:20b',
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
    
    topics = response['message']['content'].strip().split('\n')
    # Filter out empty topics
    topics = [topic.strip() for topic in topics if topic.strip()]
    return topics

def BasicWebSearchContextGeneration(topic: str) -> str:
    """ 
    Implement web search context generation combining all components:
    1. Generate search queries with Ollama
    2. Fetch URLs from SearXNG
    3. Check relevance of results
    4. Scrape content
    5. Summarize relevant content
    6. Combine all into a single context
    """
    # Step 1: Get search results
    results = searx_get_urls_topic_page(topic, 1)
    
    if not results:
        return "No search results found."
    
    # Step 2: Filter relevant results
    relevant_results = []
    for result in results:
        if check_preview_relevance(topic, result['title'], result['url'], result['content']):
            relevant_results.append(result)
    
    # Step 3: Scrape content from relevant results
    scraped_results = []
    for result in relevant_results:
        try:
            scraped = scrape_and_clean_url_content(result['url'])
            scraped['url'] = result['url']
            scraped['title'] = result['title']
            scraped['content'] = scraped['content']
            scraped_results.append(scraped)
        except Exception as e:
            print(f"Error scraping {result['url']}: {e}")
            continue
    
    # Step 4: Summarize content
    summaries = []
    for scraped in scraped_results:
        try:
            summary = summarize_content(scraped['content'], scraped['title'], topic)
            summaries.append({
                'title': scraped['title'],
                'url': scraped['url'],
                'summary': summary
            })
        except Exception as e:
            print(f"Error summarizing {scraped['title']}: {e}")
            continue
    
    # Step 5: Check summary relevance and resummarize if needed
    final_summaries = []
    for summary in summaries:
        final_summary = check_summary_relevance(topic, summary['summary'])
        final_summaries.append({
            'title': summary['title'],
            'url': summary['url'],
            'summary': final_summary
        })
    
    # Step 6: Combine all summaries into context
    combined_context = ""
    for summary in final_summaries:
        combined_context += f"Title: {summary['title']}\n"
        combined_context += f"URL: {summary['url']}\n"
        combined_context += f"Summary: {summary['summary']}\n"
        combined_context += "\n" + "="*50 + "\n\n"
    
    return combined_context

def process_user_query(user_prompt: str) -> str:
    """
    Process a user query by generating topics, searching, and generating a response.
    
    Args:
        user_prompt (str): The user's input prompt
        
    Returns:
        str: Final response to the user
    """
    print(f"Processing query: {user_prompt}")
    
    # Generate topics from the prompt
    topics = generate_topics_from_prompt(user_prompt)
    print(f"Generated topics: {topics}")
    
    # Collect context for all topics
    all_contexts = []
    for topic in topics:
        print(f"Searching for topic: {topic}")
        context = BasicWebSearchContextGeneration(topic)
        all_contexts.append(context)
    
    # Combine all contexts
    combined_context = "\n".join(all_contexts)
    
    # Generate final response using the combined context
    system_prompt = """
    You are an expert assistant that answers questions based on provided context.
    Your task is to analyze the given context and provide a comprehensive, accurate, and helpful response to the user's original question.
    If the context doesn't contain sufficient information, honestly state that.
    """
    
    user_prompt = f"""
    Based on the following context, answer the original question: "{user_prompt}"
    
    Context:
    {combined_context}
    
    Provide a clear, well-structured response that directly addresses the user's question.
    """
    
    response = chat(
        model='gpt-oss:20b',
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ],
        images=[],
        keep_alive=600,
        tools=[],
    )
    
    return response['message']['content'].strip()

def search_with_context(query: str, context: str):
    """Search with the provided context"""
    # This function would implement the search logic with context
    # For now, we'll use the existing search functions
    return searx_get_urls_topic_page(query, 1)

def run_agent(query: str, results):
    """Run the agent with search results"""
    # This function would implement the agent logic
    # For now, we'll just process the query
    response = process_user_query(query)
    print(response)

def main():
    print("POTATO Assistant is ready. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("Enter your query: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
              
            if not query:
                continue
              
            print(f"Processing query: {query}")
            
            # Get system info
            system_info = json_get_instant_system_info()
            print("System Information:")
            for key, value in system_info.items():
                print(f"  {key}: {value}")
            print()
            
            # Process the query
            response = process_user_query(query)
            print(f"Response: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import subprocess
        subprocess.run(["ollama", "stop", "gpt-oss:20b"])
        print("Exiting...")
        import sys 
        sys.exit(1)
        print("Bye!")
