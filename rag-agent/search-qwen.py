import ollama
import httpx
import asyncio
from bs4 import BeautifulSoup
from typing import List, Optional, Set, Dict, Any, Tuple, Union
import signal
import sys
import json
from concurrent.futures import ThreadPoolExecutor
import re 
from datetime import datetime
import urllib.parse

# --- CONFIGURATION AND SETTINGS ---

# Ollama Models
MAIN_RAG_MODEL = "Qwen2.5:3B-Instruct"  # Main Response, Search Planner, and Final Synthesis
SUMMARIZATION_MODEL = "gemma:2b"        # Fast Summarization and Relevance Check

# Search API
SEARCH_API_URL = "https://searx.cagancalidag.com/search"

# SearXNG Engine Configuration (For Python logic)
AVAILABLE_SEARCH_ENGINES: List[Dict[str, str]] = [
    # General Search Engines (Default for most queries)
    {"name": "google", "shortcut": "g", "type": "General"},
    {"name": "duckduckgo", "shortcut": "ddg", "type": "General"},
    {"name": "bing", "shortcut": "b", "type": "General"},
    {"name": "brave", "shortcut": "br", "type": "General"},

    # Academic Search Engines (Used for 'academic sources', 'researchers', etc.)
    {"name": "google scholar", "shortcut": "gs", "type": "Academic"},
    {"name": "pubmed", "shortcut": "pm", "type": "Academic"},
    {"name": "arxiv", "shortcut": "ax", "type": "Academic"},

    # Community/Coding Search Engines (Used for general/coding queries)
    {"name": "reddit", "shortcut": "r", "type": "Community_Text"},
    {"name": "wikipedia", "shortcut": "w", "type": "General_Fallback"}, # Specific type for easy targeting
]

# Agent Settings
MAX_URLS_SCANNED = 25                   # Maximum total unique URLs to scrape
MAX_WORKERS = 3                         # ThreadPool workers for parallel summarization
MIN_SUCCESSFUL_URLS = 3                 # Minimum successful URLs required for initial search batches (per query)

# 🚨 CRITICAL CHANGE: Max URLs to scrape PER QUERY in Synthesis Mode (Fixed the over-scraping issue)
MAX_SYNTHESIS_URLS_PER_QUERY = 2        

# Recursive Scraping Limits
MAX_RECURSIVE_DEPTH = 1                 # Max depth for documentation sites
MAX_LINKS_PER_LEVEL = 3                 # Max number of relevant links to follow from one page

# Ollama Context Settings
MAIN_MODEL_CTX = 8192 * 4               # 32k context for the main LLM synthesis
SUMM_MODEL_CTX = 8192 * 1
CHUNK_SIZE = 5000                       # Text chunking size 

# Known media/non-text heavy domains to exclude from scraping
MEDIA_DOMAINS = ["youtube.com", "youtu.be", "pinterest.com", "vimeo.com", "reddit.com"]

# Thread Pool for Parallel Model Processing
THREAD_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)
GLOBAL_SCANNED_URLS: Set[str] = set() # Global set to prevent duplicate scrapes and recursive loops

# --- UTILITY FUNCTIONS ---

def detect_language_for_output(text: str) -> str:
    """
    Uses the MAIN_RAG_MODEL to determine the output language based on user rules.
    """
    local_client = ollama.Client() 
    
    prompt = (
        "Analyze the following text. If the language is French, respond ONLY with the word 'French'. "
        "For any other language (English, Turkish, Spanish, German, etc.), respond ONLY with the word 'English'. "
        "Do not use any punctuation, explanation, or extra words. "
        f"TEXT: {text[:200]}" 
    )
    
    try:
        response = local_client.generate(
            model=MAIN_RAG_MODEL, 
            prompt=prompt, 
            options={'num_ctx': 2048, 'temperature': 0.1} 
        )
        
        result = response['response'].strip().upper()
        
        if "FRENCH" in result:
            return "French"
        
    except Exception as e:
        print(f"⚠️ AI Language detection failed, defaulting to English. Error: {e}")
        
    return "English"

# ... (get_base_url, get_links, is_relevant_link_heuristic, summarize_text_sync, check_relevance_sync functions remain the same) ...
def is_valid_url(url: str) -> bool:
    return not any(domain in url.lower() for domain in MEDIA_DOMAINS)

def get_base_url(url: str) -> str:
    try:
        parsed = httpx.URL(url)
        return f"{parsed.scheme}://{parsed.host}"
    except:
        return ""

def get_links(html_content: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        
        try:
            full_url_obj = httpx.URL(base_url).join(href)
            full_url = str(full_url_obj)
        except ValueError:
            continue 

        if full_url.startswith('http') and not full_url.lower().endswith(('.pdf', '.zip', '.jpg', '.png', '.mp4', '.gif')):
            
            if get_base_url(full_url) == get_base_url(base_url):
                 links.append(full_url)
    
    return list(set(links))

def is_relevant_link_heuristic(link: str, original_query: str) -> bool:
    normalized_query = original_query.lower().replace(",", "").split()
    link_path = link.lower()
    
    if any(keyword in link_path for keyword in normalized_query if len(keyword) > 3):
        return True
    
    if any(term in link_path for term in ["docs", "guide", "tutorial", "integration", "workflow", "pgvector", "postgresql",
                                         "documentation", "official", "developer", "api", "code", "github", "stack", "latest", "new", "research", "professor", "university"]):
        return True
        
    return False

def summarize_text_sync(text_chunk: str, url: str) -> Optional[Dict[str, str]]:
    local_client = ollama.Client() 
    full_summary = ""
    
    chunks = [text_chunk[i:i + CHUNK_SIZE] for i in range(0, len(text_chunk), CHUNK_SIZE)]
    
    MAX_CHUNKS_ALLOWED = 20
    if len(chunks) > MAX_CHUNKS_ALLOWED:
        print(f"⚠️ SKIPPED: URL {url} has {len(chunks)} chunks, exceeding the limit of {MAX_CHUNKS_ALLOWED}.")
        return None

    print(f"   -> Summarizing: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        prompt = (
            "Summarize the following text concisely, capturing key points and relevant details. "
            "Summarize in its original language. "
            "Only use information in the given text:\n\nTEXT:\n"
            f"{chunk}"
        )
        
        if len(chunks) > 1:
             prompt = (f"This is chunk {i + 1}/{len(chunks)} of a document. Summarize this part in its original language. "
                      f"Do not invent information. Focus on continuity if possible. "
                      f"TEXT CHUNK:\n{chunk}")
        
        try:
            response = local_client.generate(
                model=SUMMARIZATION_MODEL, 
                prompt=prompt, 
                options={'num_ctx': SUMM_MODEL_CTX, 'temperature': 0.1}
            )
            full_summary += response['response'].strip() + " "
        except Exception as e:
            print(f"   ⚠️ Chunk {i+1} summarization error: {e}")
            
    if not full_summary.strip():
        return None
    
    return {"url": url, "summary": full_summary.strip()}

def check_relevance_sync(summary: Dict[str, str], user_prompt: str) -> Optional[Dict[str, str]]:
    local_client = ollama.Client()
    
    relevance_prompt = f"""
    Based ONLY on the following SUMMARY, determine if the content is highly relevant to the USER PROMPT.
    If the summary is about generic topics, cookie consent, or a completely different subject, respond ONLY with 'NO'.
    If the summary contains factual information or instructions directly addressing the prompt, respond ONLY with 'YES'.
    
    USER PROMPT: {user_prompt[:500]}
    SUMMARY: {summary['summary'][:500]}
    
    RESPONSE (YES/NO):
    """
    
    try:
        response = local_client.generate(
            model=SUMMARIZATION_MODEL, # Use the fast gemma:2b
            prompt=relevance_prompt,
            options={'num_ctx': 1024, 'temperature': 0.0} # Zero temperature for strict classification
        )
        
        result = response['response'].strip().upper()
        
        if "YES" in result:
            print(f"   -> Relevance Check: YES ({summary['url']})")
            return summary
        else:
            print(f"   -> Relevance Check: NO (Filtered: {summary['url']})")
            return None
            
    except Exception as e:
        print(f"   ⚠️ Relevance Check Failed for {summary['url']}, keeping due to error: {e}")
        return summary # Default to keeping if the check fails

async def fetch_searx_results(query: str, client: httpx.AsyncClient, page: int = 1, engines_to_use: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Asynchronously fetches search results (URL, title, snippet) from the Searx API.
    Returns a list of dictionaries, not just URLs.
    """
    params = {"q": query, "format": "json"}
    
    if engines_to_use:
        params["engines"] = "+".join(engines_to_use)
        print(f"    (Using engines: {', '.join(engines_to_use)})")
    
    if page > 1:
        params["pageno"] = page
        print(f"🔍 Searching Searx for: '{query}' (Page {page})")
    else:
        print(f"🔍 Searching Searx for: '{query}'")

    SEARX_TIMEOUT = 20
    
    try:
        response = await client.get(SEARCH_API_URL, params=params, timeout=SEARX_TIMEOUT) 
        response.raise_for_status() 

        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"❌ Searx query failed ({query}): Received invalid JSON response. Raw content: {response.text[:200]}...")
            return []
            
        results = [
            {'url': result.get('url'), 
             'title': result.get('title'),
             'content': result.get('content', '') # Snippet is crucial for extraction mode
            }
            for result in data.get('results', []) 
            if result.get('url', '').startswith('http')
        ]
        
        if not results:
            if response.status_code == 200:
                print(f"⚠️ Found 0 results for '{query}' (Page {page}). Searx returned valid JSON but no URLs (Engine rejection/timeout).")
            else:
                 print(f"⚠️ Found 0 results for '{query}' (Page {page}). Searx may have timed out or found no public results.")
        else:
            print(f"✅ Found {len(results)} results for '{query}' (Page {page}).")
            
        return results
        
    except httpx.ConnectTimeout as e:
        print(f"❌ Searx query failed ({query}): Connection timed out after {SEARX_TIMEOUT} seconds. Reason: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"❌ Searx query failed ({query}): HTTP Error {e.response.status_code}. Response: {e.response.text[:100]}...")
        return []
    except Exception as e:
        print(f"❌ Searx query failed ({query}): Unexpected error: {e}")
        return []

async def scrape_url(url: str, client: httpx.AsyncClient) -> Optional[Tuple[Dict[str, str], str]]:
    """Asynchronously scrapes a URL and extracts clean text and full HTML."""
    if url in GLOBAL_SCANNED_URLS:
        return None

    print(f"🌐 Scrape initiated: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
        }
        response = await client.get(url, follow_redirects=True, timeout=100, headers=headers)
        response.raise_for_status()
        
        GLOBAL_SCANNED_URLS.add(url)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'form']):
            script_or_style.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        print(f"✅ Scrape complete: {url}")
        
        return {"url": url, "text": clean_text}, response.text

    except httpx.HTTPStatusError as e:
        print(f"⚠️ HTTP Error ({url}): {e.response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Scrape failed ({url}): {e}")
        return None

async def recursive_doc_scrape(
    start_url: str, 
    original_query: str, 
    http_client: httpx.AsyncClient, 
    max_depth: int, 
    current_depth: int = 0
) -> List[Dict[str, str]]:
    """
    Performs depth-limited, same-domain scraping for official documentation.
    """
    
    if current_depth >= max_depth or len(GLOBAL_SCANNED_URLS) >= MAX_URLS_SCANNED:
        return []

    results: List[Dict[str, str]] = []

    scrape_result_tuple = await scrape_url(start_url, http_client)
    
    if scrape_result_tuple is None:
        return []
    
    scraped_data, html_content = scrape_result_tuple
    
    if scraped_data and len(scraped_data['text']) > 100:
        results.append(scraped_data)

    if current_depth + 1 >= max_depth:
        return results

    all_links = get_links(html_content, start_url)
    
    relevant_links = [
        link for link in all_links 
        if is_relevant_link_heuristic(link, original_query) and link not in GLOBAL_SCANNED_URLS
    ][:MAX_LINKS_PER_LEVEL]

    print(f"⬇️ Depth {current_depth+1}: Following {len(relevant_links)} relevant links from {get_base_url(start_url)}")

    recursive_tasks = [
        recursive_doc_scrape(link, original_query, http_client, max_depth, current_depth + 1)
        for link in relevant_links
    ]

    all_recursive_results = await asyncio.gather(*recursive_tasks)
    
    for sub_list in all_recursive_results:
        results.extend(sub_list)
        
    return results

def generate_search_queries(user_prompt: str, current_context: str) -> Tuple[bool, str, List[Dict[str, str]]]:
    """
    Uses the Main RAG Model to analyze the user's prompt, check necessity, and
    if necessary, generate targeted queries OR recovery queries based on current context.
    
    Returns: (search_required: bool, response_mode: str, query_list: List[Dict[str, str]])
    """
    print("\n🧠 Analyzing user request and checking search necessity/recovery...")
    client_for_planning = ollama.Client() 
    
    planning_prompt = f"""
    Analyze the original user request and the current collected information (SUMMARIZED SOURCE TEXTS).
    
    First, determine if the question is simple, conversational, or general knowledge that you can answer instantly using your internal knowledge. OR, if the provided SUMMARIZED SOURCE TEXTS are already sufficient to answer the complex query fully and accurately.
    
    Second, determine the RESPONSE MODE:
    - If the user explicitly asks for specific data fields (e.g., researcher names, dates, articles, links, versions), set 'response_mode' to 'Extraction'.
    - Otherwise (coding tasks, general questions, explanatory requests), set 'response_mode' to 'Synthesis'.

    If the question is complex AND the current source texts are insufficient, set 'search_required' to true and generate 3 to 5 NEW, specific, concise search queries.
    
    CRITICAL RULE: DO NOT GENERATE SEARCH QUERIES LONGER THAN 10 WORDS. BE EXTREMELY CONCISE.
    CRITICAL RULE: DO NOT INCLUDE THE FULL USER PROMPT IN A QUERY.
    
    **Current Date:** {datetime.now().strftime('%Y-%m-%d')}
    
    Format the output STRICTLY as a JSON object with three top-level keys:
    1. 'search_required': boolean (true/false).
    2. 'response_mode': string ('Extraction' or 'Synthesis').
    3. 'queries': array of objects (if search_required is true) OR an empty array (if false).

    **Query Generation Strategy (If search_required is true):**
    Generate 3 to 5 precise, independent, natural search engine queries focusing ONLY on the missing information.
    Each query object MUST have 'query' (string) and 'domain_type' ('Academic', 'Official_Doc', 'Community_Text', or 'General').
    
    EXAMPLE 1 (Extraction): {{"search_required": true, "response_mode": "Extraction", "queries": [ {{"query": "molecular biology obesity researchers Istanbul", "domain_type": "Academic"}}, ... ]}}
    EXAMPLE 2 (Synthesis): {{"search_required": false, "response_mode": "Synthesis", "queries": []}}
    
    ORIGINAL USER QUESTION:
    {user_prompt}
    
    SUMMARIZED SOURCE TEXTS (If empty, answer using internal knowledge):
    {current_context[:17000]}
    
    JSON PLAN:
    """
    
    try:
        response = client_for_planning.generate(
            model=MAIN_RAG_MODEL,
            prompt=planning_prompt,
            options={'num_ctx': 4096, 'temperature': 0.2} # Increased for better creativity/simplification
        )
        
        json_output = response['response'].strip()
        
        if json_output.startswith("```json"):
            json_output = json_output.strip("```json\n").strip("```")
        
        try:
            last_brace_index = json_output.rindex('}')
            json_output = json_output[:last_brace_index + 1]
        except ValueError:
            pass

        plan = json.loads(json_output)
        
        if isinstance(plan, dict) and 'search_required' in plan and 'queries' in plan and 'response_mode' in plan:
            search_required = plan['search_required']
            response_mode = plan['response_mode']
            queries = plan['queries']
            
            if search_required and not isinstance(queries, list):
                raise ValueError("Queries must be a list when search is required.")
                
            print(f"✅ Generated plan: Search Required = {search_required}, Mode = {response_mode}")
            return search_required, response_mode, queries
        else:
            raise ValueError("Parsed plan is missing required keys (search_required, response_mode, queries).")

    except Exception as e:
        print(f"❌ Search planning/necessity check failed. Assuming search is required (Synthesis mode). Error: {e}")
        fallback_query = " ".join(user_prompt.split()[:10])
        return True, "Synthesis", [{"query": fallback_query, "domain_type": "General"}]


# --- MAIN RAG FLOW ---

async def main_rag_agent(user_prompt: str):
    """Manages the main RAG agent flow with a Dynamic Recovery Loop."""
    
    loop = asyncio.get_event_loop()
    
    # 1. Initial Language Detection
    lang_task = loop.run_in_executor(THREAD_POOL, detect_language_for_output, user_prompt)
    original_prompt_language = await lang_task
    
    # Initialization for the recovery loop
    context_text = ""
    all_summaries: List[Dict[str, str]] = []
    response_mode = "Synthesis" # Default mode

    # --- DYNAMIC RECOVERY LOOP (Max 3 search iterations) ---
    for iteration in range(3):
        
        # 2. Search Necessity Check / Query Generation (Initial or Recovery)
        search_required, response_mode, search_query_objects = await loop.run_in_executor(
            THREAD_POOL, generate_search_queries, user_prompt, context_text
        )

        # 🚨 FIX: Minimal Search and Context Termination Logic
        
        force_minimal_search = False
        MIN_CONTEXT_LENGTH = 1500
        # Determine if we need a final forced search iteration for minimum context
        if len(context_text) < MIN_CONTEXT_LENGTH:
            if iteration < 2:
                 search_required = True
            elif iteration == 2:
                 # If this is the last iteration, and context is too short, force one final minimal search
                 search_required = True
                 response_mode = "Synthesis"
                 force_minimal_search = True # Use fixed minimal queries
                 search_query_objects = [
                     {"query": " ".join(user_prompt.split()[:5]) + " wikipedia", "domain_type": "General"},
                     {"query": " ".join(user_prompt.split()[:5]) + " official website", "domain_type": "Official_Doc"},
                     {"query": " ".join(user_prompt.split()[:5]), "domain_type": "General"},
                 ]
                 print("⚠️ Context below minimum threshold. Forcing final Minimal Synthesis search.")
            
        elif not search_required:
             # Context is sufficient, break the loop
             print(f"⏭️ Context (Length {len(context_text)}) is sufficient. Finalizing.")
             break
        
        print(f"\n--- STARTING SEARCH ITERATION {iteration + 1} ({response_mode} Mode) ---")
        
        if not search_query_objects:
            # If the planner couldn't generate queries (e.g., failed parsing), attempt a fallback search
            search_query_objects = [{"query": " ".join(user_prompt.split()[:5]), "domain_type": "General"}]


        # 3. Initial Search & Page Refresh Logic
        async with httpx.AsyncClient(verify=True) as http_client:
            
            initial_results_with_type: List[Tuple[Dict[str, str], str]] = [] # (Result Dict, Domain Type)
            
            # --- Engine Selection Logic ---
            first_domain_type = search_query_objects[0]['domain_type']
            is_academic_mode = (first_domain_type == 'Academic')
            
            engines_to_use = []
            if is_academic_mode:
                # Use only Academic engines
                engines_to_use = [e['name'] for e in AVAILABLE_SEARCH_ENGINES if e['type'] == 'Academic']
            else:
                 # Default to General and Community engines
                engines_to_use = [e['name'] for e in AVAILABLE_SEARCH_ENGINES if e['type'] in ('General', 'Community_Text')]
            
            if force_minimal_search and not is_academic_mode:
                 # If forced, use the simplest, most reliable engines (Google/DDG) and Wikipedia
                 engines_to_use = ['google', 'duckduckgo', 'wikipedia']
                 print("🎯 Running minimal synthesis search (Google, DDG, Wiki).")
            else:
                 print(f"🎯 Limiting search to: {', '.join(engines_to_use)}")
            
            
            # --- Search Execution ---
            
            # If academic, use multi-page search. If minimal, only use page 1.
            max_pages = 1 if force_minimal_search or response_mode == "Extraction" else 3 
            search_success_count = 0
            
            for query_obj in search_query_objects:
                query = query_obj['query']
                
                for page in range(1, max_pages + 1): 
                    results = await fetch_searx_results(query, http_client, page=page, engines_to_use=engines_to_use)
                    
                    for result in results:
                         if is_valid_url(result['url']):
                            initial_results_with_type.append((result, query_obj['domain_type']))
                            search_success_count += 1
                            
                    if search_success_count >= MIN_SUCCESSFUL_URLS:
                         break
            
            # --- DYNAMIC RETRIEVAL FALLBACK (If Academic Search Fails) ---
            if is_academic_mode and search_success_count == 0:
                print("🚨 ACADEMIC FALLBACK TRIGGERED: Zero results found on specialized engines. Switching to All Engines.")
                
                all_engines = [e['name'] for e in AVAILABLE_SEARCH_ENGINES if e['type'] in ('General', 'Academic', 'Community_Text')]
                
                for query_obj in search_query_objects:
                    query = query_obj['query']
                    results = await fetch_searx_results(query, http_client, page=1, engines_to_use=all_engines)
                    
                    for result in results:
                        if is_valid_url(result['url']):
                            initial_results_with_type.append((result, 'Academic')) 
                            search_success_count += 1
            

            # 4. Filter and Prepare for Processing
            unique_results_to_process: Dict[str, Tuple[Dict[str, str], str]] = {} 
            for result, domain_type in initial_results_with_type:
                if result['url'] not in GLOBAL_SCANNED_URLS and len(unique_results_to_process) < MAX_URLS_SCANNED:
                    unique_results_to_process[result['url']] = (result, domain_type)

            results_list = list(unique_results_to_process.items())
            
            if not results_list and len(context_text) < MIN_CONTEXT_LENGTH:
                print("❗ Searx returned no new results across all pages. Cannot proceed. Ending search.")
                break
                
            new_relevant_summaries = [] 
            
            # --- PROCESS RESULTS ---
            if response_mode == "Extraction":
                print("⏭️ Extraction Mode: Skipping heavy scraping/summarization.")
                
                for url, (result, domain_type) in results_list:
                    new_relevant_summaries.append({
                        'url': url,
                        'summary': f"TITLE: {result.get('title', 'N/A')}. SNIPPET: {result.get('content', 'No abstract or snippet available.')}",
                    })
                
                for url in unique_results_to_process.keys():
                    GLOBAL_SCANNED_URLS.add(url)
                
            else: # Synthesis Mode (Original RAG Logic)
                # 5. Asynchronous Scraping & Recursive Logic
                
                print(f"\n🚀 Total {len(results_list)} new unique URLs scheduled for scraping...")
                
                all_scraped_data: List[Dict[str, str]] = []
                scraping_tasks = []

                for url, (result, domain_type) in results_list:
                    if domain_type == 'Official_Doc':
                        print(f"Starting RECURSIVE scrape for: {url}")
                        original_query = search_query_objects[0]['query']
                        scraping_tasks.append(
                            recursive_doc_scrape(url, original_query, http_client, MAX_RECURSIVE_DEPTH)
                        )
                    else:
                        scraping_tasks.append(scrape_url(url, http_client))
                
                all_results_lists = await asyncio.gather(*scraping_tasks)
                
                for item in all_results_lists:
                    if isinstance(item, list): 
                        all_scraped_data.extend(item)
                    elif isinstance(item, tuple) and item[0] is not None: 
                        all_scraped_data.append(item[0])

                # 6. Parallel Summarization (Threading)
                texts_to_summarize = [data for data in all_scraped_data if len(data['text']) > 100]
                
                print(f"\n📝 {len(texts_to_summarize)} text chunks are being summarized (Model: {SUMMARIZATION_MODEL})...")
                
                summary_tasks = [
                    loop.run_in_executor(THREAD_POOL, summarize_text_sync, item['text'], item['url'])
                    for item in texts_to_summarize
                ]
                
                summarized_results = await asyncio.gather(*summary_tasks)
                
                # 7. RELEVANCE CHECK (Filter Step)
                print("\n🔍 Starting Relevance Check on Summaries...")
                relevance_tasks = [
                     loop.run_in_executor(THREAD_POOL, check_relevance_sync, summary, user_prompt)
                     for summary in summarized_results if summary is not None
                ]

                new_relevant_summaries = [res for res in await asyncio.gather(*relevance_tasks) if res is not None]
            
            
            # End of if/else (Extraction/Synthesis)
            all_summaries.extend(new_relevant_summaries) 
            
            if not all_summaries and iteration == 0:
                print("⚠️ Initial search yielded NO USEFUL summaries. Halting recovery.")
                break

            # Update context for the next iteration check
            context_text = "\n\n---\n---\n\n".join([f"SOURCE ({s['url']}):\n{s['summary']}" for s in all_summaries])

    # --- FINAL RESPONSE GENERATION (Outside the loop) ---
    
    # 8. Final Response Generation with Main Model
    
    final_sources_list = [s['url'] for s in all_summaries]
    
    # 9. CONTEXT OPTIMIZATION: Truncate context for low-VRAM model stability
    optimized_context = context_text
    MAX_SOURCES_FOR_CONTEXT = 15
    if len(final_sources_list) > MAX_SOURCES_FOR_CONTEXT:
        print(f"✂️ WARNING: Context is dense ({len(final_sources_list)} sources). Truncating context to the first {MAX_SOURCES_FOR_CONTEXT} summarized blocks for better synthesis stability.")
        top_summaries = [f"SOURCE ({s['url']}):\n{s['summary']}" for s in all_summaries[:MAX_SOURCES_FOR_CONTEXT]]
        optimized_context = "\n\n---\n\n".join(top_summaries)


    final_prompt = f"""
    You are an expert, up-to-date AI assistant. The current date is {datetime.now().strftime('%Y-%m-%d')}.
    
    ---
    
    ORIGINAL USER QUESTION:
    {user_prompt}
    
    SUMMARIZED SOURCE TEXTS (If empty, answer using internal knowledge):
    {optimized_context}
    
    If source texts were provided, provide a comprehensive, detailed, and fluent answer, synthesizing all relevant information, paying close attention to information currency. If the user requested specific data fields (e.g., researcher name, publication date, institution) and the source texts provide them, format the response as a clear, readable list or table. In such cases, avoid general summarizing and focus ONLY on extracting the requested fields.
    
    If source texts were NOT provided, answer the question briefly using only your internal knowledge, but only for simple, general knowledge questions. For factual questions requiring current data (like dates or versions) without context, state clearly that you cannot verify the information.
    
    Your final response MUST BE IN THE {original_prompt_language} LANGUAGE.
    
    RESPONSE:
    """
    
    print("\n" + "="*50)
    print(f"🤖 GENERATING FINAL RESPONSE... (Model: {MAIN_RAG_MODEL})")
    print(f"Final response language set to: {original_prompt_language}")
    print(f"RAG Context status: {'Used' if context_text else 'Skipped/Empty'}, Mode: {response_mode}")
    print("="*50)

    try:
        client_for_rag = ollama.Client() 
        response = client_for_rag.generate(
            model=MAIN_RAG_MODEL,
            prompt=final_prompt,
            options={'num_ctx': MAIN_MODEL_CTX, 'temperature': 0.5},
            stream=True 
        )
        for chunk in response:
            print(chunk['response'], end="", flush=True)
        print("\n" + "="*50)
        
        # 10. Final Source Listing (Lists ALL unique, successful URLs, not just the optimized subset)
        if final_sources_list:
             print("\n\nSOURCES SUCCESSFULLY SCANNED AND USED:")
             for i, url in enumerate(set(final_sources_list)):
                 print(f"  {i+1}. {url}")
        else:
             print("\n\nNo external sources were successfully summarized.")

    except Exception as e:
        print(f"\n❌ Main model response failed: {e}")


# --- EXECUTION BLOCK ---

def signal_handler(sig, frame):
    """Handles Keyboard Interrupt (Ctrl+C) for clean exit."""
    print("\n\n🛑 Keyboard Interrupt (Ctrl+C) received. Shutting down all operations...")
    try:
        oclient = ollama.Client()
        oclient.cancel_all()
    except:
        pass 
    THREAD_POOL.shutdown(wait=False, cancel_futures=True) 
    sys.exit(0)

if __name__ == "__main__":
    
    signal.signal(signal.SIGINT, signal_handler)

    USER_QUERY = str(
        input("📝 Enter your complex, possibly multilingual query for the RAG agent:\n> ")
    )
    
    try:
        print(f"🌟 Starting Autonomous Multilingual RAG Agent with query:\n'{USER_QUERY[:100]}...'")
        asyncio.run(main_rag_agent(USER_QUERY))
        
    except KeyboardInterrupt:
        pass 
    finally:
        THREAD_POOL.shutdown(wait=True)
        print("✅ Program terminated and resources cleaned up.")