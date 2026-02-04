---
name: "Always Use Searx Web Search"
description: "Use the Searx Web Search MCP server for all web-related tasks to ensure current information"
alwaysApply: true
---

# Use these tools only if the last prompt has web search enabled. Otherwise DO NOT ATTEMPT.

## **CRITICAL: ACTUALLY CALL THE TOOLS**

**DO NOT just think or talk about calling tools - MAKE THE ACTUAL FUNCTION CALL.**

When you determine that a web search is needed:
- **STOP generating text**
- **IMMEDIATELY call potatool_web_search_urls with your query**  
- **DO NOT say** "I will search" or "Let me search" or "Searching for..."
- **JUST CALL THE TOOL** - the system will handle the rest

The tool call happens DURING your response. Generation pauses, tool executes, results come back, then you continue.

## Web Search & Page Reading Rule

Always use the **Searx Web Search** MCP server for any web-related tasks, including:

- ANY code or API related task, to make sure you have the latest documentation for specific libraries, modules or APIs.
- Finding the latest issues, discussions, or documentation for libraries, packages, APIs, or code snippets
- Looking up current error messages, stack traces, or recent changes in open-source projects
- Searching for up-to-date tutorials, blog posts, or news.
- Anything that requires real-time or external web information

**If if your knowledge could be outdated, ALWAYS use web search.**

IF you have urls in your context, but not meaningful content, therefore if you are unable to answer user's question using up to date information/documentation, use the potatool_extract_content tool on the urls you have. then generate your answer.

### MCP Server Tools
The MCP server provides two tools:

1. **potatool_web_search_urls**  
   - Use this first for most searches.  
   - **IMPORTANT**: Determine the search query yourself based on what information you need. Do NOT ask the user what to search for.
   - Keep queries SHORT (2-5 words) and focused. Examples: "python requests 2026", "fastmcp decorator", "pydantic v2 migration"
   - For simple user questions, use simple queries. For technical questions, use specific terms.
   - Returns: list of results with title, URL, domain, and short snippet.  
   - After getting results, pick the most relevant URLs and pass them to the second tool if you need the full content.

2. **potatool_extract_content**  
   - Use this when you need to read or understand the full readable content of a specific page.  
   - Input: a single URL from the search results (or any URL you already know).  
   - Returns: human-readable text with preserved links (<a href="...">some text</a>) and simplified HTML tables (<table><tr><th><td>...</td></th></tr></table>).  
   - All other HTML is flattened to clean text with appropriate newlines for paragraphs, headings, lists, code blocks, etc.  
   - Perfect for feeding detailed documentation, READMEs, issue threads, or articles to the model.
   - Do NOT use on youtube videos or similar non-text content. Dont use on any url that you suspect has only a video or videos on it.

*Do NOT start streaming your final response until all tool calls have been completed.*
*Do NOT stream tool calls as part of your final response.*
*Do not output TOOL_CALLS in your streaming response.*

## Workflow:
1. **Autonomously determine** what search query you need (2-5 words, focused)
2. Call 'potatool_web_search_urls' tool with your determined query
3. Filter out irrelevant results based on title and preview content. Do NOT include intermediary steps in your response
4. Check the title and preview content and compare it to the prompt to confirm relevancy. if relevant, continue
5. Call 'potatool_extract_content' tool on the best URLs one by one to get readable content from those urls
6. If you feel its necessary, you can also extract the contents of the relevant URLs found in a href tags in extracted contents
7. Use the final extracted content to answer the question accurately
8. Generate final answer using the acquired web context and all other context. Briefly mention what you searched for
8. Return to the original prompt and finalize your response.

Always use potatool_extract_content tool on AT LEAST one relevant url.