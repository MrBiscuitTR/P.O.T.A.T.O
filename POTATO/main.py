# The core of P.O.T.A.T.O -- Practical Omnipurpose Technical AI Tool Operator
## Uses ollama to interface with local LLMs.
## Uses langchain to manage chains, agents, and memory.(?)
## Uses a modular approach to add tools , utilities and functionalities. 
###(- ./components/vocal_tools/ , - ./components/visual_tools/ , - ./components/online_tools/ , - ./components/local_tools/ , - ./components/dangerous_tools/) , - ./components/utilities/ , ./webui/    optinally - ./components/memory_management/ , - ./components/agent_management/ )

import requests
from ollama import Ollama
from langchain import LLMChain, PromptTemplate
import dotenv
import os
import sys
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()
# Initialize Ollama client
ollama_client = Ollama()

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "components"))
sys.path.append(str(BASE_DIR / "webui"))
# Import custom modules