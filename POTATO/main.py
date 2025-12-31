# The core of P.O.T.A.T.O -- Practical Omnipurpose Technical AI Tool Operator
## Uses ollama to interface with local LLMs.
## Uses langchain to manage chains, agents, and memory.(?)
## Uses a modular approach to add tools , utilities and functionalities. 
###(- ./components/vocal_tools/ , - ./components/visual_tools/ , - ./components/online_tools/ , - ./components/local_tools/ , - ./components/dangerous_tools/ , - ./components/n8n/ , - ./components/utilities/ )
### Uses a webui (./webui/) to provide an interface for users to interact with P.O.T.A.T.O. shows status, logs, system usage, prompts, desktop/browser, other useful parameters.


import requests
# from langchain import LLMChain, PromptTemplate
import dotenv
import os
import sys
from pathlib import Path
from components.utilities.get_system_info import json_get_instant_system_info
# Load environment variables from .env file
dotenv.load_dotenv()
inital_system_status = json_get_instant_system_info()
# print("Initial System Status:", inital_system_status)
gpu_check = inital_system_status.get("gpus", [])
if gpu_check and not ("error" in gpu_check[0]):
    print("GPU detected:")
    for gpu in gpu_check:
        print(gpu)
        print("-" * 40)
    print("GPU information retrieval successful.")
else:
    print("No GPU detected or error retrieving GPU info.")
