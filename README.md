# P.O.T.A.T.O
## Practical Omnipurpose Technical AI Tool Operator, aka P.O.T.A.T.O , is a smart and locally running vocal AI assistant that is capable of pretty much anything. It accepts all kinds of input including voice, images, code files, folders, pdfs... It is secure and private, overly so. Needs good hardware (24+ GB of VRAM and a fast GPU) to run efficiently. Uses SearX for web search. can be configured to use TOR. It is also capable of being your friend.

P.O.T.A.T.O is a work in progress and will be updated regularly as the author finds the time to do so. It is a modular system that can be easily extended with new tools, utilities and functionalities. This project will undergo continuous development through the upcoming years.
    
### The core of P.O.T.A.T.O -- Practical Omnipurpose Technical AI Tool Operator
- Uses ollama to interface with local LLMs.
- Uses langchain to manage chains, agents, and memory.(?)
- Uses a modular approach to add tools , utilities and functionalities. 
- Components :
  - ./components/vocal_tools/  
  - ./components/visual_tools/ 
  - ./components/online_tools/ 
  - ./components/local_tools/ 
  - ./components/dangerous_tools/
  - ./components/utilities/ 
  - ./components/n8n/ 
  - ./webui/    
  optinally: 
  - ./components/memory_management/ 
  - ./components/agent_management/ 

## Installation

To install P.O.T.A.T.O,

1. Clone this repository.
2. Navigate to the project directory.
3. Rename `config.env.txt` to `.env`.
4. Open `.env` and configure the necessary environment variables
5. Install dependencies using pip and requirements.txt: ```pip install -r requirements.txt``` (run this on project root)
6. Run `main.py`. The Web GUI runs at `http://localhost:8080`. If configured correctly, it should be listening the wake word. once heard, the program will start interacting with you until you stop it manually OR say a sleep word.

## Setting up the environment variables

You MUST rename the config.env.txt and remove the .txt at the end of it. then, start configuring the file, based on your setup. The fields have explanations next to them inside comments.