# Article-Retrieval-System-with-Llama-7b
This project is a Python script designed for information retrieval using a combination of techniques including semantic search and keyword matching. The script also integrates a LLama-2-7b-chat-hf for natural language processing tasks. The retrieval process is guided by a predefined system prompt, so the responses adhere to specified guidelines. 

## Setup
1. Clone the repository to your local machine:
"git clone <repository_url>"
2. Navigate to the project directory:
"cd information-retrieval-system"
3. Install the required dependencies using pip:
"pip install -r requirements.txt"
4. Obtain the necessary Hugging Face token and update the hf_token variable in the script.
5. Place your data files (e.g., medium.csv) in the appropriate directory.
## Configuration
1. Modify the SYSTEM_PROMPT variable in the script to customize the system prompt according to your requirements.
2. Adjust the quantization_config and other parameters as needed for model configuration.
## Operation
1. Ensure that the script (information_retrieval.py) is properly configured and all dependencies are installed.
2. Run the script:
"python information_retrieval.py"
3. Enter user query when prompted and observe the system's responses.
4. Review the source documents associated with each response for further context.
