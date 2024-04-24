"""
# Install necessary libraries
!pip install llama-index
!pip install llama-index-experimental
!pip install llama-index-llms-huggingface
!pip install llama-index-embeddings-huggingface
!pip install transformers accelerate bitsandbytes

# Download "medium.csv"
!wget https://raw.githubusercontent.com/giciq/tensorflow_keras_notebooks/main/medium.csv
"""

# All the necessary imports
import os
import pandas as pd
from llama_index.core import Settings
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core.retrievers import (BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever,)
from typing import List

# Hugging Face token
hf_token = "<hugging_face_token>"

"""## Defining LLM model (Llama-2-7b-chat-hf)"""

# This prompt template is used to instruct the assistant on how to respond to user queries.
SYSTEM_PROMPT = """[INST] <>
- You are a helpful assistant that is to find answers to user's questions according to the file you are given.
- Use only tool named retriever!
- DON'T SPREAD FALSE INFORMATION.
- If you don't find an answer tell that you can't answer with the provided file.
- Be kind and helpful.
"""

# This template wraps the user query within the context of the assistant's guidelines.
query_wrapper_prompt = SimpleInputPrompt(
    "{query_str}[/INST] "
)

# Define quantization configuration for the model.
# Used to reduce the memory footprint and increase inference speed of neural networks.
# Load data in 4-bit format, using 16-bit floating-point computation,
# Select a specific quantization type ("nf4"), and enable double quantization.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Instantiate the HuggingFaceLLM  with specified parameters.
llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    # Setting the maximum number of new tokens that can be generated in a single inference.
    max_new_tokens=400,
    # Providing a system prompt that will be used during inference.
    system_prompt=SYSTEM_PROMPT,
    # Providing a wrapper prompt for query processing.
    query_wrapper_prompt=query_wrapper_prompt,
    # Specifying the size of the context window to be used during inference.
    context_window=3900,
    model_kwargs={"token": hf_token, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hf_token},
    device_map="auto",
)

# Create a service context using default configurations.
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5"
)

# Set the global service context with the newly created service context.
set_global_service_context(service_context)

"""## Loading data, extracting nodes and initializing storage context"""

# Load data
documents = SimpleDirectoryReader(input_files=["medium.csv"]).load_data()

# Extract nodes (representing documents) from the provided documents using a node parser.
nodes = Settings.node_parser.get_nodes_from_documents(documents)

# Initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

"""## Creating custom retriever"""

# Create a vector index for the documents.
# This index allows for efficient retrieval of documents based on vector representations.
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

# Create a keyword index for the documents.
# This index allows for efficient retrieval of documents based on keyword matches.
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

# Custom retriever class
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

# Define retrievers
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)

custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

# Assemble query engine
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
)

"""## Testing the Retriever that uses Llama as a LLM"""

# Make a prompt
prompt = "Describe Logistic Regression"

# Response with the custom retriever and Llama-7b
response = custom_query_engine.query(prompt)
print(response)

# Sources with which the response was generated
for node in response.source_nodes:
  print("Source:\n" + node.text + "\n\n\n")

