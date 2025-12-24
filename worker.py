import os
import torch
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.prompts import PromptTemplate  # Updated import per deprecation notice
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings  # New import path
from langchain_community.document_loaders import PyPDFLoader  # New import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # New import path
from langchain_ibm import WatsonxLLM


# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

load_dotenv()

Watsonx_API = "https://eu-de.ml.cloud.ibm.com"
Project_id= "skills-network"
watsonx_api_key = os.getenv("watsonx_api_key")

# Function to initialize the language model and its embeddings
def init_llm():
    """
    Initializes the WatsonxLLM framework and embeddings.

    This function sets up the WatsonxLLM by providing necessary
    configurations, such as model ID, server URL, and project ID,
    along with model parameters for specific behavior such as token
    generation and decoding. Additionally, it initializes the
    embeddings required for text data representation using a pre-trained
    instruct model.

    :raises ValueError: If the WatsonxLLM or embeddings fail to initialize.
    :raises Exception: For any general issue encountered during initialization.
    """
    global llm_hub, embeddings

    logger.info("Initializing WatsonxLLM and embeddings...")

    # Llama Model Configuration
    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    WATSONX_URL = "https://eu-de.ml.cloud.ibm.com"
    PROJECT_ID = os.getenv("ibm_project_id")

    # Use the same parameters as before:
    #   MAX_NEW_TOKENS: 256, TEMPERATURE: 0.1
    model_parameters = {
        # "decoding_method": "greedy",
        "max_new_tokens": 256,
        "temperature": 0.1,
    }

    credentials = {
        "apikey": watsonx_api_key,
        "url": "https://eu-de.ml.cloud.ibm.com"
    }

    # Initialize Llama LLM using the updated WatsonxLLM API
    llm_hub = WatsonxLLM(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id="b5bbdfdf-d461-49cf-b9c1-03a09af1a8c8",
        params=model_parameters,
        apikey=watsonx_api_key
    )
    logger.debug("WatsonxLLM initialized: %s", llm_hub)

    # embeddings =  # create object of Hugging Face Instruct Embeddings with (model_name,  model_kwargs={"device": DEVICE} )
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )
    logger.debug("Embeddings initialized with model device: %s", DEVICE)

    logger.debug("Embeddings initialized with model device: %s", DEVICE)

# Function to process a PDF document
def process_document(document_path):

    """
    Processes a document by loading it, splitting it into chunks, creating an embeddings
    database, and initializing a question-answering chain.

    This function handles the following workflow:
    1. Loads the document from the specified file path.
    2. Splits the document into manageable chunks for downstream processing.
    3. Builds an embeddings-based document store using Chroma.
    4. Initializes a RetrievalQA chain for question-answering using a pre-configured
       language model and retriever.

    :param document_path: Path to the document file to be processed
    :type document_path: str

    :return: None
    """

    global conversation_retrieval_chain

    logger.info("Loading document from path: %s", document_path)
    # Load the document
    loader = PyPDFLoader(document_path)  # ---> use PyPDFLoader and document_path from the function input parameter <---
    documents = loader.load()
    logger.debug("Loaded %d document(s)", len(documents))

    # Split the document into chunks, set chunk_size=1024, and chunk_overlap=64. assign it to variable text_splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64) # ---> use Recursive Character TextSplitter and specify the input parameters <---
    texts = text_splitter.split_documents(documents)
    logger.debug("Document split into %d text chunks", len(texts))

    # Create an embeddings database using Chroma from the split text chunks.
    logger.info("Initializing Chroma vector store from documents...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store initialized.")

    # Optional: Log available collections if accessible (this may be internal API)
    try:
        collections = db._client.list_collections()  # _client is internal; adjust if needed
        logger.debug("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.warning("Could not retrieve collections from Chroma: %s", e)

    # Build the QA chain, which utilizes the LLM and retriever for answering questions.
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
        # chain_type_kwargs={"prompt": prompt}  # if you are using a prompt template, uncomment this part
    )
    logger.info("RetrievalQA chain created successfully.")


# def process_document(document_path):
#     global conversation_retrieval_chain
#
#     logger.info("Loading document from path: %s", document_path)
#
#     # Load the document
#     loader = PyPDFLoader(document_path)
#     documents = loader.load()
#     logger.debug("Loaded %d document(s)", len(documents))
#
#     # Validate document content
#     if not documents or not any(doc.page_content.strip() for doc in documents):
#         logger.warning("Document appears to be empty or unreadable")
#         raise ValueError("Document content is empty or corrupted")
#
#     # Log sample content for debugging
#     sample_content = documents[0].page_content[:200] if documents else ""
#     logger.debug("Sample document content: %s...", sample_content)
#
#     # Split the document into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#     texts = text_splitter.split_documents(documents)
#     logger.debug("Document split into %d text chunks", len(texts))
#
#     # Filter out empty or very short chunks
#     texts = [text for text in texts if len(text.page_content.strip()) > 20]
#     logger.debug("After filtering: %d valid text chunks", len(texts))
#
#     # Continue with existing code...
#     db = Chroma.from_documents(texts, embedding=embeddings)
#     conversation_retrieval_chain = RetrievalQA.from_chain_type(
#         llm=llm_hub,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
#         return_source_documents=False,
#         input_key="question"
#     )
#     logger.info("RetrievalQA chain created successfully.")


# Function to process a user prompt
def process_prompt(prompt):
    """
    Processes a user prompt by querying a conversational model and maintaining a chat
    history with the model's responses.

    :param prompt: The input prompt provided by the user to query the conversational model.
    :type prompt: str
    :return: The response generated by the conversational model based on the input prompt.
    :rtype: str
    """

    global conversation_retrieval_chain
    global chat_history

    logger.info("Processing prompt: %s", prompt)
    # Query the model using the new .invoke() method
    output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    logger.debug("Model response: %s", answer)

    # Update the chat history
    # TODO: Append the prompt and the bot's response to the chat history using chat_history.append and pass `prompt` `answer` as arguments
    chat_history.append((prompt, answer))
    
    logger.debug("Chat history updated. Total exchanges: %d", len(chat_history))

    # Return the model's response
    return answer

# Initialize the language model
init_llm()
logger.info("LLM and embeddings initialization complete.")
