import os
import time
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Load environment variables (Azure endpoint, deployment, keys, etc.)
load_dotenv()

logger = logging.getLogger(__name__)

# Retrieve credentials from .env file or environment
_raw_endpoint = os.getenv("gpt_endpoint")
deployment = os.getenv("gpt_deployment")
api_version = os.getenv("gpt_api_version")

if not _raw_endpoint or not deployment or not api_version:
    raise ValueError(
        "Missing required environment variables: gpt_endpoint, gpt_deployment, "
        "gpt_api_version must all be set in .env"
    )


def _normalize_endpoint(raw: str) -> str:
    """
    AzureOpenAI(azure_endpoint=...) expects the bare Cognitive Services
    resource root, e.g. https://<resource>.openai.azure.com/ — it appends
    /openai/deployments/{deployment}/chat/completions internally.

    If gpt_endpoint accidentally includes a Foundry-style suffix like
    /openai/v1 or /openai/v1/responses (a different API shape), strip it
    so the classic chat.completions.create() call keeps working.
    """
    endpoint = raw.strip().rstrip("/")
    for suffix in ("/openai/v1/responses", "/openai/v1/chat/completions", "/openai/v1"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return endpoint.rstrip("/") + "/"


endpoint = _normalize_endpoint(_raw_endpoint)

# Initialize Azure OpenAI client for GPT model using managed identity
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version=api_version,
)


def generate_response(text_input: str) -> str:
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """
    start_time = time.time()

    # Prepare the full chat prompt with system and user messages
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are a helpful assistant working for Zava, a company that specializes in offering products to assist homeowners with do-it-yourself projects.
                        Respond to customer inquiries with relevant product recommendations and DIY tips. If a customer asks for paint, suggest one of the following three colors: blue, green, and white.
                        If a customer asks for something not related to a DIY project, politely inform them that you can only assist with DIY-related inquiries.
                        Zava has a variety of store locations across the country. If a customer asks about store availability, direct the customer to the Miami store.
                    """
                }
            ]
        },
        {"role": "user", "content": text_input}
    ]

    try:
        # Call Azure OpenAI chat API
        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_completion_tokens=10000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
    except Exception:
        logger.exception("generate_response: Azure OpenAI call failed")
        raise
    finally:
        elapsed = time.time() - start_time
        print(f"generate_response Execution Time: {elapsed} seconds")

    # Return response content
    return completion.choices[0].message.content