from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from config.settings import settings

class LLMClient:
    def __init__(self):
        # Initialize connection to Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment_name=settings.AZURE_GPT4O_DEPLOYMENT,
            max_tokens=4096,
            
        )
        
        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_EMBED_DEPLOYMENT
        )
        
    def get_llm(self) -> AzureChatOpenAI:
        return self.llm
        
    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        return self.embeddings

llm_client = LLMClient()
