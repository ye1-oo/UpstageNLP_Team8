from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']

def get_embedding_function():
    embeddings = UpstageEmbeddings(
        model="solar-embedding-1-large-query", 
        api_key=upstage_api_key)
    return embeddings
