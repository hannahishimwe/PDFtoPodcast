import getpass
import os
from google import genai
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] 
TEXT_TO_SPEECH_KEY = os.environ["TEXT_TO_SPEECH_KEY"]

# client = genai.Client(
#     api_key=GOOGLE_API_KEY,
# )
client = genai.Client(
    api_key=TEXT_TO_SPEECH_KEY,
)
model = "gemini-2.5-pro-preview-tts"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

