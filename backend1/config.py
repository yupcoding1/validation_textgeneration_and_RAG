import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and Model Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

