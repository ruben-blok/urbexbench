import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')
url = "https://openrouter.ai/api/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)

print(response.json())