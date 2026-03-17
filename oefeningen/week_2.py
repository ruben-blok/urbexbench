import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

image_path = "/home/ruben/Documenten/Code/urbexbench/img/abandoned/50.397637_4.498408.png"

client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1" 
)

base64_image = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Beschrijf wat je ziet op deze afbeelding"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
    }
]

response = client.chat.completions.create(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    messages=messages
)

result = response.choices[0].message.content

print("AI Antwoord:")
print(result)