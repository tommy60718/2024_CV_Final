from dotenv import load_dotenv
import openai
import os


load_dotenv('./.env')
print("OPENAI_API_KEY:", os.getenv('OPENAI_API_KEY'))
# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

print("Testing API Key...")
try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, can you respond to me?"}],
        max_tokens=50
    )
    print("API Response:", response['choices'][0]['message']['content'])
except Exception as e:
    print("API Error:", e)
