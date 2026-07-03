import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

SYSTEM_PROMPT = "You are Koren, your job is to be a comfortable conversationalist and teacher of English, you communicate in easy English and don't say a lot of text."

def generate_response(user_input: str) -> str:
    messages = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': user_input
        }
    ]
    
    full_response = ""
    for part in client.chat('gemma3:27b-cloud', messages=messages, stream=True):
        full_response += part['message']['content']
        
    return full_response

# Блок для быстрого теста файла
if __name__ == "__main__":
    test_prompt = "Why is the sky blue?"
    print(f"User: {test_prompt}")
    print(f"Koren: {generate_response(test_prompt)}")