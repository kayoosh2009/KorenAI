import sqlite3
import os
from datetime import datetime
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

DB_PATH = "koren_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_memory(user_id: str, fact: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO memories (user_id, fact) VALUES (?, ?)',
        (user_id, fact)
    )
    conn.commit()
    conn.close()

def get_memories(user_id: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT fact FROM memories WHERE user_id = ? ORDER BY timestamp DESC',
        (user_id,)
    )
    facts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return facts

def extract_and_save_facts(user_id: str, user_message: str, ai_response: str):
    prompt = f"""Analyze this conversation and extract any personal facts about the user.
If there are no new facts, respond with "NO_FACTS".
If there are facts, list them one per line in format: "User's [fact]"

Conversation:
User: {user_message}
AI: {ai_response}

Facts:"""
    
    try:
        response = client.chat('gemma3:27b-cloud', messages=[
            {'role': 'user', 'content': prompt}
        ], stream=False)
        
        facts_text = response['message']['content'].strip()
        
        if facts_text and facts_text != "NO_FACTS":
            facts = [f.strip() for f in facts_text.split('\n') if f.strip()]
            for fact in facts:
                save_memory(user_id, fact)
                print(f"[Memory] Saved: {fact}")
    except Exception as e:
        print(f"[Memory] Error extracting facts: {e}")

def get_context_for_llm(user_id: str) -> str:
    facts = get_memories(user_id)
    if not facts:
        return ""
    
    facts_text = "\n".join([f"- {fact}" for fact in facts])
    return f"\n\nYou remember these facts about the user:\n{facts_text}"

init_db()