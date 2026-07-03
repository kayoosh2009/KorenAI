import os
import asyncio
import threading
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv
import requests
import base64

# Импорты наших модулей
from generation import generate_response
from memory import init_db, extract_and_save_facts
from voice import speak_with_levels

load_dotenv()

# ==================== КОНФИГУРАЦИЯ ====================
WAKE_WORD = "hi koren"
SAMPLE_RATE = 16000
CHANNELS = 1
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO = os.environ.get('GITHUB_REPO')
LOCAL_DB_PATH = "koren_memory.db"

# ==================== GITHUB СИНХРОНИЗАЦИЯ ====================
def download_db_from_github():
    """Скачивает базу данных с GitHub если её нет локально"""
    if os.path.exists(LOCAL_DB_PATH):
        print("[GitHub] Local DB exists, skipping download")
        return
    
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("[GitHub] No GitHub credentials, skipping download")
        return
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{LOCAL_DB_PATH}"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json()['content']
            decoded_content = base64.b64decode(content)
            with open(LOCAL_DB_PATH, 'wb') as f:
                f.write(decoded_content)
            print("[GitHub] DB downloaded successfully")
        else:
            print(f"[GitHub] DB not found on GitHub (status {response.status_code})")
    except Exception as e:
        print(f"[GitHub] Error downloading DB: {e}")

def upload_db_to_github():
    """Загружает базу данных в GitHub"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("[GitHub] No GitHub credentials, skipping upload")
        return
    
    if not os.path.exists(LOCAL_DB_PATH):
        print("[GitHub] Local DB doesn't exist, skipping upload")
        return
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{LOCAL_DB_PATH}"
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Получаем SHA файла
    sha = None
    try:
        check_response = requests.get(url, headers={'Authorization': f'token {GITHUB_TOKEN}'})
        if check_response.status_code == 200:
            sha = check_response.json()['sha']
    except:
        pass
    
    # Читаем и кодируем файл
    with open(LOCAL_DB_PATH, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')
    
    data = {
        'message': 'Update Koren memory database',
        'content': content
    }
    if sha:
        data['sha'] = sha
    
    try:
        response = requests.put(url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            print("[GitHub] DB uploaded successfully")
        else:
            print(f"[GitHub] Error uploading DB: {response.status_code}")
    except Exception as e:
        print(f"[GitHub] Error uploading DB: {e}")

# ==================== FASTAPI ПРИЛОЖЕНИЕ ====================
app = FastAPI()

# Храним активные WebSocket соединения
active_connections: list[WebSocket] = []

@app.get("/")
async def root():
    """Отдаем index.html"""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для связи с веб-интерфейсом"""
    await websocket.accept()
    active_connections.append(websocket)
    print("[WebSocket] Client connected")
    
    try:
        while True:
            # Ждем сообщений от клиента (пока не используем)
            data = await websocket.receive_text()
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        active_connections.remove(websocket)
        print("[WebSocket] Client disconnected")

async def broadcast_message(message: dict):
    """Отправляет сообщение всем подключенным клиентам"""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            pass

# ==================== АУДИО ОБРАБОТКА ====================
def detect_wake_word(audio_data: np.ndarray) -> bool:
    """
    Простая проверка на wake word "Hi Koren"
    В продакшене лучше использовать openwakeword или vosk
    """
    # Пока заглушка - в реальности нужно использовать STT для проверки
    # Это временное решение, потом заменим на нормальное
    return False

def record_audio(duration: int = 5) -> np.ndarray:
    """Записывает аудио с микрофона"""
    print("[Audio] Recording...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32'
    )
    sd.wait()
    print("[Audio] Recording complete")
    return audio.flatten()

def transcribe_audio(audio: np.ndarray) -> str:
    """
    Преобразует аудио в текст
    Используем faster-whisper для локального распознавания
    """
    try:
        from faster_whisper import WhisperModel
        
        # Загружаем модель (базовая, быстрая)
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # Whisper ожидает float32 в диапазоне [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Транскрибируем
        segments, info = model.transcribe(audio, language="en")
        text = " ".join([segment.text for segment in segments])
        
        print(f"[STT] Transcribed: {text}")
        return text.strip()
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""

# ==================== ГЛАВНЫЙ ЦИКЛ ====================
def listen_loop():
    """
    Постоянно слушает микрофон и обрабатывает команды
    """
    print("[Main] Starting listen loop...")
    
    # Для wake word используем простую логику
    # В реальности лучше использовать openwakeword
    is_listening = False
    silence_threshold = 0.01
    silence_duration = 1.5  # секунд тишины для остановки записи
    
    while True:
        try:
            # Записываем короткими кусками
            chunk_duration = 0.5
            audio_chunk = sd.rec(
                int(chunk_duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32'
            )
            sd.wait()
            
            # Проверяем громкость
            volume = np.abs(audio_chunk).mean()
            
            # Простая логика: если громко - начинаем слушать
            if not is_listening and volume > 0.05:
                print("[Main] Detected speech, starting recording...")
                is_listening = True
                
                # Записываем полный вопрос
                full_audio = record_audio(duration=10)  # максимум 10 секунд
                
                # Транскрибируем
                user_text = transcribe_audio(full_audio)
                
                if user_text:
                    print(f"[Main] User said: {user_text}")
                    
                    # Отправляем в веб-интерфейс
                    asyncio.run(broadcast_message({
                        "type": "user_text",
                        "text": user_text
                    }))
                    
                    # Генерируем ответ
                    print("[Main] Generating response...")
                    ai_response = generate_response(user_text)
                    print(f"[Main] Koren: {ai_response}")
                    
                    # Отправляем текст в веб
                    asyncio.run(broadcast_message({
                        "type": "ai_text",
                        "text": ai_response
                    }))
                    
                    # Озвучиваем с уровнями громкости
                    print("[Main] Speaking...")
                    samples, levels, sample_rate = speak_with_levels(ai_response)
                    
                    # Отправляем уровни громкости для анимации
                    for level in levels:
                        asyncio.run(broadcast_message({
                            "type": "audio_level",
                            "level": float(level)
                        }))
                    
                    # Извлекаем факты из диалога
                    extract_and_save_facts("default_user", user_text, ai_response)
                    
                    # Загружаем обновленную базу в GitHub
                    upload_db_to_github()
                
                is_listening = False
                
        except KeyboardInterrupt:
            print("[Main] Stopping...")
            break
        except Exception as e:
            print(f"[Main] Error in listen loop: {e}")

# ==================== ЗАПУСК ====================
def main():
    """Главная функция запуска"""
    print("[Main] Initializing Koren...")
    
    # Инициализируем базу данных
    init_db()
    
    # Скачиваем базу с GitHub если нужно
    download_db_from_github()
    
    # Запускаем веб-сервер в отдельном потоке
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("[Main] Web server started on http://localhost:8000")
    
    # Запускаем цикл прослушивания в основном потоке
    listen_loop()

if __name__ == "__main__":
    main()