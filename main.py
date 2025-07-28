from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
import shutil
import openai
import os
import re
import unicodedata

# === Cargar variables de entorno ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Configuración de directorios absolutos ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audios")
RESPONSE_DIR = os.path.join(BASE_DIR, "responses")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)

# === Configuración de la base de datos ===
DATABASE_URL = "sqlite:///./interactions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    audio_filename = Column(String)
    transcription = Column(String)
    ia_response = Column(String)
    tts_audio_filename = Column(String)

class StaticResponse(Base):
    __tablename__ = "static_responses"
    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String, unique=True, index=True)
    answer = Column(String)

Base.metadata.create_all(bind=engine)

# === Modelos Pydantic ===
class TextToSpeechInput(BaseModel):
    text: str

class StaticResponseCreate(BaseModel):
    keyword: str
    answer: str

# === Instancia de la aplicación FastAPI ===
app = FastAPI(
    title="Asistente de Voz IA",
    description="Recibe audio, responde con voz generada por IA usando Whisper + GPT-4o + TTS.",
)

# === Configuración CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Montar carpeta de respuestas en /responses ===
app.mount("/responses", StaticFiles(directory=RESPONSE_DIR), name="responses")

# === Función para convertir texto a slug ===
def slugify(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[-\s]+', '-', text)

# === Endpoint: audio -> texto -> respuesta IA -> audio ===
@app.post("/ask", summary="Envía un audio, recibe respuesta en audio (mp3)")
async def ask_ai(file: UploadFile = File(...)):
    try:
        ext = file.filename.split('.')[-1]
        input_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.{ext}"
        input_path = os.path.join(AUDIO_DIR, input_filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando audio: {str(e)}")

    # Transcribir con Whisper
    try:
        with open(input_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        texto_transcrito = transcript.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribiendo: {str(e)}")

    # Buscar respuesta estática
    db = SessionLocal()
    ia_response = None
    try:
        for sr in db.query(StaticResponse).all():
            if sr.keyword.lower() in texto_transcrito.lower():
                ia_response = sr.answer
                break
    finally:
        db.close()

    # Si no hay respuesta estática, usar GPT
    if ia_response is None:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": texto_transcrito}]
            )
            ia_response = response.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error con GPT-4o: {str(e)}")

    # Generar voz con TTS
    try:
        tts_audio_filename = f"response_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.mp3"
        tts_audio_path = os.path.join(RESPONSE_DIR, tts_audio_filename)
        tts_response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ia_response
        )
        with open(tts_audio_path, "wb") as f:
            f.write(tts_response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando audio TTS: {str(e)}")

    # Guardar en la base de datos
    db = SessionLocal()
    try:
        inter = Interaction(
            audio_filename=input_filename,
            transcription=texto_transcrito,
            ia_response=ia_response,
            tts_audio_filename=tts_audio_filename
        )
        db.add(inter)
        db.commit()
    finally:
        db.close()

    return FileResponse(tts_audio_path, media_type="audio/mpeg", filename=tts_audio_filename)

# === Endpoint: convierte texto a voz directamente ===
@app.post("/text-to-speech", summary="Convierte texto a voz (audio mp3)")
def text_to_speech(payload: TextToSpeechInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

    try:
        tts_audio_filename = f"tts_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.mp3"
        tts_audio_path = os.path.join(RESPONSE_DIR, tts_audio_filename)
        tts_response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=payload.text
        )
        with open(tts_audio_path, "wb") as f:
            f.write(tts_response.content)

        db = SessionLocal()
        try:
            interaction = Interaction(
                audio_filename=None,
                transcription=payload.text,
                ia_response=payload.text,
                tts_audio_filename=tts_audio_filename
            )
            db.add(interaction)
            db.commit()
        finally:
            db.close()

        return FileResponse(tts_audio_path, media_type="audio/mpeg", filename=tts_audio_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando audio desde texto: {str(e)}")

# === Endpoint: retorna audio pregrabado según pregunta ===
@app.get("/audio-from-question", summary="Devuelve audio pregrabado para una pregunta")
def get_audio_from_question(question: str = Query(..., description="Texto exacto de la pregunta")):
    slug = slugify(question)
    filename = f"{slug}.mp3"
    filepath = os.path.join(RESPONSE_DIR, filename)

    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="No se encontró audio para esta pregunta.")

    return FileResponse(filepath, media_type="audio/mpeg", filename=filename)

# === Endpoint: crear respuesta estática ===
@app.post("/static-response/", summary="Agrega una respuesta estática asociada a una palabra clave")
def create_static_response(data: StaticResponseCreate):
    db = SessionLocal()
    exists = db.query(StaticResponse).filter(StaticResponse.keyword == data.keyword).first()
    if exists:
        db.close()
        raise HTTPException(status_code=400, detail="Keyword ya existe.")
    sr = StaticResponse(keyword=data.keyword, answer=data.answer)
    db.add(sr)
    db.commit()
    db.close()
    return {"ok": True, "keyword": data.keyword, "answer": data.answer}

# === Endpoint: devuelve el path del último audio generado ===
@app.get("/last-audio-file-path", summary="Devuelve el path del último audio generado")
def get_last_audio_path():
    db = SessionLocal()
    try:
        last_interaction = db.query(Interaction).order_by(Interaction.timestamp.desc()).first()
        if not last_interaction:
            raise HTTPException(status_code=404, detail="No hay interacciones registradas.")
        audio_url_path = f"/responses/{last_interaction.tts_audio_filename}"
        return JSONResponse(content={"audio_path": audio_url_path})
    finally:
        db.close()
