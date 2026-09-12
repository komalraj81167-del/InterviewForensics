from fastapi import FastAPI, UploadFile, File
import os
import shutil
from backend.app.qa_segmentation import segment_interview
from backend.app.transcription import transcribe_audio


app = FastAPI(title="InterviewForensics API")


UPLOAD_DIR = "backend/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "InterviewForensics API is running!",
        "status": "success"
    }


@app.post("/upload")
async def upload_interview(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Interview file uploaded successfully!",
        "filename": file.filename,
        "file_path": file_path
    }


@app.post("/transcribe")
async def transcribe_interview(filename: str):

    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {
            "error": "File not found"
        }

    transcript = transcribe_audio(file_path)

    return {
        "filename": filename,
        "transcript": transcript
    } 
@app.post("/analyze-transcript")
async def analyze_transcript(filename: str):

    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {
            "error": "File not found"
        }

    transcript = transcribe_audio(file_path)

    segments = segment_interview(transcript)

    return {
        "filename": filename,
        "total_questions": len(segments),
        "questions_and_answers": segments
    }


     