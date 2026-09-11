from fastapi import FastAPI

app = FastAPI(title="InterviewForensics API")


@app.get("/")
def home():
    return {
        "message": "InterviewForensics API is running!",
        "status": "success"
    }