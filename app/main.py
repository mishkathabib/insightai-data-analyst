from fastapi import FastAPI

app = FastAPI(title="InsightAI API")


@app.get("/")
def home():
    return {
        "message": "Welcome to InsightAI!",
        "status": "Backend is running"
    }