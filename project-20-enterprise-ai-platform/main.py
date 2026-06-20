from fastapi import FastAPI

app = FastAPI(
    title="Enterprise AI Platform"
)

@app.get("/")
def home():

    return {
        "platform":
        "Enterprise AI Platform",
        "status":
        "Running"
    }

@app.get("/health")
def health():

    return {
        "status":
        "Healthy"
    }
