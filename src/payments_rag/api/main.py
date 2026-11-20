from fastapi import FastAPI

app = FastAPI(title="Payments standards assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
