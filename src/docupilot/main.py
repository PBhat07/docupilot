
from fastapi import FastAPI

app = FastAPI(
    title="DocuPilot API",
    description="An evidence-grounded technical documentation assistant.",
    version="0.1.0",
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
