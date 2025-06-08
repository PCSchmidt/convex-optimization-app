"""FastAPI application setup."""

from __future__ import annotations

from fastapi import FastAPI

from routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI()
    application.include_router(router)
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
