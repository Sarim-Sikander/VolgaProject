from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.api import router
from app.core.config import config
from app.core.exceptions import APIException


def init_routers(app_: FastAPI) -> None:
    app_.include_router(router)


def register_exception_handlers(app_: FastAPI) -> None:
    @app_.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.msg, "detail": exc.detail},
        )


def create_app() -> FastAPI:
    app_ = FastAPI(
        title="Transcription Pipeline API",
        description="Audio to text with timestamps, plus Ollama downstream processing.",
        version="1.0.0",
        docs_url=None if config.ENVIRONMENT == "production" else "/docs",
        redoc_url=None if config.ENVIRONMENT == "production" else "/redoc",
    )

    app_.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_routers(app_=app_)
    register_exception_handlers(app_=app_)
    return app_


app = create_app()
