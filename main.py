import uvicorn

from app.core.config import config

if __name__ == "__main__":
    uvicorn.run(
        app="app.core.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.ENVIRONMENT != "production",
        workers=1,
    )
