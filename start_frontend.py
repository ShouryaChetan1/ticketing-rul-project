import os
import sys
import uvicorn

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "frontend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
