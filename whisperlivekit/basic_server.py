from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisperlivekit import WhisperLiveKit, parse_args
from whisperlivekit.audio_processor import AudioProcessor
from whisperlivekit.punctuation import PunctuationRestoreModel

import asyncio
import logging
import os, sys
import argparse
from opencc import OpenCC
cc = OpenCC('s2tw')

sys.argv = [
    "basic_server.py",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--model", "large-v3", # large-v3, turbo
    "--language", "zh"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

kit = None
model = None # Bert Model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kit, model
    kit = WhisperLiveKit()
    # 標點符號模型初始化
    model = PunctuationRestoreModel()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(kit.web_interface())


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            # 修改已經辨識內容之文字(黑色)
            for index, line in enumerate(response['lines']):
                print("Index:", index)
                response['lines'][index]['text'] = cc.convert(response['lines'][index]['text'])
                # 加上標點符號
                response['lines'][index]['text'], _ = model.restore_punctuation(response['lines'][index]['text'])

            # 修改預先辨識之文字(灰色)
            response["buffer_transcription"] = cc.convert(response["buffer_transcription"])
            await websocket.send_json(response)
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    audio_processor = AudioProcessor()

    await websocket.accept()
    logger.info("WebSocket connection opened.")
            
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        websocket_task.cancel()
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up.")

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    args = parse_args()
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }


    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    
    main()
