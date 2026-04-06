import asyncio
import json
import logging
import os
import signal
import threading
import traceback
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict

# Suppress noisy third-party warnings before any imports
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Setup minimalist logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

import psutil  # type: ignore
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel  # type: ignore

try:
    import msvcrt
except ImportError:
    msvcrt = None

from backend.llm import LLMManager  # type: ignore
from backend.database import ChatDB  # type: ignore
from backend.rag import RAGManager  # type: ignore

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("acisstant")

# Managers
db = ChatDB()
llm_manager = LLMManager()
rag_manager = RAGManager()

# --- Hotkey Watcher ---
def terminal_watcher():
    """Listens for Ctrl+R in the terminal window to restart the AI."""
    try:
        if msvcrt is None:
            logger.info("[Watcher] msvcrt not available, hotkey watcher disabled.")
            return
        logger.info("[Watcher] Terminal watcher started (Press Ctrl+R to restart).")
        while True:
            if msvcrt.kbhit():  # type: ignore
                ch = msvcrt.getch()  # type: ignore
                if ch in (b'\x12', b'\x12\r'):
                    logger.warning("[Watcher] HOTKEY: Ctrl+R detected! Restarting ACIsstant...")
                    os._exit(3)
                    break
            import time
            time.sleep(0.1)
    except Exception as e:
        logger.error(f"[Watcher] Exception in terminal watcher: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[SYSTEM] FastAPI lifespan starting.")
    try:
        t = threading.Thread(target=terminal_watcher, daemon=True)
        t.start()
    except Exception as e:
        logger.error(f"[SYSTEM] Failed to start terminal watcher: {e}")
    try:
        yield
    finally:
        logger.info("[SYSTEM] FastAPI lifespan ending.")

# --- App Initialization ---
app = FastAPI(title="ACIsstant Local AI Engineering Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# --- Global Middleware ---
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        tb_str = traceback.format_exc()
        logger.error(f"[UNHANDLED ERROR] {exc}\nTraceback:\n{tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. See backend logs for details.",
                "error_type": type(exc).__name__,
                "trace": tb_str.splitlines()[-5:],
            },
        )

# --- Models ---
class ChatRequest(BaseModel):
    chat_id: str
    message: str
    language: Optional[str] = "en-US"

# --- Endpoints ---
@app.get("/")
async def read_index():
    response = FileResponse("frontend/index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/api/health")
async def healthcheck():
    return {"status": "ok", "message": "ACIsstant backend is running."}

@app.get("/api/status")
async def status():
    import platform
    try:
        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(".")
        return {
            "status": "ok",
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "cpu_percent": cpu,
            "memory": {"total": mem.total, "available": mem.available, "percent": mem.percent},
            "disk": {"total": disk.total, "used": disk.used, "percent": disk.percent},
            "model_loaded": llm_manager.is_loaded(),
            "vector_store_loaded": rag_manager.vector_store is not None,
        }
    except Exception as e:
        logger.error(f"[STATUS] Failed to get status: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.post("/api/restart")
async def restart_server():
    logger.warning("API: Restart signal received.")
    os._exit(3)

@app.get("/api/chats")
async def get_chats():
    try:
        return db.get_chats()
    except Exception as e:
        logger.error(f"Failed to list chats: {e}")
        raise HTTPException(status_code=500, detail="Failed to list chats.")

@app.post("/api/chats")
async def create_chat(title: str = Form("New Chat")):
    try:
        chat_id = str(uuid.uuid4())
        db.create_chat(chat_id, title)
        return {"chat_id": chat_id, "title": title}
    except Exception as e:
        logger.error(f"Failed to create chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to create chat.")

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    try:
        return db.get_messages(chat_id)
    except Exception as e:
        logger.error(f"Failed to list messages for chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list messages.")

@app.put("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, title: str = Form(...)):
    try:
        db.update_chat_title(chat_id, title)
        return {"status": "renamed", "title": title}
    except Exception as e:
        logger.error(f"Failed to rename chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rename chat.")

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        db.delete_chat(chat_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat.")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    chat_id = request.chat_id
    user_msg = request.message
    
    try:
        db.add_message(chat_id, "user", user_msg)
        past_messages = db.get_messages(chat_id)
    except Exception as e:
        logger.error(f"Database error during streaming setup: {e}")
        raise HTTPException(status_code=500, detail="Database access error.")

    upload_dir = Path("data/uploads")
    all_files = [f.name for f in upload_dir.rglob("*") if f.is_file() and f.name != "README.md"]
    
    if request.language == "pt-PT":
        inventory_msg = f"Tens {len(all_files)} ficheiros disponíveis na tua base de dados: " + ", ".join(all_files) if all_files else "Não tens ficheiros carregados na base de dados."
    else:
        inventory_msg = f"You have {len(all_files)} files available in your knowledge base: " + ", ".join(all_files) if all_files else "No files are currently in your knowledge base."

    context, sources = rag_manager.query(user_msg, k=4)

    raw_system = llm_manager.get_system_prompt(request.language)
    inventory_directive = f"[SYSTEM DIRECTIVE] {inventory_msg}\nYOU MUST USE THESE FILES. YOU HAVE ACCESS TO THEM RIGHT NOW."
    
    llm_messages = [{"role": "system", "content": f"{raw_system}\n\n{inventory_directive}"}]
    
    if context:
        ctx_header = "Relevant context from YOUR LOCAL FILES:" if request.language != "pt-PT" else "Contexto relevante dos TEUS FICHEIROS LOCAIS:"
        llm_messages.append({"role": "system", "content": f"{ctx_header}\n{context}"})
        
    llm_messages.extend(past_messages[-8:])

    async def event_generator():
        full_response = ""
        queue: asyncio.Queue = asyncio.Queue()
        # get_running_loop() is correct in Python 3.10+ inside an async context
        loop = asyncio.get_running_loop()
        _DONE = object()  # sentinel

        def run_llm():
            """Run the synchronous LLM generator in a background thread."""
            try:
                for token in llm_manager.generate_stream(llm_messages):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as e:
                logger.error(f"LLM Stream Error (thread): {e}")
                loop.call_soon_threadsafe(queue.put_nowait, f"\nERROR: {str(e)}")
            finally:
                # Always signal completion so the async generator doesn't hang
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        # Kick off LLM in a thread — we intentionally don't await the future
        # because the queue/sentinel pattern below handles synchronisation.
        loop.run_in_executor(None, run_llm)

        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                full_response += item
                yield item
            if sources:
                src_chunk = f"\n\nSOURCES: {', '.join(sources)}"
                full_response += src_chunk
                yield src_chunk
        except asyncio.CancelledError:
            logger.info("[Stream] Client disconnected mid-stream.")
            raise
        finally:
            try:
                db.add_message(chat_id, "assistant", full_response)
            except Exception as e:
                logger.error(f"Failed to store assistant message: {e}")

    headers = {
        # Prevent nginx / any reverse proxy from buffering the stream
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(event_generator(), media_type="text/plain", headers=headers)

@app.post("/api/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    import shutil
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    for f in files:
        file_path = upload_dir / f.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_files.append(f.filename)
        except Exception as e:
            logger.error(f"Failed to save file {f.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {f.filename}")
            
    try:
        rag_manager.process_documents()
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")
        
    return {"status": "success", "files": saved_files}

@app.get("/api/files")
async def list_uploaded_files():
    upload_dir = Path("data/uploads")
    if not upload_dir.exists():
        return []
        
    files = []
    for f in upload_dir.rglob("*"):
        if f.is_file() and f.name != "README.md":
            files.append({
                "name": f.name,
                "path": str(f.relative_to(upload_dir)).replace("\\", "/"),
                "extension": f.suffix.lower().replace(".", "")
            })
    return files

@app.get("/api/files/download/{filename:path}")
async def download_file(filename: str):
    file_path = Path("data/uploads") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.post("/api/rag/index")
async def trigger_index():
    try:
        rag_manager.process_documents()
        return {"status": "success", "message": "Documents indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _calculate_token_usage(chat_id: str) -> dict:
    messages = db.get_messages(chat_id)
    n_ctx = llm_manager.llm.n_ctx() if llm_manager.llm else 4096
    
    system_prompt = llm_manager.get_system_prompt("en-US")
    all_messages = [{"role": "system", "content": system_prompt}]
    all_messages.extend(messages[-8:])
    
    total_tokens = 0
    if llm_manager.llm:
        prompt = llm_manager.format_prompt(all_messages)
        try:
            tokens = llm_manager.llm.tokenize(prompt.encode("utf-8"))
            total_tokens = len(tokens)
        except Exception:
            total_tokens = sum(len(m["content"]) for m in all_messages) // 4
    else:
        total_tokens = sum(len(m["content"]) for m in all_messages) // 4
        
    percentage = round(float(total_tokens) / float(n_ctx) * 100, 1) if n_ctx > 0 else 0.0
    return {
        "used_tokens": total_tokens,
        "max_tokens": n_ctx,
        "percentage": min(percentage, 100),
        "message_count": len(messages),
        "is_compressed": any(
             m.get("role") == "system" and "COMPRESSED_SUMMARY" in m.get("content", "")
             for m in messages
        )
    }

@app.get("/api/token-usage/{chat_id}")
async def get_token_usage_endpoint(chat_id: str):
    try:
        return _calculate_token_usage(chat_id)
    except Exception as e:
        logger.error(f"Failed to get token usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to get token usage.")

@app.post("/api/chat/compress/{chat_id}")
async def compress_chat(chat_id: str):
    try:
        messages = db.get_messages(chat_id)
        if len(messages) <= 6:
            return {"status": "skip", "reason": "Not enough messages to compress"}
            
        old_messages = messages[:-4]
        summary_parts = []
        for msg in old_messages:
            role_label = "User" if msg["role"] == "user" else "AI"
            content = msg["content"][:200].strip()
            summary_parts.append(f"[{role_label}]: {content}")
            
        compressed_summary = "[COMPRESSED_SUMMARY] Previous context:\n" + "\n".join(summary_parts)
        db.compress_messages(chat_id, len(old_messages), compressed_summary)
        return {"status": "compressed", "removed": len(old_messages), "summary_length": len(compressed_summary)}
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compress chat.")

@app.post("/api/chat/purge/{chat_id}")
async def purge_chat(chat_id: str):
    try:
        messages = db.get_messages(chat_id)
        if len(messages) <= 4:
            return {"status": "skip", "reason": "Too few messages to purge"}
        purge_count = max(1, len(messages) // 10)
        db.purge_oldest_messages(chat_id, purge_count)
        return {"status": "purged", "removed": purge_count, "remaining": len(messages) - purge_count}
    except Exception as e:
        logger.error(f"Purge failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to purge chat.")

import atexit
def on_shutdown():
    logger.info("[SYSTEM] Backend shutting down.")
atexit.register(on_shutdown)

if __name__ == "__main__":
    import uvicorn
    # Minimalist startup, clean logging
    logger.info("[SYSTEM] Starting Uvicorn Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
