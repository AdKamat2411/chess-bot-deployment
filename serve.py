
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os
from datetime import datetime

from src.utils import chess_manager
from src import main

# Logging setup
_logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_logs_dir, exist_ok=True)
_server_log_file = os.path.join(_logs_dir, f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def _log_server(message: str):
    """Write server message to log file."""
    try:
        with open(_server_log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Failed to write to server log: {e}")

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn)
    _log_server(f"Move request - PGN: {pgn}, Time left: {timeleft}ms")

    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
        
        # Log successful move
        _log_server(f"Move generated: {move.uci()}, Time: {time_taken:.2f}ms")
        if logs:
            _log_server(f"Bot logs:\n{logs}")
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        error_msg = f"Bot raised exception: {str(e)}"
        _log_server(error_msg)
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # Confirm type of move_probs
    if not isinstance(move_probs, dict):
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(content={m: None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
