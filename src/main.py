from .utils import chess_manager, GameContext
from chess import Move, Board
import subprocess
import os
import sys
from datetime import datetime

# Configuration
# Path to the MCTS bridge executable
# In Docker/deployment: /app/mcts_bridge
# In local dev: Try multiple possible locations
if os.path.exists("/app/mcts_bridge"):
    # Running in Docker/deployment
    MCTS_BRIDGE_PATH = "/app/mcts_bridge"
    MODEL_PATH = "/app/model.pt"
    _project_root = "/app"
else:
    # Running locally - try to find project root
    # Case 1: Separate deployment repo (MCTS/ is sibling of src/)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _possible_root1 = os.path.dirname(_script_dir)  # Go up from src/ to repo root
    # Case 2: Nested in ChessMirror (my-chesshacks-bot/src/)
    _possible_root2 = os.path.dirname(os.path.dirname(_possible_root1))  # Go up to ChessMirror
    
    # Check which structure we're in
    if os.path.exists(os.path.join(_possible_root1, "MCTS", "mcts_bridge")):
        # Separate deployment repo
        _project_root = _possible_root1
    elif os.path.exists(os.path.join(_possible_root2, "MCTS", "mcts_bridge")):
        # Nested in ChessMirror
        _project_root = _possible_root2
    else:
        # Fallback: assume separate repo structure
        _project_root = _possible_root1
    
    MCTS_BRIDGE_PATH = os.path.join(_project_root, "MCTS", "mcts_bridge")
    MODEL_PATH = os.path.join(_project_root, "MCZeroV1.pt")

# MCTS parameters
MAX_ITERATIONS = 20000
MAX_SECONDS = 2
CPUCT = 1.5

# Logging configuration
# Try to create logs directory in appropriate location
if os.path.exists(os.path.join(_project_root, "my-chesshacks-bot")):
    # Nested structure (ChessMirror/my-chesshacks-bot/)
    _logs_dir = os.path.join(_project_root, "my-chesshacks-bot", "logs")
else:
    # Separate deployment repo
    _logs_dir = os.path.join(_project_root, "logs")
os.makedirs(_logs_dir, exist_ok=True)

def _get_log_file():
    """Get log file path for current session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_logs_dir, f"mcts_bot_{timestamp}.log")

_log_file = _get_log_file()

def _log_to_file(message: str):
    """Write message to log file."""
    try:
        with open(_log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"[MCTS] Failed to write to log file: {e}")


def uci_to_move(board: Board, uci_str: str) -> Move:
    """Convert UCI string (e.g., 'e2e4') to python-chess Move object."""
    try:
        # python-chess handles UCI conversion automatically
        move = Move.from_uci(uci_str)
        
        # Verify it's legal
        if move not in board.legal_moves:
            raise ValueError(f"Move {uci_str} is not legal in current position")
        
        return move
    except ValueError as e:
        raise ValueError(f"Invalid UCI move format '{uci_str}': {e}")


@chess_manager.entrypoint
def mcts_move(ctx: GameContext):
    """
    Generate a move using MCTS + CNN.
    Calls the C++ MCTS engine via subprocess.
    """
    # Get current board FEN
    fen = ctx.board.fen()
    
    # Check if we have legal moves
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Debug: Print paths
    side_to_move = "White" if ctx.board.turn else "Black"
    log_msg = f"=== Move Request ===\nSide to move: {side_to_move}\nFEN: {fen}\nBridge: {MCTS_BRIDGE_PATH}\nModel: {MODEL_PATH}"
    print(log_msg)
    _log_to_file(log_msg)
    
    # Check if bridge executable exists
    if not os.path.exists(MCTS_BRIDGE_PATH):
        print(f"[MCTS] ERROR: Bridge not found at {MCTS_BRIDGE_PATH}")
        print(f"[MCTS] Falling back to random move")
        # Fallback to random move if bridge doesn't exist
        import random
        move = random.choice(legal_moves)
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
        ctx.logProbabilities(move_probs)
        return move
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[MCTS] ERROR: Model not found at {MODEL_PATH}")
        print(f"[MCTS] Falling back to random move")
        # Fallback to random move if model doesn't exist
        import random
        move = random.choice(legal_moves)
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
        ctx.logProbabilities(move_probs)
        return move
    
    try:
        print(f"[MCTS] Calling bridge with: {MCTS_BRIDGE_PATH} {MODEL_PATH} {fen[:50]}...")
        
        # Call MCTS bridge
        result = subprocess.run(
            [
                MCTS_BRIDGE_PATH,
                MODEL_PATH,
                fen,
                str(MAX_ITERATIONS),
                str(MAX_SECONDS),
                str(CPUCT)
            ],
            capture_output=True,
            text=True,
            timeout=MAX_SECONDS + 10,  # Add buffer for overhead
            check=True
        )
        
        # Save debug output to log file
        if result.stderr:
            print(f"[MCTS] Bridge stderr output saved to log")
            _log_to_file("=== MCTS Bridge Debug Output ===\n" + result.stderr)
        
        # Parse output (should be UCI move string)
        uci_move = result.stdout.strip()
        print(f"[MCTS] Bridge returned: {uci_move}")
        _log_to_file(f"Selected move: {uci_move}")
        
        if not uci_move:
            # Check stderr for error messages
            if result.stderr:
                error_msg = result.stderr
                print(f"[MCTS] ERROR: {error_msg}")
                raise RuntimeError(f"MCTS bridge error: {error_msg}")
            raise RuntimeError("MCTS bridge returned empty output")
        
        # Convert UCI to python-chess Move
        move = uci_to_move(ctx.board, uci_move)
        print(f"[MCTS] Selected move: {move.uci()}")
        
        # Get move probabilities from MCTS (we'll use uniform for now since
        # the bridge doesn't return probabilities - could enhance later)
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
        # Set the chosen move to have higher probability
        move_probs[move] = 0.5
        # Renormalize
        total = sum(move_probs.values())
        move_probs = {m: p / total for m, p in move_probs.items()}
        
        ctx.logProbabilities(move_probs)
        
        return move
        
    except subprocess.TimeoutExpired:
        error_msg = f"MCTS bridge timed out after {MAX_SECONDS + 5} seconds"
        print(f"[MCTS] ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        print(f"[MCTS] ERROR: Bridge failed with return code {e.returncode}")
        print(f"[MCTS] stderr: {e.stderr}")
        print(f"[MCTS] stdout: {e.stdout}")
        raise RuntimeError(f"MCTS bridge failed: {error_msg}")
    except ValueError as e:
        print(f"[MCTS] ERROR: Invalid move: {e}")
        raise ValueError(f"Invalid move from MCTS bridge: {e}")
    except Exception as e:
        print(f"[MCTS] ERROR: Unexpected error: {type(e).__name__}: {e}")
        raise


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    Can reset caches, model state, etc.
    """
    global _log_file
    _log_file = _get_log_file()
    side_to_move = "White" if ctx.board.turn else "Black"
    reset_msg = f"=== New Game Started ===\nFEN: {ctx.board.fen()}\nSide to move: {side_to_move}\nLog file: {_log_file}"
    print(reset_msg)
    _log_to_file(reset_msg)
