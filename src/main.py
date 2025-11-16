from .utils import chess_manager, GameContext
from chess import Move, Board, Square
import os
import sys
import shutil
import urllib.request
import zipfile
from datetime import datetime
import torch

# ============================================================================
# GLOBAL STATE
# ============================================================================

MODEL_READY = False
MODEL_PATH = None
_model = None
_project_root = None

# One-Step MCTS parameters
LAMBDA = 1.0  # Weight for value difference in PUCT formula

# ============================================================================
# PATH DETECTION
# ============================================================================

# Try to find project root
# Case 1: Separate deployment repo (MCTS/ is sibling of src/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_possible_root1 = os.path.dirname(_script_dir)  # Go up from src/ to repo root
# Case 2: Nested in ChessMirror (my-chesshacks-bot/src/)
_possible_root2 = os.path.dirname(os.path.dirname(_possible_root1))  # Go up to ChessMirror

# Debug: Print detected paths
print(f"[PATH DEBUG] Script dir: {_script_dir}")
print(f"[PATH DEBUG] Possible root 1: {_possible_root1}")
print(f"[PATH DEBUG] Possible root 2: {_possible_root2}")

# Check which structure we're in
if os.path.exists(os.path.join(_possible_root1, "MCZeroV3.pt")) or os.path.exists(os.path.join(_possible_root1, "requirements.txt")):
    _project_root = _possible_root1
    print(f"[PATH DEBUG] Using separate repo structure, root: {_project_root}")
elif os.path.exists(os.path.join(_possible_root2, "MCZeroV3.pt")) or os.path.exists(os.path.join(_possible_root2, "requirements.txt")):
    _project_root = _possible_root2
    print(f"[PATH DEBUG] Using nested structure, root: {_project_root}")
else:
    # Fallback: assume separate repo structure
    _project_root = _possible_root1
    print(f"[PATH DEBUG] Fallback to separate repo structure, root: {_project_root}")

# List directory contents for debugging
if os.path.exists(_project_root):
    try:
        contents = os.listdir(_project_root)
        print(f"[PATH DEBUG] Root directory contents: {contents}")
    except Exception as e:
        print(f"[PATH DEBUG] Could not list root directory: {e}")

MODEL_PATH = os.path.join(_project_root, "MCZeroV3.pt")

print(f"[PATH DEBUG] Final model path: {MODEL_PATH}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def ensure_model_ready():
    """
    Ensure the model file exists, is valid, and loaded into memory.
    Downloads from HuggingFace if needed. Runs exactly once at startup.
    """
    global MODEL_READY, MODEL_PATH, _model
    
    if MODEL_READY:
        return
    
    print("[INIT] Checking model...")
    
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[INIT] Model file not found at {MODEL_PATH}")
        _download_model_from_huggingface()
    else:
        # 2. Check file size - if too small (<1MB), download from HuggingFace
        try:
            model_size = os.path.getsize(MODEL_PATH)
            print(f"[INIT] Model file size: {model_size} bytes ({model_size / (1024*1024):.2f} MB)")
            
            if model_size < 1024 * 1024:  # Less than 1MB is definitely wrong
                print(f"[INIT] Model file is too small ({model_size} bytes) - downloading from HuggingFace")
                _download_model_from_huggingface()
        except Exception as e:
            print(f"[INIT] WARNING: Could not check model file: {type(e).__name__}: {e}")
            print(f"[INIT] Attempting to download from HuggingFace")
            _download_model_from_huggingface()
    
    # 3. Validate final size
    try:
        model_size = os.path.getsize(MODEL_PATH)
        if model_size < 1024 * 1024:
            raise RuntimeError(f"Model file is still too small after download: {model_size} bytes")
        print(f"[INIT] Model file validated: {model_size} bytes ({model_size / (1024*1024):.2f} MB)")
    except Exception as e:
        raise RuntimeError(f"Failed to validate model file: {type(e).__name__}: {e}")
    
    # 4. Try zip validation once (wrapped in try/except)
    try:
        with zipfile.ZipFile(MODEL_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if not file_list:
                print(f"[INIT] WARNING: Model file appears to be an empty zip archive")
            else:
                print(f"[INIT] Model file validated as zip archive with {len(file_list)} files")
    except zipfile.BadZipFile:
        print(f"[INIT] WARNING: Model file is not a valid zip archive (PyTorch models are zip files)")
        print(f"[INIT] WARNING: File may be corrupted, but will attempt to use anyway")
    except Exception as e:
        print(f"[INIT] WARNING: Could not validate model zip: {type(e).__name__}: {e}")
        print(f"[INIT] Will attempt to use model anyway")
    
    # 5. Load model into memory
    print("[INIT] Loading PyTorch model...")
    try:
        _model = torch.jit.load(MODEL_PATH, map_location='cpu')
        _model.eval()  # Set to evaluation mode
        print("[INIT] Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {type(e).__name__}: {e}")
    
    # 6. Mark as ready
    MODEL_READY = True
    print("[INIT] Model ready.")


def _download_model_from_huggingface():
    """Download model from HuggingFace Hub. Called only during initialization."""
    global MODEL_PATH
    
    print("[INIT] Downloading model from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        
        hf_repo_id = os.environ.get("HF_MODEL_REPO", "Hiyo1256/chess-mcts-models")
        hf_filename = "MCZeroV3.pt"
        
        print(f"[INIT] Downloading {hf_filename} from {hf_repo_id}...")
        downloaded_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            local_dir=os.path.dirname(MODEL_PATH),
            local_dir_use_symlinks=False
        )
        
        # Move to expected location if needed
        if downloaded_path != MODEL_PATH:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)  # Remove old/corrupted file
            shutil.move(downloaded_path, MODEL_PATH)
        
        print(f"[INIT] Model downloaded successfully from HuggingFace!")
        model_size = os.path.getsize(MODEL_PATH)
        print(f"[INIT] Downloaded model size: {model_size} bytes ({model_size / (1024*1024):.2f} MB)")
        
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Install with: pip install huggingface_hub")
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HuggingFace: {type(e).__name__}: {e}")

# Initialize model at module import
ensure_model_ready()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

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

# ============================================================================
# BOARD ENCODING
# ============================================================================

def board_to_tensor(board: Board) -> torch.Tensor:
    """Encode chess board to neural network input tensor."""
    tensor = torch.zeros(1, 18, 8, 8, dtype=torch.float32)
    piece_to_channel = {
        ('P', True): 0, ('N', True): 1, ('B', True): 2, ('R', True): 3,
        ('Q', True): 4, ('K', True): 5, ('P', False): 6, ('N', False): 7,
        ('B', False): 8, ('R', False): 9, ('Q', False): 10, ('K', False): 11
    }
    
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            r = square // 8
            c = square % 8
            ch = piece_to_channel[(piece.symbol().upper(), piece.color)]
            tensor[0, ch, r, c] = 1.0
    
    tensor[0, 12] = 1.0 if board.turn else 0.0
    tensor[0, 13] = 1.0 if board.has_kingside_castling_rights(True) else 0.0
    tensor[0, 14] = 1.0 if board.has_queenside_castling_rights(True) else 0.0
    tensor[0, 15] = 1.0 if board.has_kingside_castling_rights(False) else 0.0
    tensor[0, 16] = 1.0 if board.has_queenside_castling_rights(False) else 0.0
    
    if board.ep_square is not None:
        r = board.ep_square // 8
        c = board.ep_square % 8
        tensor[0, 17, r, c] = 1.0
    
    return tensor

# ============================================================================
# POLICY TO MOVE MAPPING
# ============================================================================

def move_to_policy_index(move: Move) -> int:
    """
    Convert Move to policy index.
    Policy index = from_square * 64 + to_square
    """
    from_sq = move.from_square
    to_sq = move.to_square
    return from_sq * 64 + to_sq

# ============================================================================
# INFERENCE
# ============================================================================

def predict_position(board: Board, model: torch.jit.ScriptModule):
    """Run neural network inference on a position. Returns (policy_dict, value)."""
    input_tensor = board_to_tensor(board)
    with torch.no_grad():
        policy_logits, value_tensor = model(input_tensor)
        value = float(value_tensor.item())
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, value
        
        import torch.nn.functional as F
        policy_probs = F.softmax(policy_logits.view(-1), dim=0)
        move_probs = {}
        total = 0.0
        
        for m in legal_moves:
            idx = move_to_policy_index(m)
            if 0 <= idx < len(policy_probs):
                p = float(policy_probs[idx])
                move_probs[m.uci()] = p
                total += p
        
        if total > 0:
            for k in move_probs:
                move_probs[k] /= total
        else:
            u = 1.0 / len(legal_moves)
            move_probs = {m.uci(): u for m in legal_moves}
    
    return move_probs, value

def one_step_mcts(board: Board, model: torch.jit.ScriptModule, lambda_weight=LAMBDA):
    """One-step MCTS: evaluate current and child positions, select best move."""
    current_policy, current_value = predict_position(board, model)
    if not current_policy:
        raise ValueError("No legal moves available")
    
    move_scores = {}

    sorted_moves = sorted(current_policy.items(), key=lambda x: x[1], reverse=True)

    K = 4
    moves_to_evaluate = sorted_moves[:K]
    
    for move_uci, prob in moves_to_evaluate:
        move = Move.from_uci(move_uci)
        b2 = board.copy()
        b2.push(move)
        _, child_value = predict_position(b2, model)
        child_value = -child_value
        diff = child_value - current_value
        score = prob + lambda_weight * diff
        move_scores[move_uci] = score
    
    return max(move_scores.items(), key=lambda x: x[1])[0]

# ============================================================================
# MOVE GENERATION
# ============================================================================

@chess_manager.entrypoint
def mcts_move(ctx: GameContext):
    """Generate a move using One-Step MCTS with neural network."""
    fen = ctx.board.fen()
    legal = list(ctx.board.generate_legal_moves())
    if not legal:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    try:
        model = _model
        if model is None:
            raise FileNotFoundError("Model not loaded")
        
        best = one_step_mcts(ctx.board, model)
        move = Move.from_uci(best)
        if move not in legal:
            raise ValueError("Illegal move selected")
        
        policy_dict, _ = predict_position(ctx.board, model)
        move_probs = {m: policy_dict.get(m.uci(), 0.0) for m in legal}
        ctx.logProbabilities(move_probs)
        return move
        
    except FileNotFoundError:
        import random
        move = random.choice(legal)
        u = 1.0 / len(legal)
        ctx.logProbabilities({m: u for m in legal})
        return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins."""
    global _log_file
    _log_file = _get_log_file()
