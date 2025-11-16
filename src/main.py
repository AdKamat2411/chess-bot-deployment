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
if os.path.exists(os.path.join(_possible_root1, "MCZeroV1.pt")) or os.path.exists(os.path.join(_possible_root1, "requirements.txt")):
    _project_root = _possible_root1
    print(f"[PATH DEBUG] Using separate repo structure, root: {_project_root}")
elif os.path.exists(os.path.join(_possible_root2, "MCZeroV1.pt")) or os.path.exists(os.path.join(_possible_root2, "requirements.txt")):
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

MODEL_PATH = os.path.join(_project_root, "MCZeroV1.pt")

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
        hf_filename = "MCZeroV1.pt"
        
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

def encode_board(board: Board) -> torch.Tensor:
    """
    Encode chess board to 18-channel 8x8 tensor.
    Channels 0-11: Piece planes (white pawn, knight, bishop, rook, queen, king, then black pieces)
    Channel 12: Side to move (1.0 for white, 0.0 for black) - full 8x8 plane
    Channels 13-16: Castling rights (white kingside, white queenside, black kingside, black queenside)
    Channel 17: En passant target square
    Returns: torch.Tensor of shape (1, 18, 8, 8)
    """
    # Initialize tensor: (18 channels, 8 rows, 8 cols)
    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
    
    # Channels 0-11: Pieces
    piece_channels = {
        ('P', True): 0,   # White pawn
        ('N', True): 1,   # White knight
        ('B', True): 2,   # White bishop
        ('R', True): 3,   # White rook
        ('Q', True): 4,   # White queen
        ('K', True): 5,   # White king
        ('p', False): 6,  # Black pawn
        ('n', False): 7,  # Black knight
        ('b', False): 8,  # Black bishop
        ('r', False): 9,  # Black rook
        ('q', False): 10, # Black queen
        ('k', False): 11, # Black king
    }
    
    # Encode pieces
    for rank in range(8):
        for file in range(8):
            square = Square(file + rank * 8)
            piece = board.piece_at(square)
            
            if piece is not None:
                piece_symbol = piece.symbol()
                is_white = piece.color
                channel = piece_channels.get((piece_symbol, is_white))
                if channel is not None:
                    tensor[channel, rank, file] = 1.0
    
    # Channel 12: Side to move (1.0 for white, 0.0 for black)
    side_to_move_value = 1.0 if board.turn else 0.0
    tensor[12, :, :] = side_to_move_value
    
    # Channels 13-16: Castling rights
    if board.has_kingside_castling_rights(True):   # White kingside
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(True): # White queenside
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(False):  # Black kingside
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(False): # Black queenside
        tensor[16, :, :] = 1.0
    
    # Channel 17: En passant target square
    if board.ep_square is not None:
        ep_rank = board.ep_square // 8
        ep_file = board.ep_square % 8
        tensor[17, ep_rank, ep_file] = 1.0
    
    # Add batch dimension: (1, 18, 8, 8)
    return tensor.unsqueeze(0)

# ============================================================================
# POLICY TO MOVE MAPPING
# ============================================================================

def policy_index_to_move(policy_idx: int) -> tuple:
    """
    Convert policy index (0-4095) to (from_square, to_square).
    Policy index = from_square * 64 + to_square
    """
    from_square = policy_idx // 64
    to_square = policy_idx % 64
    return (from_square, to_square)

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

def run_inference(board: Board) -> tuple:
    """
    Run model inference on board position.
    Returns: (policy_tensor, value)
    - policy_tensor: torch.Tensor of shape (4096,) with policy logits
    - value: float in [-1, 1] range
    """
    global _model
    
    assert MODEL_READY, "Model not ready — initialization failed"
    assert _model is not None, "Model not loaded"
    
    # Encode board to tensor
    board_tensor = encode_board(board)
    
    # Run inference
    with torch.no_grad():
        outputs = _model(board_tensor)
        
        # Model outputs: (policy_logits, value)
        if isinstance(outputs, tuple):
            policy_logits = outputs[0]  # Shape: (1, 4096)
            value = outputs[1].item()   # Shape: (1,)
        else:
            # Handle case where model returns single tensor
            policy_logits = outputs[0] if len(outputs) > 0 else outputs
            value = 0.0
    
    # Remove batch dimension from policy: (4096,)
    if policy_logits.dim() > 1:
        policy_logits = policy_logits.squeeze(0)
    
    return policy_logits, value

# ============================================================================
# MOVE GENERATION
# ============================================================================

@chess_manager.entrypoint
def mcts_move(ctx: GameContext):
    """
    Generate a move using direct neural network inference.
    Gets policy vector of 4096 dimensions and returns legal move with highest probability.
    """
    assert MODEL_READY, "Model not ready — initialization failed"
    
    # Get current board
    board = ctx.board
    
    # Check if we have legal moves
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Debug: Print move request
    side_to_move = "White" if board.turn else "Black"
    log_msg = f"=== Move Request ===\nSide to move: {side_to_move}\nFEN: {board.fen()}"
    print(log_msg)
    _log_to_file(log_msg)
    
    try:
        # Time the inference
        import time
        move_start_time = time.perf_counter()
        
        # Run inference
        policy_logits, value = run_inference(board)
        
        # Apply softmax to get probabilities
        policy_probs = torch.softmax(policy_logits, dim=0)
        
        # Map legal moves to their policy indices and get probabilities
        move_probs_dict = {}
        for move in legal_moves:
            policy_idx = move_to_policy_index(move)
            prob = policy_probs[policy_idx].item()
            move_probs_dict[move] = prob
        
        if not move_probs_dict:
            raise RuntimeError("No legal moves found in policy")
        
        # Find move with highest probability
        best_move = max(move_probs_dict, key=move_probs_dict.get)
        best_prob = move_probs_dict[best_move]
        
        move_end_time = time.perf_counter()
        move_time_ms = (move_end_time - move_start_time) * 1000  # Convert to milliseconds
        
        print(f"[MCTS] Policy inference complete. Best move: {best_move.uci()} (prob: {best_prob:.4f})")
        print(f"[MCTS] Position value: {value:.4f}")
        print(f"[MCTS] Move generation time: {move_time_ms:.2f}ms")
        _log_to_file(f"Selected move: {best_move.uci()} (prob: {best_prob:.4f}, value: {value:.4f}, time: {move_time_ms:.2f}ms)")
        
        # Normalize probabilities for logging
        total_prob = sum(move_probs_dict.values())
        normalized_probs = {m: p / total_prob for m, p in move_probs_dict.items()}
        
        ctx.logProbabilities(normalized_probs)
        
        return best_move
        
    except Exception as e:
        print(f"[MCTS] ERROR: Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
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
