from .utils import chess_manager, GameContext
from chess import Move, Board
import subprocess
import os
import sys
import shutil
import urllib.request
import zipfile
from datetime import datetime

# ============================================================================
# GLOBAL STATE
# ============================================================================

MODEL_READY = False
MODEL_PATH = None
MCTS_BRIDGE_PATH = None
_project_root = None

# MCTS parameters
MAX_ITERATIONS = 20000
MAX_SECONDS = 1
CPUCT = 1.5

# ============================================================================
# PATH DETECTION AND BRIDGE SETUP
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
bridge_path1 = os.path.join(_possible_root1, "MCTS", "mcts_bridge")
bridge_path2 = os.path.join(_possible_root2, "MCTS", "mcts_bridge")

print(f"[PATH DEBUG] Checking bridge at: {bridge_path1} (exists: {os.path.exists(bridge_path1)})")
print(f"[PATH DEBUG] Checking bridge at: {bridge_path2} (exists: {os.path.exists(bridge_path2)})")

if os.path.exists(bridge_path1):
    # Separate deployment repo
    _project_root = _possible_root1
    print(f"[PATH DEBUG] Using separate repo structure, root: {_project_root}")
elif os.path.exists(bridge_path2):
    # Nested in ChessMirror
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

# Try multiple possible locations for the bridge
possible_bridge_paths = [
    os.path.join(_project_root, "MCTS", "mcts_bridge"),  # Standard location
    os.path.join(_project_root, "MCTS", "Chess", "mcts_bridge"),  # In Chess subdirectory
]

MCTS_BRIDGE_PATH = None
for bridge_path in possible_bridge_paths:
    if os.path.exists(bridge_path):
        MCTS_BRIDGE_PATH = bridge_path
        print(f"[PATH DEBUG] Found bridge at: {bridge_path}")
        break

if not MCTS_BRIDGE_PATH or not os.path.exists(MCTS_BRIDGE_PATH):
    # Bridge doesn't exist - try to build it using subprocess
    MCTS_BRIDGE_PATH = possible_bridge_paths[0]
    mcts_dir = os.path.join(_project_root, "MCTS")
    makefile_path = os.path.join(mcts_dir, "Makefile")
    
    print(f"[BUILD] Bridge not found, attempting to build it...")
    print(f"[BUILD] MCTS directory: {mcts_dir}")
    print(f"[BUILD] Target bridge path: {MCTS_BRIDGE_PATH}")
    
    # Check if we have the necessary files to build
    if os.path.exists(mcts_dir) and os.path.exists(makefile_path):
        libtorch_path = None
        libtorch_install_dir = "/opt/libtorch"
        
        # Step 1: Check if LibTorch exists, if not download it
        # Note: LibTorch cannot be installed via apt-get - it must be downloaded from PyTorch
        # Competition organizers can pre-install it to speed up startup
        libtorch_paths_to_check = [
            os.environ.get("LIBTORCH_PATH"),  # Check environment variable first
            libtorch_install_dir,  # /opt/libtorch (standard location)
            "/usr/local/libtorch",
            os.path.expanduser("~/libtorch"),
        ]
        
        for path in libtorch_paths_to_check:
            if path and os.path.exists(path) and os.path.isdir(path):
                libtorch_path = path
                print(f"[BUILD] Found pre-installed LibTorch at: {libtorch_path}")
                break
        
        if not libtorch_path:
            print(f"[BUILD] LibTorch not found, downloading using Python...")
            try:
                # Create install directory
                os.makedirs(os.path.dirname(libtorch_install_dir), exist_ok=True)
                
                # Download LibTorch CPU version using Python urllib
                libtorch_url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
                libtorch_zip = "/tmp/libtorch.zip"
                
                print(f"[BUILD] Downloading LibTorch from {libtorch_url}...")
                print(f"[BUILD] This may take a few minutes (~200MB download)...")
                
                # Download using urllib.request (built-in, no external dependencies)
                def show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
                    if block_num % 100 == 0:  # Print every 100 blocks to avoid spam
                        print(f"[BUILD] Download progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                
                try:
                    urllib.request.urlretrieve(libtorch_url, libtorch_zip, show_progress)
                    print(f"[BUILD] Download complete, extracting...")
                except Exception as e:
                    print(f"[BUILD] ERROR: Failed to download LibTorch: {type(e).__name__}: {e}")
                    raise
                
                # Extract using Python's zipfile module (built-in, no external dependencies)
                try:
                    with zipfile.ZipFile(libtorch_zip, 'r') as zip_ref:
                        print(f"[BUILD] Extracting LibTorch to /tmp...")
                        zip_ref.extractall("/tmp")
                    
                    # Move to final location
                    if os.path.exists("/tmp/libtorch"):
                        if os.path.exists(libtorch_install_dir):
                            shutil.rmtree(libtorch_install_dir)
                        os.rename("/tmp/libtorch", libtorch_install_dir)
                        libtorch_path = libtorch_install_dir
                        print(f"[BUILD] LibTorch installed at: {libtorch_path}")
                    else:
                        print(f"[BUILD] ERROR: LibTorch extraction failed - /tmp/libtorch not found")
                        # List what was extracted
                        try:
                            extracted = os.listdir("/tmp")
                            print(f"[BUILD] Contents of /tmp: {extracted}")
                        except:
                            pass
                except Exception as e:
                    print(f"[BUILD] ERROR: Failed to extract LibTorch: {type(e).__name__}: {e}")
                    raise
                
                # Clean up zip file
                try:
                    os.remove(libtorch_zip)
                except:
                    pass
                    
            except Exception as e:
                print(f"[BUILD] ERROR: Exception during LibTorch installation: {type(e).__name__}: {e}")
                import traceback
                print(f"[BUILD] Traceback: {traceback.format_exc()}")
        
        # Step 2: Check if build tools are available
        # Note: In Modal, build tools should be pre-installed by organizers
        build_tools_available = False
        try:
            subprocess.run(["which", "g++"], capture_output=True, check=True)
            subprocess.run(["which", "make"], capture_output=True, check=True)
            build_tools_available = True
            print(f"[BUILD] Build tools (g++, make) are available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"[BUILD] ERROR: Build tools (g++, make) not found")
            print(f"[BUILD] Organizers need to pre-install: build-essential")
            # Don't try to install via apt-get in Modal - it's not supported
        
        # Step 3: Build the bridge if we have everything
        if libtorch_path and build_tools_available:
            print(f"[BUILD] Attempting to build bridge with LIBTORCH_PATH={libtorch_path}")
            try:
                env = os.environ.copy()
                env["LIBTORCH_PATH"] = libtorch_path
                # Set LD_LIBRARY_PATH for runtime
                lib_path = os.path.join(libtorch_path, "lib")
                if "LD_LIBRARY_PATH" in env:
                    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
                else:
                    env["LD_LIBRARY_PATH"] = lib_path
                
                build_result = subprocess.run(
                    ["make", "Bridge"],
                    cwd=mcts_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for build
                )
                
                if build_result.returncode == 0:
                    # Check if the bridge was successfully built
                    if os.path.exists(MCTS_BRIDGE_PATH):
                        print(f"[BUILD] SUCCESS! Bridge built at: {MCTS_BRIDGE_PATH}")
                        # Make sure it's executable
                        os.chmod(MCTS_BRIDGE_PATH, 0o755)
                    else:
                        print(f"[BUILD] WARNING: Build completed but bridge not found at expected location")
                        if build_result.stdout:
                            print(f"[BUILD] Build stdout: {build_result.stdout}")
                        if build_result.stderr:
                            print(f"[BUILD] Build stderr: {build_result.stderr}")
                else:
                    print(f"[BUILD] Build failed with return code {build_result.returncode}")
                    if build_result.stdout:
                        print(f"[BUILD] Build stdout: {build_result.stdout}")
                    if build_result.stderr:
                        print(f"[BUILD] Build stderr: {build_result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"[BUILD] ERROR: Build timed out after 10 minutes")
            except Exception as e:
                print(f"[BUILD] ERROR: Exception during build: {type(e).__name__}: {e}")
        else:
            if not libtorch_path:
                print(f"[BUILD] ERROR: LibTorch not available - cannot build bridge")
            if not build_tools_available:
                print(f"[BUILD] ERROR: Build tools not available - cannot build bridge")
    else:
        print(f"[BUILD] ERROR: Cannot build - MCTS directory or Makefile not found")
    
    # Final check - did we successfully build it?
    if not os.path.exists(MCTS_BRIDGE_PATH):
        print(f"[BUILD] Bridge still not found after build attempt")
        # List MCTS directory contents for debugging
        if os.path.exists(mcts_dir):
            try:
                mcts_contents = os.listdir(mcts_dir)
                print(f"[PATH DEBUG] MCTS directory contents: {mcts_contents}")
                # Check if there are any executables
                for item in mcts_contents:
                    item_path = os.path.join(mcts_dir, item)
                    if os.path.isfile(item_path) and os.access(item_path, os.X_OK):
                        print(f"[PATH DEBUG] Found executable in MCTS/: {item}")
            except Exception as e:
                print(f"[PATH DEBUG] Could not list MCTS directory: {e}")

MODEL_PATH = os.path.join(_project_root, "MCZeroV1.pt")

print(f"[PATH DEBUG] Final bridge path: {MCTS_BRIDGE_PATH}")
print(f"[PATH DEBUG] Final model path: {MODEL_PATH}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def ensure_model_ready():
    """
    Ensure the model file exists, is valid, and ready to use.
    Downloads from HuggingFace if needed. Runs exactly once at startup.
    """
    global MODEL_READY, MODEL_PATH
    
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
    
    # 5. Mark as ready
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
# UTILITY FUNCTIONS
# ============================================================================

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


def run_bridge(fen: str) -> str:
    """
    Execute the MCTS bridge subprocess and return the UCI move string.
    """
    assert MODEL_READY, "Model not ready — initialization failed"
    assert os.path.exists(MCTS_BRIDGE_PATH), f"Bridge not found at {MCTS_BRIDGE_PATH}"
    
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
        _log_to_file("=== MCTS Bridge Debug Output ===\n" + result.stderr)
    
    # Parse output (should be UCI move string)
    uci_move = result.stdout.strip()
    
    if not uci_move:
        # Check stderr for error messages
        if result.stderr:
            error_msg = result.stderr
            raise RuntimeError(f"MCTS bridge error: {error_msg}")
        raise RuntimeError("MCTS bridge returned empty output")
    
    return uci_move

# ============================================================================
# MOVE GENERATION
# ============================================================================

@chess_manager.entrypoint
def mcts_move(ctx: GameContext):
    """
    Generate a move using MCTS + CNN.
    Calls the C++ MCTS engine via subprocess.
    """
    assert MODEL_READY, "Model not ready — initialization failed"
    
    # Get current board FEN
    fen = ctx.board.fen()
    
    # Check if we have legal moves
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Debug: Print move request
    side_to_move = "White" if ctx.board.turn else "Black"
    log_msg = f"=== Move Request ===\nSide to move: {side_to_move}\nFEN: {fen}"
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
    
    try:
        # Call MCTS bridge
        uci_move = run_bridge(fen)
        
        print(f"[MCTS] Bridge returned: {uci_move}")
        _log_to_file(f"Selected move: {uci_move}")
        
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
        error_msg = f"MCTS bridge timed out after {MAX_SECONDS + 10} seconds"
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
