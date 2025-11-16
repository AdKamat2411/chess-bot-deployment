from .utils import chess_manager, GameContext
from chess import Move, Board
import subprocess
import os
import sys
import shutil
import urllib.request
import zipfile
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
    # Running locally or in non-Docker deployment - try to find project root
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
        os.path.join(_project_root, "mcts_bridge"),  # Root level (Docker-style)
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
            # Competition organizers can pre-install it in their Docker image to speed up startup
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

# MCTS parameters
MAX_ITERATIONS = 20000
MAX_SECONDS = 1
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
    
    # Check if model exists and is valid
    if not os.path.exists(MODEL_PATH):
        print(f"[MCTS] ERROR: Model not found at {MODEL_PATH}")
        print(f"[MCTS] Falling back to random move")
        # Fallback to random move if model doesn't exist
        import random
        move = random.choice(legal_moves)
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
        ctx.logProbabilities(move_probs)
        return move
    
    # Validate model file - check if it's a Git LFS pointer or corrupted
    # If it's a pointer or missing, try to download from Hugging Face
    model_needs_download = False
    try:
        model_size = os.path.getsize(MODEL_PATH)
        print(f"[MCTS] Model file size: {model_size} bytes ({model_size / (1024*1024):.2f} MB)")
        
        # Check if it's suspiciously small (Git LFS pointer files are ~130 bytes)
        if model_size < 1024:  # Less than 1KB is definitely wrong
            print(f"[MCTS] WARNING: Model file is too small ({model_size} bytes) - likely a Git LFS pointer file")
            model_needs_download = True
        
        # Check if it's a Git LFS pointer file (starts with "version https://git-lfs.github.com/spec/v1")
        if not model_needs_download:
            with open(MODEL_PATH, 'rb') as f:
                first_bytes = f.read(100)
                if first_bytes.startswith(b'version https://git-lfs.github.com/spec/v1'):
                    print(f"[MCTS] WARNING: Model file is a Git LFS pointer, not the actual model")
                    model_needs_download = True
    except Exception as e:
        print(f"[MCTS] WARNING: Could not check model file: {type(e).__name__}: {e}")
        model_needs_download = True
    
    # Try to download from Hugging Face if needed
    if model_needs_download:
        print(f"[MCTS] Attempting to download model from Hugging Face...")
        try:
            from huggingface_hub import hf_hub_download
            
            # Hugging Face repo for model download fallback
            hf_repo_id = os.environ.get("HF_MODEL_REPO", "Hiyo1256/chess-mcts-models")
            hf_filename = "MCZeroV1.pt"
            
            print(f"[MCTS] Downloading {hf_filename} from {hf_repo_id}...")
            downloaded_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=hf_filename,
                local_dir=os.path.dirname(MODEL_PATH),
                local_dir_use_symlinks=False
            )
            
            # Move to expected location if needed
            if downloaded_path != MODEL_PATH:
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)  # Remove pointer file
                shutil.move(downloaded_path, MODEL_PATH)
            
            print(f"[MCTS] Model downloaded successfully from Hugging Face!")
            # Re-check file size
            model_size = os.path.getsize(MODEL_PATH)
            print(f"[MCTS] Downloaded model size: {model_size} bytes ({model_size / (1024*1024):.2f} MB)")
        except ImportError:
            print(f"[MCTS] ERROR: huggingface_hub not installed. Install with: pip install huggingface_hub")
            print(f"[MCTS] Falling back to random move")
            import random
            move = random.choice(legal_moves)
            move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
            ctx.logProbabilities(move_probs)
            return move
        except Exception as e:
            print(f"[MCTS] ERROR: Failed to download model from Hugging Face: {type(e).__name__}: {e}")
            print(f"[MCTS] Falling back to random move")
            import random
            move = random.choice(legal_moves)
            move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
            ctx.logProbabilities(move_probs)
            return move
    
    # Validate the model file is a valid zip archive (PyTorch models are zip files)
    try:
        with zipfile.ZipFile(MODEL_PATH, 'r') as zip_ref:
            # Check if it has the expected structure
            file_list = zip_ref.namelist()
            if not file_list:
                print(f"[MCTS] WARNING: Model file appears to be an empty zip archive")
            else:
                print(f"[MCTS] Model file appears to be a valid zip archive with {len(file_list)} files")
    except zipfile.BadZipFile:
        print(f"[MCTS] WARNING: Model file is not a valid zip archive (PyTorch models are zip files)")
        print(f"[MCTS] WARNING: File may be corrupted, but will attempt to load anyway")
    except Exception as e:
        print(f"[MCTS] WARNING: Could not validate model file: {type(e).__name__}: {e}")
        print(f"[MCTS] Will attempt to use it anyway")
    
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
