from .utils import chess_manager, GameContext
from chess import Move, Board
import subprocess
import os
import sys
import shutil
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
            libtorch_paths_to_check = [
                os.environ.get("LIBTORCH_PATH"),
                libtorch_install_dir,
                "/usr/local/libtorch",
                os.path.expanduser("~/libtorch"),
            ]
            
            for path in libtorch_paths_to_check:
                if path and os.path.exists(path) and os.path.isdir(path):
                    libtorch_path = path
                    print(f"[BUILD] Found LibTorch at: {libtorch_path}")
                    break
            
            if not libtorch_path:
                print(f"[BUILD] LibTorch not found, downloading...")
                try:
                    # Create install directory
                    os.makedirs(os.path.dirname(libtorch_install_dir), exist_ok=True)
                    
                    # Download LibTorch CPU version
                    libtorch_url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
                    libtorch_zip = "/tmp/libtorch.zip"
                    
                    print(f"[BUILD] Downloading LibTorch from {libtorch_url}...")
                    download_result = subprocess.run(
                        ["wget", "-O", libtorch_zip, libtorch_url],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout for download
                    )
                    
                    if download_result.returncode == 0:
                        print(f"[BUILD] Extracting LibTorch...")
                        # Extract LibTorch
                        extract_result = subprocess.run(
                            ["unzip", "-q", libtorch_zip, "-d", "/tmp"],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        
                        if extract_result.returncode == 0:
                            # Move to final location
                            if os.path.exists("/tmp/libtorch"):
                                if os.path.exists(libtorch_install_dir):
                                    shutil.rmtree(libtorch_install_dir)
                                os.rename("/tmp/libtorch", libtorch_install_dir)
                                libtorch_path = libtorch_install_dir
                                print(f"[BUILD] LibTorch installed at: {libtorch_path}")
                            else:
                                print(f"[BUILD] ERROR: LibTorch extraction failed - /tmp/libtorch not found")
                        else:
                            print(f"[BUILD] ERROR: Failed to extract LibTorch: {extract_result.stderr}")
                        
                        # Clean up zip file
                        try:
                            os.remove(libtorch_zip)
                        except:
                            pass
                    else:
                        print(f"[BUILD] ERROR: Failed to download LibTorch: {download_result.stderr}")
                        # Try curl as fallback
                        print(f"[BUILD] Trying curl as fallback...")
                        curl_result = subprocess.run(
                            ["curl", "-L", "-o", libtorch_zip, libtorch_url],
                            capture_output=True,
                            text=True,
                            timeout=600
                        )
                        if curl_result.returncode == 0:
                            extract_result = subprocess.run(
                                ["unzip", "-q", libtorch_zip, "-d", "/tmp"],
                                capture_output=True,
                                text=True,
                                timeout=300
                            )
                            if extract_result.returncode == 0 and os.path.exists("/tmp/libtorch"):
                                if os.path.exists(libtorch_install_dir):
                                    shutil.rmtree(libtorch_install_dir)
                                os.rename("/tmp/libtorch", libtorch_install_dir)
                                libtorch_path = libtorch_install_dir
                                print(f"[BUILD] LibTorch installed at: {libtorch_path}")
                        try:
                            os.remove(libtorch_zip)
                        except:
                            pass
                except subprocess.TimeoutExpired:
                    print(f"[BUILD] ERROR: Download/extraction timed out")
                except Exception as e:
                    print(f"[BUILD] ERROR: Exception during LibTorch installation: {type(e).__name__}: {e}")
            
            # Step 2: Check and install build tools if needed
            build_tools_available = False
            try:
                subprocess.run(["which", "g++"], capture_output=True, check=True)
                subprocess.run(["which", "make"], capture_output=True, check=True)
                build_tools_available = True
                print(f"[BUILD] Build tools (g++, make) are available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"[BUILD] Build tools not found, attempting to install...")
                try:
                    # Try to install build tools (this requires sudo/root, may fail)
                    install_result = subprocess.run(
                        ["apt-get", "update"],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if install_result.returncode == 0:
                        install_result = subprocess.run(
                            ["apt-get", "install", "-y", "build-essential", "wget", "unzip"],
                            capture_output=True,
                            text=True,
                            timeout=600
                        )
                        if install_result.returncode == 0:
                            build_tools_available = True
                            print(f"[BUILD] Build tools installed successfully")
                        else:
                            print(f"[BUILD] WARNING: Failed to install build tools: {install_result.stderr}")
                    else:
                        print(f"[BUILD] WARNING: Failed to update package list: {install_result.stderr}")
                except subprocess.TimeoutExpired:
                    print(f"[BUILD] WARNING: Build tools installation timed out")
                except Exception as e:
                    print(f"[BUILD] WARNING: Could not install build tools: {type(e).__name__}: {e}")
            
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
