# ChessHacks Bot - Deployment Repository

This is the deployment repository for the ChessHacks MCTS bot.

## Structure
lol 
- `src/` - Bot implementation
- `serve.py` - FastAPI server
- `requirements.txt` - Python dependencies
- `MCTS/` - C++ MCTS engine source (built in Docker)
- `Dockerfile` - Container build configuration
- `chessnet_new_ts.pt` - Neural network model (via Git LFS)

## Deployment

This repository is configured for ChessHacks deployment:

1. Push to GitHub
2. Connect in ChessHacks dashboard
3. ChessHacks will automatically:
   - Build the Docker container
   - Compile the C++ MCTS engine
   - Run your bot

### For Competition Organizers (Modal/Deployment Platform)

**Minimum Required:**
```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
```

**What's needed:**
- `build-essential`: Contains `g++`, `make`, and other C++ build tools (REQUIRED)

**What's NOT needed:**
- `wget` or `curl`: Bot uses Python's built-in `urllib` to download LibTorch
- `unzip`: Bot uses Python's built-in `zipfile` to extract LibTorch

**Optional (for faster startup):**
If you want to pre-install LibTorch to avoid runtime download (~200MB, takes a few minutes):
```dockerfile
# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download and install LibTorch (CPU version, ~200MB)
# Note: You can use Python to download if wget isn't available
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip', '/tmp/libtorch.zip')" \
    && python3 -c "import zipfile; zipfile.ZipFile('/tmp/libtorch.zip').extractall('/tmp')" \
    && mv /tmp/libtorch /opt/libtorch \
    && rm /tmp/libtorch.zip

ENV LIBTORCH_PATH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

**Important Notes:** 
- LibTorch cannot be installed via `apt-get` - it must be downloaded from PyTorch's website
- If LibTorch is not pre-installed, the bot will automatically download it at runtime using Python (slower first startup, but works)
- **Git LFS Required**: The model file `MCZeroV1.pt` is stored in Git LFS. The deployment system MUST run `git lfs pull` after cloning to get the actual model file (not just the pointer). If Git LFS is not configured, the bot will detect this and fall back to random moves.

## Local Testing

If you have Docker installed:

```bash
docker build -t chesshacks-bot .
docker run -p 5058:5058 chesshacks-bot
```

## Model File

The model file (`chessnet_new_ts.pt`) should be tracked with Git LFS:

```bash
git lfs install
git lfs track "chessnet_new_ts.pt"
git add .gitattributes chessnet_new_ts.pt
```

If the file is too large, configure the Dockerfile to download it during build.
