# ChessHacks Bot - Deployment Repository

This is the deployment repository for the ChessHacks MCTS bot.

## Structure
- `src/` - Bot implementation
- `serve.py` - FastAPI server
- `requirements.txt` - Python dependencies
- `MCTS/` - C++ MCTS engine source
- `MCZeroV1.pt` - Neural network model (downloaded from Hugging Face Hub)

## Deployment

This repository is configured for ChessHacks deployment:

1. Push to GitHub
2. Connect in ChessHacks dashboard
3. The bot will automatically:
   - Compile the C++ MCTS engine at runtime
   - Download LibTorch if needed
   - Run your bot

### For Competition Organizers (Modal/Deployment Platform)

**Minimum Required:**
- `build-essential`: Contains `g++`, `make`, and other C++ build tools (REQUIRED)
  - Install via: `apt-get update && apt-get install -y build-essential`

**What's NOT needed:**
- `wget` or `curl`: Bot uses Python's built-in `urllib` to download LibTorch
- `unzip`: Bot uses Python's built-in `zipfile` to extract LibTorch

**Optional (for faster startup):**
If you want to pre-install LibTorch to avoid runtime download (~200MB, takes a few minutes):
```bash
# Download and install LibTorch (CPU version, ~200MB)
python3 -c "import urllib.request; urllib.request.urlretrieve('https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip', '/tmp/libtorch.zip')"
python3 -c "import zipfile; zipfile.ZipFile('/tmp/libtorch.zip').extractall('/tmp')"
mv /tmp/libtorch /opt/libtorch
rm /tmp/libtorch.zip
export LIBTORCH_PATH=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

**Important Notes:** 
- LibTorch cannot be installed via `apt-get` - it must be downloaded from PyTorch's website
- If LibTorch is not pre-installed, the bot will automatically download it at runtime using Python (slower first startup, but works)
- **Model Download**: The model file `MCZeroV1.pt` is automatically downloaded from Hugging Face Hub (`Hiyo1256/chess-mcts-models`) at runtime if not found locally.

## Local Testing

Run the bot locally:

```bash
python serve.py
```

## Model File

The model file (`MCZeroV1.pt`) is automatically downloaded from Hugging Face Hub at runtime:
- Repository: `Hiyo1256/chess-mcts-models`
- Filename: `MCZeroV1.pt`

The bot will download the model automatically if it's not found locally. No manual setup required!
