# ChessHacks Bot - Deployment Repository

This is the deployment repository for the ChessHacks MCTS bot.

## Structure

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
