#!/usr/bin/env python3
"""
Script to upload MCZeroV1.pt model to Hugging Face Hub

Usage:
1. Install: pip install huggingface_hub
2. Login: huggingface-cli login
3. Run: python upload_model_to_hf.py
"""

import os
from huggingface_hub import HfApi, login

# Configuration
MODEL_FILE = "MCZeroV1.pt"
REPO_ID = "AdKamat2411/chess-mcts-models"  # Change this to your username/repo-name
REPO_TYPE = "model"  # or "dataset" if you prefer

def upload_model():
    """Upload the model file to Hugging Face Hub"""
    
    # Check if model file exists
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file '{MODEL_FILE}' not found!")
        print(f"Make sure you're running this from the repo root directory.")
        return False
    
    file_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)  # Size in MB
    print(f"Found model file: {MODEL_FILE} ({file_size:.2f} MB)")
    
    # Login to Hugging Face (if not already logged in)
    try:
        login()  # This will prompt for token if not already logged in
    except Exception as e:
        print(f"Note: {e}")
        print("You may need to run: huggingface-cli login")
    
    # Initialize API
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            exist_ok=True,  # Don't error if repo already exists
            private=False  # Set to True if you want a private repo
        )
        print(f"Repository '{REPO_ID}' is ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload the model file
    print(f"\nUploading {MODEL_FILE} to {REPO_ID}...")
    print("This may take a few minutes (~223MB)...")
    
    try:
        api.upload_file(
            path_or_fileobj=MODEL_FILE,
            path_in_repo=MODEL_FILE,  # Name in the repo
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"\n✅ Success! Model uploaded to: https://huggingface.co/{REPO_ID}")
        print(f"\nYou can now use this in your code:")
        print(f'  hf_repo_id = "{REPO_ID}"')
        print(f'  hf_filename = "{MODEL_FILE}"')
        return True
    except Exception as e:
        print(f"\n❌ Error uploading model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Upload MCZeroV1.pt to Hugging Face Hub")
    print("=" * 60)
    print(f"Repository: {REPO_ID}")
    print(f"Model file: {MODEL_FILE}")
    print("=" * 60)
    
    success = upload_model()
    
    if success:
        print("\n✅ Upload complete!")
        print("\nNext steps:")
        print("1. Update src/main.py line 375 with your repo ID")
        print("2. The bot will automatically download from Hugging Face if Git LFS fails")
    else:
        print("\n❌ Upload failed. Check the error messages above.")

