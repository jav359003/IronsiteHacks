# Ironsite RAG (Cosmos + Qdrant)

This repo indexes construction videos with NVIDIA Cosmos-Reason2 (video captioning) and Cosmos-Embed1 (video/text embeddings), stores vectors in Qdrant, and supports semantic search over clips. The setup below uses Hugging Face model weights (no Docker).

## Prerequisites
- Python 3.10+
- NVIDIA GPU + drivers (recommended for Reason2/Embed1 performance)
- `ffmpeg` (optional, for slicing longer videos into clips)
- Hugging Face access approved for:
  - [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
  - [nvidia/Cosmos-Embed1-224p](https://huggingface.co/nvidia/Cosmos-Embed1-224p) or [nvidia/Cosmos-Embed1-336p](https://huggingface.co/nvidia/Cosmos-Embed1-336p)

## Setup
### 1) Create a virtual environment and install dependencies
```bash
conda create -n ironsite python=3.11
conda activate ironsite
pip install -r requirements.txt
```

### 2) Set environment variables
Create a `.env` file (or export in your shell):
```bash
# Model IDs (Hugging Face)
COSMOS_REASON2_MODEL=nvidia/Cosmos-Reason2-8B
COSMOS_EMBED1_MODEL=nvidia/Cosmos-Embed1-224p

# If the models are gated, either run `huggingface-cli login`
# or set a token:
HUGGINGFACE_HUB_TOKEN=your_hf_token

# Optional overrides
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=cosmos_embed1
VIDEO_DIR=/Users/damith/ironsite-rag/data/ironsite-trimmed
```

### 3) Start Qdrant
If you already have Qdrant running, skip this step. Otherwise, use Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest
```

## Usage
### Index videos
```bash
python scripts/index_videos.py --video-dir /path/to/videos
```

Options:
- `--clip-seconds` (default: 15)
- `--stride-seconds` (default: 30)
- `--limit` (default: 0 = all videos)
- `--prompt` (captioning prompt)

### Query the index
```bash
python scripts/query.py "worker tightening pipe fitting" --limit 5
```

## Notes
- If `ffmpeg` is not installed, videos are indexed without slicing.
- The first run will create the Qdrant collection based on the embedding size returned by Cosmos-Embed1.
- Cosmos-Embed1 variants differ in input resolution and embedding size (224p -> 256-dim, 336p -> 768-dim). Adjust `QDRANT_COLLECTION` if you change embedding size to avoid mixing vectors.
