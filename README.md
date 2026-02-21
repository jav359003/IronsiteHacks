# IronsiteHacks

<!-- This repo indexes construction videos with NVIDIA Cosmos-Reason2 (video captioning) and Cosmos-Embed1 (video/text embeddings), stores vectors in Qdrant, and supports semantic search over clips. -->

## Prerequisites
- Python 3.10+
- Docker + Docker Compose
- NVIDIA GPU + drivers (required to run Cosmos NIM containers)
- `ffmpeg` (optional, for slicing longer videos into clips)

## Setup
### 1) Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Set environment variables
Create a `.env` file (or export in your shell):
```bash
# Required for NIM containers
NGC_API_KEY=your_ngc_api_key

# Optional: if your NIM endpoints are secured
NIM_API_KEY=your_nim_api_key

# Optional overrides (defaults shown)
COSMOS_REASON2_BASE_URL=http://127.0.0.1:8000/v1
COSMOS_REASON2_MODEL=nvidia/cosmos-reason2-2b
COSMOS_EMBED1_BASE_URL=http://127.0.0.1:8001
COSMOS_EMBED1_MODEL=nvidia/cosmos-embed1
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=cosmos_embed1
VIDEO_DIR=/Users/damith/ironsite-rag/data/ironsite-trimmed
```

### 3) Start Qdrant + Cosmos NIM services
```bash
docker compose up -d
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
