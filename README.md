# IronsiteHacks

<!-- This repo indexes construction videos with NVIDIA Cosmos-Reason2 (video captioning) and Cosmos-Embed1 (video/text embeddings), stores vectors in Qdrant, and supports semantic search over clips. -->

## Prerequisites

- Docker
- NVIDIA GPU + drivers (required to run Cosmos NIM containers)
- `ffmpeg` (optional, for slicing longer videos into clips)

## Setup

### 1) Create a virtual environment and install dependencies
```bash
conda create -n ironsite python=3.11
pip install -r requirements.txt
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
