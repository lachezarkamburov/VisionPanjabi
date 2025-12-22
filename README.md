# gg-poker-vision-engine

A Dockerized poker strategy engine that ingests Twitch frames from GGPoker, uses OpenCV template matching for hero card recognition, and maps the resulting hand into an 8x8 strategy matrix.

## Features

- **Vision agent** powered by OpenCV + Streamlink for frame capture.
- **Template matching** for card recognition (drop GGPoker card templates into `templates/`).
- **Strategy engine** driven by an 8x8 matrix stored in `charts/strategy_matrix.json`.
- **Persistent storage** for strategy charts and local SQLite hand history data in `data/`.

## Repository Layout

```
./src
./templates
./charts
./data
```

## Setup

1. Add your GGPoker card templates (`.png`) to `templates/`.
2. (Optional) Update ROI coordinates in `src/main.py` to match your stream layout.

### Docker

```bash
docker-compose up --build
```

## Configuration Notes

- The vision agent uses template matching to achieve high-confidence card detection.
- The strategy engine checks the 8x8 matrix. If the matched hand is in a **Red** zone, it outputs **"4-Bet Bluff"**.

## Strategy Matrix Format

`charts/strategy_matrix.json` expects:

- `ranks`: ordered list of 8 ranks (rows/columns).
- `matrix`: 8x8 array of zone strings (`Red`, `Green`, etc.).

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```
