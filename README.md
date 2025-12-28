# poker-vision-engine

A Dockerized poker strategy engine that ingests YouTube frames from Poker, uses OpenCV template matching for hero card recognition, and maps the resulting hand into an 8x8 strategy matrix.

## Features

- **Vision agent** powered by OpenCV + Streamlink for frame capture.
- **Template matching** for card recognition (drop Poker card templates into `templates/`).
- **Strategy engine** driven by an 8x8 matrix stored in `charts/strategy_matrix.json`.
- **Persistent storage** for strategy charts and local SQLite hand history data in `data/`.

## Repository Layout

```
./src
./templates
./charts
./data
./config.yaml
```

## Setup

1. Add your Poker card templates (`.png`) to `templates/`.
2. Update `config.yaml` with your stream URL and ROI coordinates.
3. (Optional) Define manual table layouts if auto-detection is not reliable.
2. (Optional) Update ROI coordinates in `src/main.py` to match your stream layout.

### Docker

```bash
docker-compose up --build
```

## Configuration Notes

- The vision agent uses template matching to achieve high-confidence card detection.
- Multi-table detection defaults to `auto_detect: true` and can track up to 6 tables.
- The strategy engine checks the 8x8 matrix. If the matched hand is in a **Red** zone, it outputs **"4-Bet Bluff"**.

## Example Output

```json
{
  "table_1": {
    "cards": ["As", "Kh"],
    "stack_size": "140px",
    "dealer_button": false,
    "strategy": {
      "hand": "AK",
      "zone": "Red",
      "action": "4-Bet Bluff"
    }
  }
}
```

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

## Local Browser Console

```bash
python launch_localhost.py
```

Open your browser devtools console to see the local log message.
