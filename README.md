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

## 🎴 Creating Card Templates

VisionPanjabi needs templates for all 52 playing cards. You have two options:

### Option 1: Screenshot-Based Extraction (Recommended)

Extract templates automatically from a folder of poker screenshots:

```bash
python src/main.py --extract-templates screenshots/
```

Workflow:
1. Capture clear screenshots of your tables.
2. Place them in a folder (e.g., `screenshots/`).
3. Run the command above.
4. Label any cards the system could not auto-identify.

Best results:
- Use crisp screenshots with the full card visible.
- Keep resolutions consistent across images.
- Avoid heavy compression or motion blur.

Templates are saved to `templates/` and can be validated with `python src/validate_templates.py`.

### Option 2: Automated Extraction from Video

Extract templates automatically from your poker video:

```bash
python src/smart_extract.py "video/your-poker-game.mp4"
```

The script will:
1. Scan your video for cards
2. Find all unique cards
3. Show you each card to label
4. Save templates automatically

Typical extraction time: 5-10 minutes for 40-50 cards.

### Option 3: Manual Creation

Create templates manually by cropping card corners:
1. Pause video on clear card
2. Crop top-left corner (rank + suit symbol)
3. Save as `{rank}{suit}.png` (e.g., `Kc.png`, `As.png`)
4. Template should be ~60×90 pixels

See `docs/TEMPLATE_GUIDE.md` for detailed instructions.

### Validate Templates

Check your templates:

```bash
python src/validate_templates.py
```

## Local Browser Console

```bash
python launch_localhost.py
```

Open your browser devtools console to see the local log message.
