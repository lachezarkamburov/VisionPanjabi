@codex I can see from the debug images that the detector IS finding contours around the cards, but the filtering logic is selecting player avatars and UI elements instead of the actual playing cards.

Problem 1: In 07_detected_cards.png, the green boxes are around player faces (0.97 confidence) and UI elements (0.50 confidence), NOT the playing cards visible in the original frame.
Problem 2: Even when templates are extracted, the output shows generic template names instead of actual card names.

Current output (WRONG):
```
🎴 Hero Cards: card_table_3_left | card_table_3_right
🎯 STRATEGY RECOMMENDATION:
   Hand: CC
   Zone: Unknown
   Action: Hold
```

Expected output (CORRECT):
```
🎴 Hero Cards: Ad | Ah
🎯 STRATEGY RECOMMENDATION:
   Hand: AA
   Zone: Green
   Action: All-in
```

Root causes:
- The current filtering prefers larger, more rectangular UI elements over the actual cards
- Card recognition (OCR) is not working, so cards keep generic filenames

Solutions needed:

Fix 1: Improve Card Detection Filtering

**Add Position-Based Filtering**
Cards are always in specific areas of the poker table (center-bottom area for hero cards). Add region-of-interest filtering:
```python
def detect_cards(self, frame: np.ndarray, debug: bool = False) -> List[Dict]:
    # ... existing detection code ...
    
    h, w = frame.shape[:2]
    
    # Define card regions (hero cards are typically in bottom 40% of frame, center area)
    card_region_y_min = int(h * 0.35)  # Cards appear in bottom 65% of frame
    card_region_y_max = int(h * 0.85)  # But not at very bottom (UI)
    card_region_x_min = int(w * 0.15)  # Not at edges
    card_region_x_max = int(w * 0.85)
    
    filtered_cards = []
    for card in detected_cards:
        card_center_y = card["y"] + card["height"] // 2
        card_center_x = card["x"] + card["width"] // 2
        
        # Check if card is in expected region
        in_card_region = (card_region_y_min <= card_center_y <= card_region_y_max and
                         card_region_x_min <= card_center_x <= card_region_x_max)
        
        if in_card_region:
            # Boost confidence for cards in correct region
            card["confidence"] *= 1.5
            filtered_cards.append(card)
        elif card["confidence"] > 0.95:  # Very high confidence, keep anyway
            filtered_cards.append(card)
    
    return filtered_cards
```

**Add Size-Based Filtering**
Playing cards have consistent absolute size:
```python
# In detect_cards, replace relative area check with absolute size check
# Cards should be roughly 40-100 pixels wide and 60-140 pixels tall
if (0.8 <= aspect_ratio <= 2.0 and
    40 <= w <= 100 and        # Absolute width range
    60 <= h <= 140 and        # Absolute height range
    w * h > 2000):            # Minimum area in pixels
```

**Exclude Non-Card Shapes**
Exclude circular/rounded shapes (player avatars):
```python
# After getting bounding rectangle, check if contour is rectangular
perimeter = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

# Cards have 4 corners, avatars are circular (many points)
if len(approx) < 4 or len(approx) > 6:
    continue  # Skip non-rectangular shapes
```

**Look for Card Pairs**
Cards appear in pairs (hero cards side-by-side). Boost confidence for paired rectangles:
```python
# After detecting all potential cards, look for horizontal pairs
for i, card1 in enumerate(detected_cards):
    for j, card2 in enumerate(detected_cards):
        if i >= j:
            continue
        
        # Check if horizontally adjacent
        y_diff = abs(card1["y"] - card2["y"])
        if card1["x"] < card2["x"]:
            x_gap = card2["x"] - (card1["x"] + card1["width"])
        else:
            x_gap = card1["x"] - (card2["x"] + card2["width"])
        
        # If cards are paired (same Y, small X gap)
        if y_diff < 20 and 5 <= x_gap <= 50:
            # Boost confidence for both
            card1["confidence"] *= 1.3
            card2["confidence"] *= 1.3
            card1["paired"] = True
            card2["paired"] = True
```

**Update Debug Visualization**
Add region overlay to debug images:
```python
# In debug visualization, draw the card search region
cv2.rectangle(debug_filtered, 
             (card_region_x_min, card_region_y_min),
             (card_region_x_max, card_region_y_max),
             (255, 0, 0), 2)  # Blue rectangle showing search area
```

Fix 2: Improve Card Recognition to Show Actual Card Names

The current OCR isn't working. Improve the fallback method to better recognize cards:

**Enhanced Rank Detection Without OCR**
```python
def _detect_rank_fallback(self, thresh_corner: np.ndarray) -> Optional[str]:
    """
    Improved fallback rank detection using pattern analysis.
    """
    # Get the top-left portion where rank appears (typically 25% of card)
    h, w = thresh_corner.shape
    rank_region = thresh_corner[0:int(h*0.3), 0:int(w*0.3)]
    
    # Count white pixels in specific patterns to identify ranks
    white_pixels = cv2.countNonZero(rank_region)
    total_pixels = rank_region.shape[0] * rank_region.shape[1]
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # Find contours to count distinct shapes
    contours, _ = cv2.findContours(rank_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Heuristic mapping based on complexity
    # A = simple, single line + single shape
    # K = complex, multiple shapes
    # Q = complex with curves
    # J = medium complexity
    # T (10) = two separate digits
    # 2-9 = varying complexity
    
    if len(contours) >= 2:
        # Likely 10 (two digits) or complex face card
        if white_ratio > 0.15:
            return 'T'  # Ten has two characters
        return 'K'  # Default to King for complex shapes
    elif len(contours) == 1:
        if white_ratio < 0.08:
            return 'A'  # Ace is simple
        elif white_ratio > 0.12:
            return 'Q'  # Queen is fuller
        else:
            return 'J'  # Jack is medium
    
    # If uncertain, try to match based on height/width
    return None
```

**Improved Suit Detection**
```python
def _detect_suit(self, card_image: np.ndarray) -> Optional[str]:
    """Enhanced suit detection using both color and shape."""
    h, w = card_image.shape[:2]
    
    # Extract suit symbol region (just below rank, top-left quadrant)
    suit_region = card_image[int(h*0.15):int(h*0.35), 0:int(w*0.3)]
    
    # Method 1: Color detection (more reliable)
    is_red = self._is_red_suit(suit_region)
    
    # Method 2: Shape analysis
    gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 'h' if is_red else 's'  # Default based on color
    
    # Get largest contour (suit symbol)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    aspect = h / w if w > 0 else 1.0
    
    # Analyze shape
    # Hearts: rounded top, pointed bottom, aspect ~1.1-1.3
    # Diamonds: pointed top and bottom, aspect ~1.3-1.6
    # Clubs: rounded with stem, aspect ~1.1-1.4
    # Spades: pointed top, rounded bottom, aspect ~1.2-1.5
    
    if is_red:
        # Heart vs Diamond
        if 1.3 < aspect < 1.7:
            return 'd'  # Diamond is taller/narrower
        else:
            return 'h'  # Heart is more square
    else:
        # Club vs Spade
        # Check top portion - spades have point at top
        top_portion = thresh[0:int(h*0.3), :]
        top_white = cv2.countNonZero(top_portion)
        
        if top_white < h * w * 0.1:  # Pointed top
            return 's'  # Spade
        else:
            return 'c'  # Club
```

**Alternative: Template Matching for Ranks**
If heuristics don't work well, create small template images for each rank (A-K) and match:
```python
def _detect_rank_by_template_matching(self, rank_region: np.ndarray) -> Optional[str]:
    """Match against pre-made rank templates."""
    # Assume we have rank_templates/ folder with A.png, K.png, etc.
    rank_templates_dir = Path("rank_templates")
    
    if not rank_templates_dir.exists():
        return None
    
    best_match = None
    best_score = 0.0
    
    for rank_file in rank_templates_dir.glob("*.png"):
        rank_template = cv2.imread(str(rank_file), cv2.IMREAD_GRAYSCALE)
        
        # Resize rank_region to match template size
        resized = cv2.resize(rank_region, (rank_template.shape[1], rank_template.shape[0]))
        
        # Template matching
        result = cv2.matchTemplate(resized, rank_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = rank_file.stem  # Filename without extension
    
    if best_score > 0.6:  # Threshold for match
        return best_match
    
    return None
```

Fix 3: Ensure Recognized Names Are Used

In template_extractor.py, make sure recognized card names are actually saved as filenames:
```python
# In auto_extract_templates function:
card_name = recognizer.recognize_card(card_image)

if card_name and len(card_name) == 2:  # Valid format like "Ah", "Kd"
    filename = f"{card_name}.png"
    logger.info(f"✅ Recognized card: {card_name}")
else:
    filename = f"card_{table_id}_{position}.png"
    logger.warning(f"❌ Could not recognize card, using generic name: {filename}")
    unrecognized_count += 1
```

## Expected Results

After these fixes:

1. **Debug images** should show green boxes around actual playing cards, not player faces
2. **Console output** should show:
```
🎴 Hero Cards: Ad | Ah
🎯 STRATEGY RECOMMENDATION:
   Hand: AA
   Zone: Green  
   Action: All-in
```

3. **Templates folder** should contain files like:
```
templates/
├── Ad.png
├── Ah.png
├── Kc.png
├── 9s.png
etc.
```

Copy this ENTIRE comment to PR #17! 🎯🃏
This addresses both the detection AND the naming issues! 🚀
