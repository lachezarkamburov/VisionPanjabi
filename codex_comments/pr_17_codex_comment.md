🎯 NEW PR REQUEST FOR CODEX
Title: Fix Card Detection to Recognize Actual Playing Cards
Problem
The current auto-detection system is detecting player avatars and UI elements instead of the actual playing cards visible in the poker video frames.
Current behavior:

Detects 0 cards (or wrong regions like player faces)
Output shows: 🎴 Hero Cards: card_table_3_left | card_table_3_right
Strategy shows: Hand: CC, Zone: Unknown, Action: Hold

Expected behavior:

Detect actual playing cards from the video
Output shows: 🎴 Hero Cards: Ad | Ah
Strategy shows: Hand: AA, Zone: Green, Action: All-in

Evidence from debug images:

06_all_contours.png shows many contours detected (including cards)
07_detected_cards.png shows green boxes around player avatars (0.97 confidence) and UI elements, NOT the actual playing cards


Required Changes
File 1: src/card_detector.py
Change 1.1: Add Position-Based Filtering
Playing cards appear in predictable locations on poker tables. Filter detections to focus on these regions.
In the detect_cards method, after detecting all contours and before returning, add:
```python
def detect_cards(self, frame: np.ndarray, debug: bool = False) -> List[Dict]:
    """Detect card positions using edge detection with position filtering."""
    
    # ... existing grayscale, blur, edge detection, contour finding code ...
    
    detected_cards = []
    h, w = frame.shape[:2]
    
    # Define card search region (playing cards are in center-bottom area)
    card_region_y_min = int(h * 0.35)  # Skip top 35% (headers, menus)
    card_region_y_max = int(h * 0.85)  # Skip bottom 15% (footer UI)
    card_region_x_min = int(w * 0.10)  # Skip left/right edges
    card_region_x_max = int(w * 0.90)
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        
        # Filter 1: Exclude non-rectangular shapes (player avatars are circular)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 6:
            continue  # Skip circles/irregular shapes
        
        # Calculate properties
        aspect_ratio = h_rect / w_rect if w_rect > 0 else 0
        area = w_rect * h_rect
        
        # Filter 2: Absolute size filtering (cards are consistent pixel size)
        # Playing cards in this video are roughly 40-120px wide, 60-180px tall
        if not (40 <= w_rect <= 120 and 60 <= h_rect <= 180 and area > 2000):
            continue
        
        # Filter 3: Aspect ratio (cards are vertical rectangles, ~1.2-1.8 ratio)
        if not (0.9 <= aspect_ratio <= 2.2):
            continue
        
        # Filter 4: Position filtering (cards in expected region)
        card_center_y = y + h_rect // 2
        card_center_x = x + w_rect // 2
        
        in_card_region = (card_region_y_min <= card_center_y <= card_region_y_max and
                         card_region_x_min <= card_center_x <= card_region_x_max)
        
        # Calculate confidence
        ideal_aspect = 1.4
        aspect_confidence = 1.0 - min(abs(aspect_ratio - ideal_aspect) / ideal_aspect, 1.0)
        
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / area if area > 0 else 0
        
        confidence = (aspect_confidence * 0.6 + rectangularity * 0.4)
        
        # Boost confidence if in expected region
        if in_card_region:
            confidence *= 1.5
        
        # Only keep high-confidence detections in card region
        if confidence > 0.5 and in_card_region:
            detected_cards.append({
                "x": x,
                "y": y,
                "width": w_rect,
                "height": h_rect,
                "confidence": min(confidence, 1.0)  # Cap at 1.0
            })
    
    # Filter 5: Boost confidence for horizontally paired cards
    for i, card1 in enumerate(detected_cards):
        for j, card2 in enumerate(detected_cards):
            if i >= j:
                continue
            
            # Check if cards are horizontally adjacent (hero cards)
            y_diff = abs(card1["y"] - card2["y"])
            
            if card1["x"] < card2["x"]:
                x_gap = card2["x"] - (card1["x"] + card1["width"])
            else:
                x_gap = card1["x"] - (card2["x"] + card2["width"])
            
            # Cards are paired if: similar Y position, small horizontal gap
            if y_diff < 25 and 3 <= x_gap <= 60:
                card1["confidence"] = min(card1["confidence"] * 1.4, 1.0)
                card2["confidence"] = min(card2["confidence"] * 1.4, 1.0)
    
    # Sort by confidence
    detected_cards.sort(key=lambda c: c["confidence"], reverse=True)
    
    logger.info(f"Detected {len(detected_cards)} potential cards")
    
    if debug:
        self._save_debug_images(frame, detected_cards, card_region_y_min, 
                               card_region_y_max, card_region_x_min, card_region_x_max)
    
    return detected_cards


def _save_debug_images(self, frame: np.ndarray, detected_cards: List[Dict],
                       y_min: int, y_max: int, x_min: int, x_max: int):
    """Save debug visualization with search region."""
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    
    # Draw search region in blue
    debug_region = frame.copy()
    cv2.rectangle(debug_region, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
    cv2.putText(debug_region, "Card Search Region", (x_min + 10, y_min + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imwrite(str(debug_dir / "08_search_region.png"), debug_region)
    
    # Draw detected cards
    debug_detected = frame.copy()
    for idx, card in enumerate(detected_cards):
        color = (0, 255, 0) if card["confidence"] > 0.7 else (0, 165, 255)
        cv2.rectangle(debug_detected,
                     (card["x"], card["y"]),
                     (card["x"] + card["width"], card["y"] + card["height"]),
                     color, 3)
        cv2.putText(debug_detected, f"{card['confidence']:.2f}",
                   (card["x"], card["y"] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(str(debug_dir / "09_filtered_cards.png"), debug_detected)
    logger.info(f"Debug images saved to {debug_dir}/")
```

File 2: src/card_recognizer.py
Change 2.1: Improve Rank Detection Fallback
The current fallback doesn't work. Implement better heuristics:
```python
def _detect_rank_fallback(self, thresh_corner: np.ndarray) -> Optional[str]:
    """
    Improved fallback rank detection using pattern analysis.
    Analyzes contour count and white pixel density.
    """
    if thresh_corner is None or thresh_corner.size == 0:
        return None
    
    # Find contours in rank region
    contours, _ = cv2.findContours(thresh_corner, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate white pixel ratio
    white_pixels = cv2.countNonZero(thresh_corner)
    total_pixels = thresh_corner.shape[0] * thresh_corner.shape[1]
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    num_contours = len(contours)
    
    # Heuristic mapping based on complexity
    # A = simple (1 contour, low density)
    # K, Q, J = complex (1-2 contours, higher density)
    # T (10) = two separate shapes (2+ contours)
    # 2-9 = varying patterns
    
    if num_contours == 0:
        return None
    elif num_contours >= 2:
        # Likely "10" (two digits) or multi-part letter
        if white_ratio > 0.15:
            return 'T'  # Ten has two characters
        return 'K'  # King is complex
    elif num_contours == 1:
        # Single character - use density to differentiate
        if white_ratio < 0.07:
            return 'A'  # Ace is sparse
        elif white_ratio > 0.14:
            return 'Q'  # Queen is dense
        elif white_ratio > 0.10:
            return 'K'  # King medium-high
        else:
            return 'J'  # Jack medium
    
    # Default fallback
    return 'A'
```
Change 2.2: Improve Suit Detection
```python
def _detect_suit(self, card_image: np.ndarray) -> Optional[str]:
    """Enhanced suit detection using color + shape."""
    h, w = card_image.shape[:2]
    
    # Extract suit symbol region (below rank, top-left area)
    suit_region = card_image[int(h*0.15):int(h*0.40), 0:int(w*0.35)]
    
    if suit_region.size == 0:
        return 'h'  # Default fallback
    
    # Color detection
    is_red = self._is_red_suit(suit_region)
    
    # Shape analysis
    gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 'h' if is_red else 's'
    
    # Analyze largest contour (suit symbol)
    largest = max(contours, key=cv2.contourArea)
    x, y, w_s, h_s = cv2.boundingRect(largest)
    aspect = h_s / w_s if w_s > 0 else 1.0
    
    # Use color + aspect ratio to determine suit
    if is_red:
        # Hearts vs Diamonds
        # Diamonds are taller/narrower (aspect > 1.4)
        return 'd' if aspect > 1.4 else 'h'
    else:
        # Clubs vs Spades
        # Spades are slightly taller (aspect > 1.35)
        return 's' if aspect > 1.35 else 'c'
```

File 3: src/template_extractor.py
Change 3.1: Ensure Proper Card Names Are Saved
```python
def auto_extract_templates(vision: MultiTableVision, templates_dir: Path, 
                          force: bool = False) -> None:
    """Extract templates with automatic naming."""
    
    # ... existing checks and setup ...
    
    recognizer = CardRecognizer()
    template_count = 0
    recognized_count = 0
    
    for table_id, layout in vision.table_layouts.items():
        # Extract left card
        left_roi = vision._get_table_roi(layout, vision.base_rois.hero_left)
        left_card_img = frame[left_roi.y:left_roi.y+left_roi.height,
                             left_roi.x:left_roi.x+left_roi.width]
        
        # Try to recognize
        card_name = recognizer.recognize_card(left_card_img)
        
        if card_name and len(card_name) == 2 and card_name[0] in 'AKQJT98765432' and card_name[1] in 'hdcs':
            filename = f"{card_name}.png"
            recognized_count += 1
            logger.info(f"✅ Recognized: {card_name}")
        else:
            filename = f"card_{table_id}_left.png"
            logger.warning(f"❌ Could not recognize card from {table_id} left, using generic name")
        
        cv2.imwrite(str(templates_dir / filename), left_card_img)
        template_count += 1
        
        # Same for right card
        # ... (repeat above logic for hero_right)
    
    logger.info(f"Extracted {template_count} templates ({recognized_count} recognized)")
    
    # Reload matchers
    vision.hero_left_matcher.templates = vision.hero_left_matcher._load_templates()
    vision.hero_right_matcher.templates = vision.hero_right_matcher._load_templates()
```

Testing Instructions
After implementing these changes:
```bash
# Clean slate
rm -rf templates/ debug/

# Run with debug
python src/main.py --auto-detect-roi --debug

# Check debug folder
explorer debug/
```

## Expected Results

**Console Output:**
```
INFO:card_detector:Detected 8 potential cards
INFO:card_detector:Found 4 card pairs
✅ Recognized: Ad
✅ Recognized: Ah
...

🎴 Hero Cards: Ad | Ah
🎯 STRATEGY RECOMMENDATION:
   Hand: AA
   Zone: Green
   Action: All-in
```

**Debug Images:**
- `debug/08_search_region.png` - Blue box showing where we search for cards
- `debug/09_filtered_cards.png` - Green boxes around ACTUAL PLAYING CARDS (not avatars)

**Templates Folder:**
```
templates/
├── Ad.png
├── Ah.png
├── Kc.png
├── 9s.png
...
```

Success Criteria

✅ debug/09_filtered_cards.png shows green boxes on actual playing cards
✅ Console shows real card names like Ad, Ah, Kc
✅ Strategy shows proper poker hands like AA, AK, KK
✅ No more card_table_X_left in output


Copy this ENTIRE prompt to Codex to create a new PR! 🎯🚀Claude is AI and can make mistakes. Please double-check responses.
