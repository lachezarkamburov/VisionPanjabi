import logging
from typing import Optional

import cv2
import numpy as np


class CardRecognizer:
    """Recognize poker card rank and suit from card images using OCR and color analysis."""

    RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    SUITS = ["h", "d", "c", "s"]  # hearts, diamonds, clubs, spades

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def recognize_card(self, card_image: np.ndarray) -> Optional[str]:
        """
        Recognize card from image using OCR and color detection.

        Args:
            card_image: OpenCV image of a playing card.

        Returns:
            Card name like "Ah", "Kc", "6s" or None if recognition fails.
        """
        rank = self._detect_rank(card_image)
        suit = self._detect_suit(card_image)

        if rank and suit:
            card_name = f"{rank}{suit}"
            self.logger.info("Recognized card: %s", card_name)
            return card_name

        self.logger.warning("Failed to recognize card")
        return None

    def _detect_rank(self, card_image: np.ndarray) -> Optional[str]:
        """
        Detect card rank using OCR on the top-left corner.

        Strategy:
        1. Extract top-left corner (where rank is typically shown).
        2. Convert to grayscale and threshold.
        3. Use template matching or OCR to identify rank character.
        4. Match against known ranks: A, K, Q, J, T, 9-2.
        """
        h, w = card_image.shape[:2]
        corner = card_image[0 : int(h * 0.3), 0 : int(w * 0.3)]

        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            import pytesseract

            text = pytesseract.image_to_string(
                thresh,
                config="--psm 10 -c tessedit_char_whitelist=AKQJT98765432",
            ).strip()

            text = text.replace("0", "Q").replace("l", "1").replace("I", "1")

            if text in self.RANKS:
                return text

            if text and text[0] in self.RANKS:
                return text[0]

        except ImportError:
            self.logger.warning(
                "pytesseract not installed, using fallback method. "
                "Install with: Windows 'choco install tesseract', "
                "Mac 'brew install tesseract', Linux 'sudo apt-get install tesseract-ocr'"
            )
            return self._detect_rank_fallback(thresh)
        except Exception as exc:
            self.logger.warning(
                "OCR failed (%s), using fallback method. "
                "Install Tesseract with: Windows 'choco install tesseract', "
                "Mac 'brew install tesseract', Linux 'sudo apt-get install tesseract-ocr'",
                type(exc).__name__,
            )
            return self._detect_rank_fallback(thresh)

        return None

    def _detect_rank_fallback(self, thresh_corner: np.ndarray) -> Optional[str]:
        """
        Improved fallback rank detection using pattern analysis.
        Analyzes contour count and white pixel density.
        """
        if thresh_corner is None or thresh_corner.size == 0:
            return None

        # Find contours in rank region
        contours, _ = cv2.findContours(
            thresh_corner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
                return "T"  # Ten has two characters
            return "K"  # King is complex
        elif num_contours == 1:
            # Single character - use density to differentiate
            if white_ratio < 0.07:
                return "A"  # Ace is sparse
            elif white_ratio > 0.14:
                return "Q"  # Queen is dense
            elif white_ratio > 0.10:
                return "K"  # King medium-high
            else:
                return "J"  # Jack medium

        # Default fallback
        return "A"

    def _detect_suit(self, card_image: np.ndarray) -> Optional[str]:
        """
        Detect card suit using color analysis.

        Strategy:
        1. Extract suit symbol region (below rank in top-left corner).
        2. Analyze dominant color.
        3. Analyze shape to distinguish among suits.
        """
        h, w = card_image.shape[:2]
        suit_region = card_image[int(h * 0.15) : int(h * 0.40), 0 : int(w * 0.35)]

        if suit_region.size == 0:
            return "h"  # Default fallback

        # Color detection
        is_red = self._is_red_suit(suit_region)

        # Shape analysis
        gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return "h" if is_red else "s"

        # Analyze largest contour (suit symbol)
        largest = max(contours, key=cv2.contourArea)
        x, y, w_s, h_s = cv2.boundingRect(largest)
        aspect = h_s / w_s if w_s > 0 else 1.0

        # Use color + aspect ratio to determine suit
        if is_red:
            # Hearts vs Diamonds
            # Diamonds are taller/narrower (aspect > 1.4)
            return "d" if aspect > 1.4 else "h"
        else:
            # Clubs vs Spades
            # Spades are slightly taller (aspect > 1.35)
            return "s" if aspect > 1.35 else "c"

    def _is_red_suit(self, suit_region: np.ndarray) -> bool:
        """Check if suit symbol is red or black based on color analysis."""
        hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)

        mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 50, 255))
        black_pixels = cv2.countNonZero(mask_black)

        return red_pixels > black_pixels

    def _analyze_suit_shape(self, suit_region: np.ndarray) -> str:
        """
        Analyze suit symbol shape.

        Returns:
            "rounded", "pointed", or "diamond".
        """
        gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return "rounded"

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio: float = float(w) / h if h > 0 else 1.0

        if 0.8 < aspect_ratio < 1.2:
            return "diamond"
        if aspect_ratio < 0.8:
            return "pointed"
        return "rounded"
