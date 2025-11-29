"""
Text normalization helper for the renderer.

Functions:
  normalize_text_for_renderer(text) -> cleaned string suitable for run_pose_to_video_mediapipe.text_to_pose_sequence

Behavior:
  - Normalize unicode (NFKD), remove diacritics
  - Convert common punctuation to ASCII equivalents
  - Keep letters, digits, spaces, periods and commas
  - Collapse multiple spaces
  - Trim
"""
import unicodedata
import re

def normalize_text_for_renderer(text: str) -> str:
    if text is None:
        return ""
    # Normalize and remove diacritics
    s = unicodedata.normalize('NFKD', str(text))
    s = ''.join(c for c in s if not unicodedata.combining(c))

    # Replace Spanish opening punctuation and unusual dashes
    s = s.replace('¿', '').replace('¡', '')
    s = s.replace('—', '-').replace('–', '-')

    # Keep only letters, digits, space, period, comma and basic punctuation
    # Convert non-supported punctuation to spaces (so tokens separate)
    s = re.sub(r"[^A-Za-z0-9 .,\-]", ' ', s)

    # Collapse multiple spaces
    s = re.sub(r"\s+", ' ', s)
    s = s.strip()
    return s
