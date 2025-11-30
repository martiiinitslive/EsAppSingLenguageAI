"""Speech-to-text utilities.

Primary behaviour:
- Try to transcribe using `faster-whisper` (local Whisper). This is more
  robust and works offline.
- If `faster-whisper` is not installed or fails, fall back to
  `speech_recognition` with Google Web Speech.

The function prints diagnostic output so the backend logs show what was used
and what text (if any) was produced.
"""

def _whisper_transcribe(audio_path, model_name="small"):
    try:
        from faster_whisper import WhisperModel
    except Exception:
        raise RuntimeError("faster-whisper not available")

    try:
        # prefer float32 on CPU for compatibility
        model = WhisperModel(model_name, device="cpu", compute_type="float32")
    except TypeError:
        model = WhisperModel(model_name, device="cpu")

    # Request transcription (no translation)
    raw = model.transcribe(str(audio_path), task="transcribe")

    # Coerce various shapes returned by faster-whisper into a list of (start,end,text)
    def _is_segment_like(obj):
        try:
            if obj is None:
                return False
            if hasattr(obj, 'start') and hasattr(obj, 'end'):
                return True
            if isinstance(obj, (list, tuple)) and len(obj) >= 3:
                return True
        except Exception:
            pass
        return False

    def _coerce_to_segments(raw_obj):
        segs = []
        # If tuple/list, try to find a member that holds segments
        if isinstance(raw_obj, (list, tuple)):
            for item in raw_obj:
                if item is None:
                    continue
                if hasattr(item, 'segments'):
                    raw_obj = item.segments
                    break
                try:
                    it = iter(item)
                    first = None
                    for first in it:
                        break
                    if _is_segment_like(first):
                        raw_obj = [first] + list(it)
                        break
                except Exception:
                    pass

        if hasattr(raw_obj, 'segments'):
            raw_iter = raw_obj.segments
        elif hasattr(raw_obj, 'data'):
            raw_iter = raw_obj.data
        else:
            raw_iter = raw_obj

        try:
            if not isinstance(raw_iter, (list, tuple)):
                raw_list = list(raw_iter)
            else:
                raw_list = list(raw_iter)
        except Exception:
            raw_list = []

        for segment in raw_list:
            try:
                if hasattr(segment, 'start') and hasattr(segment, 'end'):
                    start = float(segment.start)
                    end = float(segment.end)
                    text = getattr(segment, 'text', None)
                    if text is None:
                        text = str(segment)
                else:
                    start = float(segment[0])
                    end = float(segment[1])
                    text = segment[2]
            except Exception:
                continue
            segs.append((start, end, text))

        if not segs:
            plain = None
            try:
                if isinstance(raw_obj, (list, tuple)) and len(raw_obj) >= 1 and isinstance(raw_obj[0], str):
                    plain = raw_obj[0]
                elif isinstance(raw_obj, str):
                    plain = raw_obj
                elif hasattr(raw_obj, 'text'):
                    plain = raw_obj.text
            except Exception:
                plain = None
            if plain:
                segs.append((0.0, 0.0, str(plain)))

        return segs

    segments = _coerce_to_segments(raw)
    full = '\n'.join([s[2].strip() for s in segments if s and s[2]])
    print(f"[STT][whisper] Transcription (model={model_name}): {full}")
    return full if full != '' else None


def _google_transcribe(audio_path, language="es-ES"):
    try:
        import speech_recognition as sr
    except Exception:
        raise RuntimeError("speech_recognition not available")

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language=language)
        print(f"[STT][google] Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("[STT][google] Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"[STT][google] Recognition service error: {e}")
        return None


def speech_to_text(audio_path, prefer="whisper"):
    """Transcribe `audio_path` to text.

    Args:
        audio_path: path to WAV/ audio file
        prefer: 'whisper' to prefer faster-whisper, 'google' to prefer SpeechRecognition

    Returns:
        Transcribed string or None if not recognized.
    """
    # Try Whisper first (recommended) unless user asked otherwise
    if prefer == "whisper":
        try:
            txt = _whisper_transcribe(audio_path)
            if txt:
                return txt
        except Exception as e:
            print(f"[STT] faster-whisper not used: {e}")

        # fallback to google
        try:
            return _google_transcribe(audio_path)
        except Exception as e:
            print(f"[STT] speech_recognition not available: {e}")
            return None

    else:
        # prefer google first
        try:
            txt = _google_transcribe(audio_path)
            if txt:
                return txt
        except Exception as e:
            print(f"[STT] speech_recognition not available: {e}")

        try:
            return _whisper_transcribe(audio_path)
        except Exception as e:
            print(f"[STT] faster-whisper not available: {e}")
            return None
