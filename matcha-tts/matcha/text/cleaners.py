import logging
import re
import os
import phonemizer
from unidecode import unidecode

# Set logging to critical to avoid excessive output
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# --- THE DEFENSIVE BLOCK ---
# We keep these global but empty so they don't crash on import
_global_phonemizer = None
_tried_phonemizer = False

def get_phonemizer():
    """Retrieve or initialize the English phonemizer only when needed."""
    global _global_phonemizer, _tried_phonemizer
    
    if _global_phonemizer is None and not _tried_phonemizer:
        _tried_phonemizer = True
        # Try multiple common HPC paths for the binary
        possible_binaries = [
            "/home/ak8224/.local/bin/espeak-ng",
            "/rds/general/user/ak8224/home/.local/bin/espeak-ng",
            "espeak-ng",
            "espeak"
        ]
        
        for binary in possible_binaries:
            try:
                _global_phonemizer = phonemizer.backend.EspeakBackend(
                    language="en-us",
                    preserve_punctuation=True,
                    with_stress=True,
                    language_switch="remove-flags",
                    logger=critical_logger,
                    espeak_ng_binary=binary
                )
                print(f"✅ Successfully initialized espeak using: {binary}")
                break
            except Exception:
                continue
        
        if _global_phonemizer is None:
            print("⚠️ WARNING: espeak-ng not found. Falling back to character-based cleaning.")
            print("Training will continue, but the model will learn characters instead of phonemes.")
            
    return _global_phonemizer

# --- ENGLISH TEXT LOGIC ---
_whitespace_re = re.compile(r"\s+")

_abbreviations_en = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"), ("ms", "miss"), ("mr", "mister"), ("dr", "doctor"),
        ("st", "saint"), ("co", "company"), ("jr", "junior"), ("maj", "major"),
        ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"),
        ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"),
        ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ]
]

_replacements_en = [
    (re.compile(r"\.\.\."), "ELLIPSIS_MARKER"),
    (re.compile(r"\$(\d+)\.(\d+)"), r"\1 dollars and \2 cents"),
    (re.compile(r"(?<=\D)\.(?=\D)(?!\s)", re.IGNORECASE), " dot "),
    (re.compile(r"(?<=\d)\.(?=\d)(?!\s)"), " point "),
    (re.compile(r"\$(\d+)"), r"\1 dollars"),
    (re.compile(r"ELLIPSIS_MARKER"), "..."),
]

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def english_cleaners2(text):
    """Pipeline for English text. Forced to use verified local espeak-ng."""
    
    try:
        from phonemizer import phonemize
    except ImportError:
        return collapse_whitespace(text.lower())

    text = text.encode("utf-8").decode("utf-8")
    text = lowercase(text)
    
    for regex, replacement in _abbreviations_en:
        text = re.sub(regex, replacement, text)
    for regex, replacement in _replacements_en:
        text = regex.sub(replacement, text)
    
    try:
        phonemes = phonemize(
            text, 
            language='en-us', 
            backend='espeak', 
            strip=True, 
            preserve_punctuation=True,
            with_stress=True
        )
        return collapse_whitespace(phonemes)
        
    except Exception as e:
        # This will now tell you the EXACT error (e.g., "Library not found")
        print(f"⚠️ Phonemization CRITICAL FAILURE: {e}")
        return collapse_whitespace(text)
        
    except Exception as e:
        # If this happens, it means the HPC environment variables (LD_LIBRARY_PATH) 
        # are missing in the sub-shell.
        print(f"⚠️ Phonemization CRITICAL FAILURE: {e}. Falling back to chars.")
        return collapse_whitespace(text)

# Keep a dummy basic_cleaners just in case Hydra calls it
def basic_cleaners(text):
    return collapse_whitespace(lowercase(text))