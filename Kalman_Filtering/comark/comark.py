import json
import logging
from config import COMARK_FILE


def load_comark_data():
    """Load co-marking JSON file specified in the project config.

    Returns:
        The parsed JSON object on success, or None if the file could not be read/parsed.
    """
    try:
        with open(COMARK_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except FileNotFoundError:
        logging.error(f"COMARK file not found: {COMARK_FILE}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse COMARK JSON ({COMARK_FILE}): {e}")
    except Exception as e:
        logging.error(f"Unexpected error while loading COMARK file {COMARK_FILE}: {e}")
    return None