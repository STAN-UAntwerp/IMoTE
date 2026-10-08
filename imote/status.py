"""User-facing status messages.

Callbacks report the outcome of an action by writing one of these messages to the ``ids.STORE_STATUS`` store.
The status bar above the tree graph shows the latest message, older messages stay available in its history.

Levels map to Bootstrap colors:
    success: an action worked, e.g. a tree was fitted or saved.
    info:    neutral information, e.g. nothing had to change.
    warning: the action isn't possible right now, the user can fix it (e.g. no node selected).
    danger:  something failed, e.g. a file couldn't be read or a fit crashed.
"""
import logging
import sys
import time
from datetime import datetime

logger = logging.getLogger(__name__)

SUCCESS = "success"
INFO = "info"
WARNING = "warning"
ERROR = "danger"


def _message(level: str, text: str) -> dict:
    # id is unique per message, so repeating the same text still counts as a new message
    return {"id": time.time_ns(), "level": level, "text": text, "time": datetime.now().strftime("%H:%M:%S")}


def success(text: str) -> dict:
    logger.debug(text)
    return _message(SUCCESS, text)


def info(text: str) -> dict:
    logger.debug(text)
    return _message(INFO, text)


def warning(text: str) -> dict:
    logger.warning(text)
    return _message(WARNING, text)


def error(text: str) -> dict:
    logger.error(text, exc_info=sys.exc_info()[0] is not None)
    return _message(ERROR, text)


def with_note(message: dict, note: str) -> dict:
    return _message(message["level"], f"{message['text']} {note}")


WELCOME = {**info("Click a node in the graph to see its details. "
                  "Use the cards on the right to fit, edit or explore the tree."), "time": ""}
