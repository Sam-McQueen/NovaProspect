"""
sierra_prospector/core/logger.py
=====================================
Structured JSON logging. Every agent uses get_logger().

Usage:
    from core.logger import get_logger
    log = get_logger("my_agent")
    log.info("Processing cell", tile_id="Z05_R12_C34", level=5)
    log.error("Failed", tile_id="Z05_R12_C34", error=str(e))
"""

import logging
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from config.settings import LOGS_DIR, LOG_LEVEL, LOG_TO_FILE, LOG_TO_CONSOLE, LOG_JSON_FORMAT


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
            "module": record.module,
            "line":   record.lineno,
        }
        for key, val in record.__dict__.items():
            if key not in ("msg","args","levelname","levelno","pathname","filename",
                           "module","exc_info","exc_text","stack_info","lineno",
                           "funcName","created","msecs","relativeCreated","thread",
                           "threadName","processName","process","message","name","taskName"):
                try:
                    json.dumps(val)
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)
        if record.exc_info:
            payload["exception"] = {
                "type":    record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "trace":   traceback.format_exception(*record.exc_info),
            }
        return json.dumps(payload, ensure_ascii=False)


class ProspectorLogger:
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, log_level: int, msg: str, **kwargs):
        self._logger.log(log_level, msg, extra=kwargs)

    def debug(self,    msg: str, **kwargs): self._log(logging.DEBUG,    msg, **kwargs)
    def info(self,     msg: str, **kwargs): self._log(logging.INFO,     msg, **kwargs)
    def warning(self,  msg: str, **kwargs): self._log(logging.WARNING,  msg, **kwargs)
    def error(self,    msg: str, **kwargs): self._log(logging.ERROR,    msg, **kwargs)
    def critical(self, msg: str, **kwargs): self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, exc: Optional[Exception] = None, **kwargs):
        if exc:
            kwargs["exc_type"]    = type(exc).__name__
            kwargs["exc_message"] = str(exc)
            kwargs["exc_trace"]   = traceback.format_exc()
        self._logger.error(msg, exc_info=bool(exc), extra=kwargs)


def _setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    if root.handlers:
        return
    fmt = JSONFormatter() if LOG_JSON_FORMAT else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )
    if LOG_TO_CONSOLE:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if LOG_TO_FILE:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (LOGS_DIR / "agents").mkdir(exist_ok=True)
        (LOGS_DIR / "system").mkdir(exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / "system" / "prospector.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)


_setup_logging()


def get_logger(name: str) -> ProspectorLogger:
    agent_log_path = LOGS_DIR / "agents" / f"{name}.log"
    agent_log_path.parent.mkdir(parents=True, exist_ok=True)
    underlying = logging.getLogger(name)
    if not any(isinstance(h, logging.FileHandler) and
               str(agent_log_path) in str(h.baseFilename)
               for h in underlying.handlers):
        fh = logging.FileHandler(agent_log_path)
        fmt = JSONFormatter() if LOG_JSON_FORMAT else logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        )
        fh.setFormatter(fmt)
        underlying.addHandler(fh)
    return ProspectorLogger(name)
