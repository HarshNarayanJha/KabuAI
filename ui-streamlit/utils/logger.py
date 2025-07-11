import logging
import os
from typing import override


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    level_colors = {
        "DEBUG": "\033[34m",  # Blue
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }

    class ColorFormatter(logging.Formatter):
        @override
        def format(self, record: logging.LogRecord):
            record.levelname_color = level_colors.get(record.levelname, "\033[0m") + record.levelname + "\033[0m"
            return super().format(record)

    formatter = ColorFormatter(
        "\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - %(levelname_color)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    logging.getLogger("watchdog").setLevel("ERROR")
    logging.getLogger("urllib3").setLevel("ERROR")
    logging.getLogger("requests_sse.client").setLevel("ERROR")
