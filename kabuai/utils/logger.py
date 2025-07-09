import logging
import os


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[31m%(levelname)s\033[0m - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
