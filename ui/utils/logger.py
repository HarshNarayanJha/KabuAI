import logging
import os


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="\033[32m%(asctime)s\033[0m - \033[34m%(name)s\033[0m - \033[31m%(levelname)s\033[0m - \033[37m%(message)s\033[0m",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("watchdog").setLevel("ERROR")
    logging.getLogger("urllib3").setLevel("ERROR")
    logging.getLogger("requests_sse.client").setLevel("ERROR")
