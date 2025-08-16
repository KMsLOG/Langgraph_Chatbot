import logging
import sys

def setup_logger():
    """로거 설정"""
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        stream=sys.stdout 
    )
