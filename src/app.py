from src.logger_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

def main():
    print("Hello from langgraph-chatbot!")


if __name__ == "__main__":
    main()
