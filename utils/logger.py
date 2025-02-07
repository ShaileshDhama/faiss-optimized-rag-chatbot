import logging
from config import LOG_FILE

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_event(event):
    """Log system events."""
    logging.info(event)
