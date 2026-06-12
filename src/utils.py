# src/utils.py
import logging
import os

def setup_logger(log_file_name="Assignment1_Execution.log"):
    # Check if logs directory exists, if not create it
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    logger = logging.getLogger("RL_Assignment1_Logger")
    logger.setLevel(logging.INFO)
    
    # clear existing handlers to avoid duplicate logs if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # mode='w' new file each time, encoding='utf-8' to support all characters
    fh = logging.FileHandler(f"logs/{log_file_name}", mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # set
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
        
    return logger

def log_section(logger, title, is_main_title=False):
    # formatting
    if is_main_title:
        logger.info("=" * 50)
        logger.info(f"{title:^50}")
        logger.info("=" * 50)
        logger.info("-" * 50)
    else:
        logger.info("=" * 50)
        logger.info(f"{title:^50}")
        logger.info("=" * 50)