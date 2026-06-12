# src/utils.py
import logging
import os

def setup_logger(log_file_name="execution.log"):
    # check logs floder
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    logger = logging.getLogger("RL_Logger")
    logger.setLevel(logging.INFO)
    
    # avoid duplicate log
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(f"logs/{log_file_name}")
        fh.setLevel(logging.INFO)
        
        # Stream Handler (print on Jupyter Notebook)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # set formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger