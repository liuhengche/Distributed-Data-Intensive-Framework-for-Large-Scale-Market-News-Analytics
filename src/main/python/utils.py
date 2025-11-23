import re
import json
import os
import time
from datetime import datetime, timezone
import logging
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
import numpy as np

def setup_logging(log_file):
    """Configure logging with file and console handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_current_timestamp():
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def safe_float(value, default=0.0):
    """Safely convert value to float, handling various formats"""
    try:
        if isinstance(value, str):
            # Remove common formatting characters
            value = value.replace(',', '').replace('$', '').strip()
            # Handle percentage values
            if value.endswith('%'):
                value = value[:-1]
            # Handle empty or invalid values
            if value in ['-', '', 'N/A', '--', 'n/a']:
                return default
            # Handle values with + sign
            if value.startswith('+'):
                value = value[1:]
        return float(value)
    except (ValueError, TypeError):
        return default

def create_success_logger(logger):
    """Create a custom success logging level"""
    SUCCESS_LEVEL_NUM = 25
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
    
    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
    
    logging.Logger.success = success
    return logger

def save_json(data, filepath):
    """Save data to JSON file with proper formatting"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath):
    """Load data from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)