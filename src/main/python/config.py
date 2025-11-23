import os
from datetime import datetime

# Base directory setup
BASE_DIR = "yahoo_finance_pipeline"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "stock_information"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "news_information"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "analysis"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "visualizations"), exist_ok=True)

# User agent for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

# URLs
MOST_ACTIVE_STOCKS_URL = "https://finance.yahoo.com/markets/stocks/most-active/?start={start}&count={count}"
TOPIC_URLS = [
    "https://sg.finance.yahoo.com/topic/latestnews/",
    "https://sg.finance.yahoo.com/topic/stocks/",
    "https://sg.finance.yahoo.com/topic/economy/",
    "https://sg.finance.yahoo.com/topic/earnings/",
    "https://sg.finance.yahoo.com/topic/ipo/",
    "https://sg.finance.yahoo.com/topic/cryptocurrency/",
    "https://sg.finance.yahoo.com/topic/singapore/",
    "https://sg.finance.yahoo.com/topic/technology/",
    "https://sg.finance.yahoo.com/topic/property/",
    "https://sg.finance.yahoo.com/topic/markets/",
    "https://sg.finance.yahoo.com/topic/commodities/",
    "https://sg.finance.yahoo.com/topic/currencies/",
]

# Crawling limits
DEFAULT_MAX_STOCKS = 2000
DEFAULT_MAX_ARTICLES_PER_TOPIC = 1000
DEFAULT_MAX_TOPICS = 8

# File paths
LOG_FILE = os.path.join(BASE_DIR, "logs", f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")