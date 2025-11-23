#!/usr/bin/env python3
"""
Yahoo Finance Data Pipeline
Combines stock data crawling, news article crawling, and sentiment analysis
"""

import os
import json
import time
import re
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dateparser import parse as date_parse
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# ======================
# SETUP AND CONFIGURATION
# ======================

# Create directory structure
BASE_DIR = "yahoo_finance_pipeline"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "stock_information"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "news_information"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "analysis"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# Configure logging
log_file = os.path.join(BASE_DIR, "logs", f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

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
MAX_STOCKS = 200
MAX_ARTICLES_PER_TOPIC = 15
MAX_TOPICS = 8

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
    """Safely convert value to float, handling various formats (增强版)"""
    try:
        if isinstance(value, str):
            # 移除常见格式化字符
            value = value.replace(',', '').replace('$', '').replace(' ', '').strip()
            # 处理百分比、单位后缀
            if value.endswith('%'):
                value = value[:-1]
            # 处理空值或无效值
            if value in ['-', '', 'N/A', '--', 'n/a', 'None']:
                return default
            # 处理科学计数法
            if 'e' in value.lower():
                return float(value)
            # 处理带符号的值
            if value.startswith(('+', '-')):
                return float(value)
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default

# ======================
# STOCK DATA CRAWLER
# ======================

class StockDataCrawler:
    """Crawler for Yahoo Finance most active stocks data"""
    
    def __init__(self):
        self.session = SESSION
        self.base_url = "https://finance.yahoo.com"
    
    def parse_stock_row(self, row):
        """Parse a single stock row from the most active stocks table"""
        try:
            # Get symbol - look for the span with class containing 'symbol'
            symbol = "N/A"
            symbol_span = row.find('span', class_=lambda x: x and 'symbol' in x)
            if symbol_span:
                symbol_text = clean_text(symbol_span.text.strip())
                if symbol_text:
                    symbol = symbol_text.split()[0]  # Take first part (remove trailing spaces)
            
            # Get company name - look for the cell with data-testid-cell="companyshortname.raw"
            name = "N/A"
            name_cell = row.find('td', {'data-testid-cell': 'companyshortname.raw'})
            if name_cell:
                name_div = name_cell.find('div', class_=lambda x: x and 'companyName' in x)
                if name_div:
                    name = clean_text(name_div.text.strip())
                else:
                    name = clean_text(name_cell.text.strip())
            
            # Get price - look for fin-streamer with data-field="regularMarketPrice"
            price = "N/A"
            price_streamer = row.find('fin-streamer', {'data-field': 'regularMarketPrice'})
            if price_streamer:
                price_text = clean_text(price_streamer.text.strip())
                if price_text:
                    price = price_text
            
            # Get change amount - look for fin-streamer with data-field="regularMarketChange"
            change_amount = "N/A"
            change_streamer = row.find('fin-streamer', {'data-field': 'regularMarketChange'})
            if change_streamer:
                change_span = change_streamer.find('span', class_=lambda x: x and ('txt-positive' in x or 'txt-negative' in x))
                if change_span:
                    change_text = clean_text(change_span.text.strip())
                    if change_text:
                        change_amount = change_text
                else:
                    change_text = clean_text(change_streamer.text.strip())
                    if change_text:
                        change_amount = change_text
            
            # Get change percent - look for fin-streamer with data-field="regularMarketChangePercent"
            change_percent = "N/A"
            percent_streamer = row.find('fin-streamer', {'data-field': 'regularMarketChangePercent'})
            if percent_streamer:
                percent_span = percent_streamer.find('span', class_=lambda x: x and ('txt-positive' in x or 'txt-negative' in x))
                if percent_span:
                    percent_text = clean_text(percent_span.text.strip())
                    if percent_text:
                        change_percent = percent_text
                else:
                    percent_text = clean_text(percent_streamer.text.strip())
                    if percent_text:
                        change_percent = percent_text
            
            # Get volume - look for fin-streamer with data-field="regularMarketVolume"
            volume = 0.0  # 改为数值默认值
            volume_streamer = row.find('fin-streamer', {'data-field': 'regularMarketVolume'})
            if volume_streamer:
                volume_text = clean_text(volume_streamer.text.strip())
                if volume_text and volume_text not in ['N/A', '--', '-']:
                    # 清洗成交量：移除逗号，处理百万/十亿单位（如 1.2M → 1200000，5B → 5000000000）
                    volume_text = volume_text.replace(',', '').upper()
                    if volume_text.endswith('M'):
                        volume = safe_float(volume_text[:-1]) * 1_000_000
                    elif volume_text.endswith('B'):
                        volume = safe_float(volume_text[:-1]) * 1_000_000_000
                    elif volume_text.endswith('K'):
                        volume = safe_float(volume_text[:-1]) * 1_000
                    else:
                        volume = safe_float(volume_text)  # 普通数值直接转换
            
            # Get market cap - look for fin-streamer with data-field="marketCap"
            market_cap = "N/A"
            market_cap_streamer = row.find('fin-streamer', {'data-field': 'marketCap'})
            if market_cap_streamer:
                market_cap_text = clean_text(market_cap_streamer.text.strip())
                if market_cap_text:
                    market_cap = market_cap_text
            
            # Get P/E ratio - look for cell with data-testid-cell="peratio.lasttwelvemonths"
            pe_ratio = "N/A"
            pe_cell = row.find('td', {'data-testid-cell': 'peratio.lasttwelvemonths'})
            if pe_cell:
                pe_text = clean_text(pe_cell.text.strip())
                if pe_text and pe_text not in ['--', '-', 'N/A']:
                    pe_ratio = pe_text
            
            # Get 52-week change - look for fin-streamer with data-field="fiftyTwoWeekChangePercent"
            week52_change = "N/A"
            week52_streamer = row.find('fin-streamer', {'data-field': 'fiftyTwoWeekChangePercent'})
            if week52_streamer:
                week52_span = week52_streamer.find('span', class_=lambda x: x and ('txt-positive' in x or 'txt-negative' in x))
                if week52_span:
                    week52_text = clean_text(week52_span.text.strip())
                    if week52_text:
                        week52_change = week52_text
            
            # Get 52-week range - look for cell with data-testid-cell="fiftyTwoWeekRange"
            week52_range = "N/A"
            range_cell = row.find('td', {'data-testid-cell': 'fiftyTwoWeekRange'})
            if range_cell:
                labels_div = range_cell.find('div', class_=lambda x: x and 'labels' in x)
                if labels_div:
                    spans = labels_div.find_all('span')
                    if len(spans) >= 2:
                        low = clean_text(spans[0].text.strip())
                        high = clean_text(spans[1].text.strip())
                        week52_range = f"{low} - {high}"
            
            # Validate minimum required data
            if symbol == "N/A" or price == "N/A" or symbol == "" or price == "":
                return None
            
            # Convert numeric values safely
            price_value = safe_float(price)
            change_amount_value = safe_float(change_amount)
            change_percent_value = safe_float(change_percent.replace('%', ''))
            pe_ratio_value = safe_float(pe_ratio)
            week52_change_value = safe_float(week52_change.replace('%', ''))
            
            return {
                'symbol': symbol,
                'name': name,
                'price': price_value,
                'change_amount': change_amount_value,
                'change_percent': change_percent_value,
                'volume': volume,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio_value,
                'week52_change': week52_change_value,
                'week52_range': week52_range,
                'timestamp': get_current_timestamp(),
                'url': f"{self.base_url}/quote/{symbol}/"
            }
            
        except Exception as e:
            logger.error(f"Error parsing stock row: {e}")
            return None
    
    def crawl_most_active_stocks(self, max_stocks=200):
        """Crawl most active stocks from Yahoo Finance"""
        logger.info(f"🔍 Starting to crawl most active stocks (target: {max_stocks})...")
        
        all_stocks = []
        start = 0
        count = 25  # Smaller batch size for reliability
        
        while len(all_stocks) < max_stocks:
            url = MOST_ACTIVE_STOCKS_URL.format(start=start, count=count)
            logger.info(f"🔗 Fetching URL: {url}")
            
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find rows with the exact structure
                rows = soup.find_all('tr', {'data-testid': 'data-table-v2-row'})
                
                logger.info(f"📊 Found {len(rows)} rows with data-testid='data-table-v2-row'")
                
                if not rows:
                    logger.warning("⚠️ No rows found with the expected structure. Trying alternative selectors...")
                    # Try alternative selectors
                    rows = soup.find_all('tr', class_=lambda x: x and 'row' in x)
                    logger.info(f"🔄 Found {len(rows)} rows with class containing 'row'")
                
                if not rows:
                    logger.error("❌ No rows found at all. Stopping crawl.")
                    break
                
                new_stocks = []
                
                for i, row in enumerate(rows):
                    # Skip rows that don't have enough cells
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 5:
                        continue
                    
                    stock_data = self.parse_stock_row(row)
                    if stock_data:
                        new_stocks.append(stock_data)
                        logger.debug(f"✅ Row {i} ({stock_data['symbol']}): ${stock_data['price']:.2f} ({stock_data['change_percent']:+.2f}%)")
                    
                    # Stop if we have enough for this batch
                    if len(new_stocks) >= count:
                        break
                
                logger.info(f"📊 Processed {len(rows)} rows, found {len(new_stocks)} valid stocks")
                
                if not new_stocks:
                    logger.warning("⚠️ No valid stocks found in this batch. Stopping crawl.")
                    break
                
                # Remove duplicates by symbol
                unique_stocks = []
                seen_symbols = set()
                for stock in new_stocks:
                    if stock['symbol'] not in seen_symbols:
                        seen_symbols.add(stock['symbol'])
                        unique_stocks.append(stock)
                
                all_stocks.extend(unique_stocks)
                logger.info(f"✅ Found {len(unique_stocks)} unique stocks in this batch. Total: {len(all_stocks)}")
                
                # If we got fewer stocks than requested, we might be at the end
                if len(new_stocks) < count * 0.5:  # Less than 50% of requested
                    logger.info("🔄 Reached end of available data. Stopping crawl.")
                    break
                
                start += count
                time.sleep(2)  # Longer delay to be extra respectful
                
                if len(all_stocks) >= max_stocks:
                    all_stocks = all_stocks[:max_stocks]
                    logger.info(f"🎯 Reached target of {max_stocks} stocks. Stopping crawl.")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error crawling stocks from {url}: {e}")
                break
        
        logger.info(f"✅ Successfully crawled {len(all_stocks)} most active stocks")
        return all_stocks
    
    def save_stock_data(self, stocks, filename=None):
        """Save stock data to JSON and CSV files"""
        if not stocks:
            logger.warning("⚠️ No stocks to save")
            return None, None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"most_active_stocks_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(BASE_DIR, "stock_information", f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stocks, f, ensure_ascii=False, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(stocks)
        csv_path = os.path.join(BASE_DIR, "stock_information", f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"💾 Saved stock data to {json_path} and {csv_path}")
        
        # Generate summary statistics
        summary = {
            "total_stocks": len(stocks),
            "average_price": df['price'].mean(),
            "average_change_percent": df['change_percent'].mean(),
            "positive_changes": len(df[df['change_percent'] > 0]),
            "negative_changes": len(df[df['change_percent'] < 0]),
            "top_gainers": df.nlargest(5, 'change_percent')[['symbol', 'name', 'change_percent']].to_dict('records'),
            "top_losers": df.nsmallest(5, 'change_percent')[['symbol', 'name', 'change_percent']].to_dict('records'),
            "timestamp": get_current_timestamp()
        }
        
        summary_path = os.path.join(BASE_DIR, "analysis", f"stock_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📈 Stock summary saved to {summary_path}")
        
        return json_path, csv_path

# ======================
# NEWS ARTICLE CRAWLER
# ======================

class NewsArticleCrawler:
    """Crawler for Yahoo Finance news articles with stock symbol extraction"""
    
    def __init__(self):
        self.session = SESSION
        self.most_active_stocks_url = MOST_ACTIVE_STOCKS_URL
    
    def extract_stock_symbols_from_story_section(self, story_section):
        """
        Extract stock symbols directly from the story section HTML before redirecting to article URL
        """
        stock_symbols = set()
        
        try:
            # Find all ticker wrapper elements
            ticker_wrappers = story_section.find_all('span', class_=lambda x: x and 'ticker-wrapper' in x)
            
            for wrapper in ticker_wrappers:
                # Look for symbol span within the wrapper
                symbol_span = wrapper.find('span', class_=lambda x: x and 'symbol' in x)
                if symbol_span:
                    symbol_text = clean_text(symbol_span.text.strip())
                    if symbol_text:
                        # Clean and format the symbol (remove trailing spaces, convert to uppercase)
                        symbol = symbol_text.split()[0].strip().upper()
                        if symbol and symbol not in ['N/A', '--', '']:
                            # Remove suffixes like -USD, -SGD, etc. for cleaner stock symbols
                            base_symbol = re.split(r'[-_]', symbol)[0]
                            stock_symbols.add(base_symbol)
                            stock_symbols.add(symbol)  # Also keep the full symbol
            
            # Look for fin-streamer elements with data-symbol attribute as fallback
            fin_streamers = story_section.find_all('fin-streamer', {'data-symbol': True})
            for streamer in fin_streamers:
                symbol = streamer.get('data-symbol', '').strip().upper()
                if symbol and symbol not in ['N/A', '--', '']:
                    base_symbol = re.split(r'[-_]', symbol)[0]
                    stock_symbols.add(base_symbol)
                    stock_symbols.add(symbol)
            
        except Exception as e:
            logger.error(f"❌ Error extracting stock symbols from story section: {e}")
        
        return list(stock_symbols)
    
    def scrape_article_content(self, url):
        """Scrape full article content from a Yahoo Finance article URL"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title_tag = soup.find('h1', attrs={'data-testid': 'hero-header-title'})
            title = clean_text(title_tag.get_text()) if title_tag else ""

            # Extract publish date
            time_tag = soup.find('time')
            pub_date_raw = time_tag.get_text(strip=True) if time_tag else None
            pub_date_iso = None
            pub_date_utc = None
            if pub_date_raw:
                parsed = date_parse(pub_date_raw)
                if parsed:
                    # Assume SGT timezone if not specified
                    if parsed.tzinfo is None:
                        from dateutil import tz
                        parsed = parsed.replace(tzinfo=tz.gettz("Asia/Singapore"))
                    pub_date_utc = parsed.astimezone(timezone.utc)
                    pub_date_iso = pub_date_utc.isoformat()
                    pub_date_utc = int(pub_date_utc.timestamp())

            # Extract author
            author_tag = soup.find('div', attrs={'data-testid': 'author-profile'})
            author = clean_text(author_tag.get_text()) if author_tag else "Yahoo Finance"

            # Extract body content
            body_div = soup.find('div', attrs={'data-testid': 'article-body'})
            paragraphs = body_div.find_all('p') if body_div else []
            body_text = "\n".join([clean_text(p.get_text()) for p in paragraphs])
            clean_body = clean_text(body_text)

            return {
                "title": title,
                "author": author,
                "publish_date_iso": pub_date_iso,
                "publish_date_utc": pub_date_utc,
                "body": clean_body
            }
        except Exception as e:
            logger.warning(f"⚠️ Error scraping article {url}: {e}")
            return {
                "title": "",
                "author": "Yahoo Finance",
                "publish_date_iso": None,
                "publish_date_utc": None,
                "body": ""
            }
    
    def scrape_yahoo_finance_news_from_url(self, url, max_articles=20):
        """Scrape news from a specific Yahoo Finance topic URL"""
        logger.info(f"🔍 Scraping Yahoo Finance news from: {url}")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all story items using stable data-testid
            story_sections = soup.find_all('section', {'data-testid': 'storyitem'})
            logger.info(f"📰 Found {len(story_sections)} articles on {url}")

            posts = []
            chunks = []

            for idx, sec in enumerate(tqdm(story_sections[:max_articles], desc=f"Processing {urlparse(url).path.strip('/')}")):
                try:
                    # --- Extract from list view ---
                    headline_tag = sec.find('h3')
                    if not headline_tag:
                        continue
                    headline = clean_text(headline_tag.get_text())

                    link_tag = sec.find('a', href=True)
                    if not link_tag:
                        continue
                    href = link_tag['href']
                    article_url = urljoin("https://sg.finance.yahoo.com", href)

                    # Publisher & time (from list view)
                    publish_info = sec.find(class_=re.compile(r'publishing|time', re.I))
                    publisher = "Yahoo Finance"
                    publish_date_raw = None
                    if publish_info:
                        text = clean_text(publish_info.get_text())
                        parts = text.split('•')
                        if len(parts) >= 2:
                            publisher = parts[0].strip()
                            publish_date_raw = parts[1].strip()

                    # --- Extract stock symbols BEFORE redirecting to article URL ---
                    stock_symbols = self.extract_stock_symbols_from_story_section(sec)
                    if stock_symbols:
                        logger.info(f"✅ Found {len(stock_symbols)} stock symbols: {stock_symbols}")

                    # --- Scrape full article ---
                    article_data = self.scrape_article_content(article_url)
                    full_text = article_data["body"] or headline  # fallback to headline

                    # Generate post_id
                    post_id = str(uuid.uuid4())

                    # Unified publish date
                    pub_utc = article_data["publish_date_utc"] or int(datetime.now(timezone.utc).timestamp())
                    pub_iso = article_data["publish_date_iso"] or datetime.now(timezone.utc).isoformat()

                    # --- POST metadata (Reddit-style) ---
                    post = {
                        "post_id": post_id,
                        "created_utc": pub_utc,
                        "created_iso": pub_iso,
                        "subreddit": "finance",  # pseudo-subreddit
                        "author": article_data["author"],
                        "score": 0,  # not available
                        "num_comments": 0,  # not available
                        "text_raw": full_text,
                        "url": article_url,
                        "source": "Yahoo Finance",
                        "type": "article",
                        "headline": headline,
                        "publisher": publisher,
                        "stock_symbols": stock_symbols,
                        "most_active_stocks_url": self.most_active_stocks_url
                    }
                    posts.append(post)

                    # --- CHUNK metadata (NLP-ready) ---
                    emoji_count = len(re.findall(r'[^\w\s,.\'\"!?;:]', full_text))  # crude emoji/ symbol count
                    token_count = len(full_text.split())

                    chunk = {
                        "post_id": post_id,
                        "parent_id": None,
                        "chunk_idx": 0,
                        "chunk_total": 1,
                        "timestamp_utc": pub_utc,
                        "date_bucket": datetime.fromtimestamp(pub_utc, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "source": "Yahoo Finance",
                        "platform": "yahoo_finance",
                        "author": article_data["author"],
                        "tickers": stock_symbols,  # Using extracted symbols as tickers
                        "stock_symbols": stock_symbols,
                        "url_count": len(re.findall(r'https?://', full_text)),
                        "emoji_count": emoji_count,
                        "token_count": token_count,
                        "clean_text": full_text,
                        "lang": "en",
                        "most_active_stocks_url": self.most_active_stocks_url
                    }
                    chunks.append(chunk)

                    # Be polite
                    time.sleep(1.0)

                except Exception as e:
                    logger.error(f"❌ Error processing article {idx} from {url}: {e}")
                    continue

            return {"posts": posts, "chunks": chunks}
        
        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {e}")
            return {"posts": [], "chunks": []}
    
    def scrape_all_topics(self, max_articles_per_topic=5, max_topics=5):
        """Scrape news from all topic URLs"""
        logger.info(f"🚀 Starting news crawl from {min(max_topics, len(TOPIC_URLS))} topics...")
        
        all_posts = []
        all_chunks = []
        topics_processed = 0
        
        for url in TOPIC_URLS[:max_topics]:
            topics_processed += 1
            logger.info(f"📖 Processing topic {topics_processed}/{max_topics}: {url}")
            
            try:
                data = self.scrape_yahoo_finance_news_from_url(url, max_articles=max_articles_per_topic)
                
                if data["posts"]:
                    all_posts.extend(data["posts"])
                    all_chunks.extend(data["chunks"])
                    logger.info(f"✅ Successfully scraped {len(data['posts'])} articles from {url}")
                else:
                    logger.warning(f"⚠️ No articles found from {url}")
                
                # Be extra polite between topics
                time.sleep(2.0)
                
            except Exception as e:
                logger.error(f"❌ Error scraping {url}: {e}")
                continue
        
        logger.info(f"🎉 Total articles scraped: {len(all_posts)} from {topics_processed} topics")
        return {"posts": all_posts, "chunks": all_chunks}
    
    def save_news_data(self, data, filename=None):
        """Save news data to JSON files"""
        if not data["posts"]:
            logger.warning("⚠️ No news articles to save")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yahoo_finance_news_{timestamp}"
        
        # Save full data
        json_path = os.path.join(BASE_DIR, "news_information", f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save posts separately
        posts_path = os.path.join(BASE_DIR, "news_information", f"{filename}_posts.json")
        with open(posts_path, 'w', encoding='utf-8') as f:
            json.dump(data["posts"], f, ensure_ascii=False, indent=2)
        
        # Save chunks separately
        chunks_path = os.path.join(BASE_DIR, "news_information", f"{filename}_chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(data["chunks"], f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved news data to {json_path}, {posts_path}, and {chunks_path}")
        
        # Generate news summary
        total_articles = len(data["posts"])
        articles_with_stocks = sum(1 for post in data["posts"] if post.get("stock_symbols"))
        total_stock_mentions = sum(len(post.get("stock_symbols", [])) for post in data["posts"])
        
        unique_tickers = set()
        for post in data["posts"]:
            for symbol in post.get("stock_symbols", []):
                unique_tickers.add(symbol)
        
        summary = {
            "total_articles": total_articles,
            "articles_with_stock_symbols": articles_with_stocks,
            "total_stock_mentions": total_stock_mentions,
            "unique_tickers": list(unique_tickers),
            "most_mentioned_tickers": self.get_most_mentioned_tickers(data["posts"]),
            "topics_scraped": TOPIC_URLS[:MAX_TOPICS],
            "most_active_stocks_url": MOST_ACTIVE_STOCKS_URL,
            "timestamp": get_current_timestamp()
        }
        
        summary_path = os.path.join(BASE_DIR, "analysis", f"news_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📈 News summary saved to {summary_path}")
        
        return json_path
    
    def get_most_mentioned_tickers(self, posts, top_n=10):
        """Get the most mentioned tickers across all articles"""
        ticker_counts = {}
        
        for post in posts:
            for symbol in post.get("stock_symbols", []):
                ticker_counts[symbol] = ticker_counts.get(symbol, 0) + 1
        
        # Sort by count and get top N
        sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"ticker": ticker, "count": count} for ticker, count in sorted_tickers[:top_n]]

# ======================
# SENTIMENT ANALYSIS
# ======================

class SentimentAnalyzer:
    """Perform sentiment analysis using FinBERT"""
    
    def __init__(self):
        logger.info("🔄 Initializing FinBERT sentiment analyzer...")
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Check if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"{'GPU' if device == 0 else 'CPU'} will be used for sentiment analysis")
            
            # Load FinBERT model
            self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            
            logger.info("✅ FinBERT sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing FinBERT: {e}")
            logger.info("🔄 Falling back to VADER sentiment analysis")
            
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                
                self.analyzer = SentimentIntensityAnalyzer()
                self.use_vader = True
                logger.info("✅ VADER sentiment analyzer initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Error initializing VADER: {e}")
                logger.warning("⚠️ No sentiment analysis available. Will skip sentiment analysis.")
                self.use_vader = False
    
    def analyze_sentiment(self, text, max_length=512):
        """Analyze sentiment of text using FinBERT or VADER as fallback"""
        if not text or len(text.strip()) < 10:
            return {"label": "neutral", "score": 0.0}
        
        try:
            if hasattr(self, 'use_vader') and self.use_vader:
                # Use VADER
                scores = self.analyzer.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                
                return {"label": label, "score": round(compound, 4)}
            
            else:
                # Use FinBERT
                # Truncate text to max_length tokens
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    text = self.tokenizer.convert_tokens_to_string(tokens)
                
                result = self.sentiment_pipeline(text)[0]
                label = result['label'].lower()
                score = result['score']
                
                # Map FinBERT labels to standard format
                if label == "positive":
                    return {"label": "positive", "score": round(score, 4)}
                elif label == "negative":
                    return {"label": "negative", "score": round(score, 4)}
                else:
                    return {"label": "neutral", "score": round(score, 4)}
        
        except Exception as e:
            logger.warning(f"⚠️ Error during sentiment analysis: {e}")
            return {"label": "neutral", "score": 0.0}
    
    def analyze_articles(self, articles):
        """Analyze sentiment for multiple articles"""
        logger.info(f"🧠 Analyzing sentiment for {len(articles)} articles...")
        
        results = []
        for i, article in enumerate(tqdm(articles, desc="Analyzing sentiment")):
            text = article.get("text_raw", "") or article.get("clean_text", "") or article.get("headline", "")
            sentiment = self.analyze_sentiment(text)
            
            result = article.copy()
            result.update({
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "sentiment_analyzed_at": get_current_timestamp()
            })
            results.append(result)
            
            # Be gentle with the model
            if i > 0 and i % 10 == 0:
                time.sleep(1)
        
        logger.info("✅ Sentiment analysis completed")
        return results
    
    def save_sentiment_results(self, results, filename=None):
        """Save sentiment analysis results"""
        if not results:
            logger.warning("⚠️ No sentiment results to save")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}"
        
        filepath = os.path.join(BASE_DIR, "analysis", f"{filename}.json")
        
        # Organize results by sentiment
        positive = [r for r in results if r.get("sentiment_label") == "positive"]
        negative = [r for r in results if r.get("sentiment_label") == "negative"]
        neutral = [r for r in results if r.get("sentiment_label") == "neutral"]
        
        summary = {
            "total_articles": len(results),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "neutral_count": len(neutral),
            "positive_percentage": round(len(positive) / len(results) * 100, 2),
            "negative_percentage": round(len(negative) / len(results) * 100, 2),
            "neutral_percentage": round(len(neutral) / len(results) * 100, 2),
            "average_sentiment_score": round(np.mean([r.get("sentiment_score", 0) for r in results]), 4),
            "most_positive_article": max(results, key=lambda x: x.get("sentiment_score", 0)) if results else None,
            "most_negative_article": min(results, key=lambda x: x.get("sentiment_score", 0)) if results else None,
            "timestamp": get_current_timestamp(),
            "details": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Sentiment analysis results saved to {filepath}")
        logger.info(f"📊 Sentiment distribution: Positive: {len(positive)}, Negative: {len(negative)}, Neutral: {len(neutral)}")
        
        return filepath

# ======================
# DATA INTEGRATION AND ANALYSIS
# ======================

class DataIntegration:
    """Integrate stock data with news sentiment analysis"""
    
    def __init__(self):
        pass
    
    def integrate_stock_news(self, stocks, news_with_sentiment):
        """Integrate stock data with news sentiment"""
        logger.info("🔗 Integrating stock data with news sentiment...")
        
        # Create stock ticker mapping for quick lookup
        stock_map = {stock['symbol'].upper(): stock for stock in stocks}
        
        integrated_data = []
        
        for news_item in news_with_sentiment:
            stock_symbols = news_item.get("stock_symbols", [])
            
            if stock_symbols:
                for symbol in stock_symbols:
                    clean_symbol = symbol.upper().split('-')[0]  # Handle symbols like "BONK-USD"
                    
                    if clean_symbol in stock_map:
                        stock_data = stock_map[clean_symbol]
                        
                        integrated_item = {
                            "news_id": news_item.get("post_id"),
                            "stock_symbol": clean_symbol,
                            "company_name": stock_data.get("name", ""),
                            "current_price": stock_data.get("price", 0.0),
                            "daily_change_percent": stock_data.get("change_percent", 0.0),
                            "market_cap": stock_data.get("market_cap", ""),
                            "news_headline": news_item.get("headline", ""),
                            "news_url": news_item.get("url", ""),
                            "news_sentiment": news_item.get("sentiment_label", "neutral"),
                            "sentiment_score": news_item.get("sentiment_score", 0.0),
                            "published_at": news_item.get("created_iso", ""),
                            "analysis_timestamp": get_current_timestamp()
                        }
                        integrated_data.append(integrated_item)
        
        logger.info(f"✅ Integrated {len(integrated_data)} stock-news pairs")
        return integrated_data
    
    def analyze_correlations(self, integrated_data):
        """Analyze correlations between news sentiment and stock performance"""
        if not integrated_data:
            logger.warning("⚠️ No integrated data for correlation analysis")
            return None
        
        logger.info("📈 Analyzing correlations between news sentiment and stock performance...")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(integrated_data)
        
        if df.empty:
            logger.warning("⚠️ Empty DataFrame for correlation analysis")
            return None
        
        # Group by stock symbol and calculate average sentiment and performance
        symbol_groups = df.groupby('stock_symbol').agg({
            'sentiment_score': 'mean',
            'daily_change_percent': 'mean',
            'news_sentiment': lambda x: x.value_counts().index[0]  # Most common sentiment
        }).reset_index()
        
        # Calculate correlation between sentiment score and daily change
        if len(symbol_groups) > 1:
            correlation = symbol_groups['sentiment_score'].corr(symbol_groups['daily_change_percent'])
        else:
            correlation = 0.0
        
        # Identify significant correlations
        positive_correlations = symbol_groups[
            (symbol_groups['sentiment_score'] > 0.1) & 
            (symbol_groups['daily_change_percent'] > 0)
        ]
        
        negative_correlations = symbol_groups[
            (symbol_groups['sentiment_score'] < -0.1) & 
            (symbol_groups['daily_change_percent'] < 0)
        ]
        
        analysis = {
            "overall_correlation": round(correlation, 4),
            "total_stock_news_pairs": len(integrated_data),
            "unique_stocks_analyzed": len(symbol_groups),
            "positive_correlations": {
                "count": len(positive_correlations),
                "stocks": positive_correlations.to_dict('records')
            },
            "negative_correlations": {
                "count": len(negative_correlations),
                "stocks": negative_correlations.to_dict('records')
            },
            "sentiment_distribution": {
                "positive": len(df[df['news_sentiment'] == 'positive']),
                "negative": len(df[df['news_sentiment'] == 'negative']),
                "neutral": len(df[df['news_sentiment'] == 'neutral'])
            },
            "timestamp": get_current_timestamp()
        }
        
        logger.info(f"📊 Overall sentiment-price correlation: {correlation:.4f}")
        logger.info(f"📈 Positive correlations found: {len(positive_correlations)} stocks")
        logger.info(f"📉 Negative correlations found: {len(negative_correlations)} stocks")
        
        return analysis
    
    def save_analysis(self, integrated_data, correlation_results, filename=None):
        """Save analysis results"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_news_analysis_{timestamp}"
        
        # Save integrated data
        integrated_path = os.path.join(BASE_DIR, "analysis", f"{filename}_integrated.json")
        with open(integrated_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f, ensure_ascii=False, indent=2)
        
        # Save correlation results
        if correlation_results:
            correlation_path = os.path.join(BASE_DIR, "analysis", f"{filename}_correlations.json")
            with open(correlation_path, 'w', encoding='utf-8') as f:
                json.dump(correlation_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Correlation analysis saved to {correlation_path}")
        
        logger.info(f"💾 Integrated data saved to {integrated_path}")
        return integrated_path

# ======================
# MAIN PIPELINE
# ======================

def run_pipeline():
    """Main pipeline execution function"""
    logger.info("🚀 Starting Yahoo Finance Data Pipeline")
    logger.info(f"📁 Base directory: {BASE_DIR}")
    start_time = time.time()
    
    try:
        # Step 1: Crawl stock data
        logger.info("\n" + "="*50)
        logger.info("📊 STEP 1: CRAWLING STOCK DATA")
        logger.info("="*50)
        
        stock_crawler = StockDataCrawler()
        stocks = stock_crawler.crawl_most_active_stocks(max_stocks=MAX_STOCKS)
        
        if not stocks:
            logger.error("❌ No stock data crawled. Cannot proceed with pipeline.")
            return
        
        stock_crawler.save_stock_data(stocks)
        
        # Step 2: Crawl news articles
        logger.info("\n" + "="*50)
        logger.info("📰 STEP 2: CRAWLING NEWS ARTICLES")
        logger.info("="*50)
        
        news_crawler = NewsArticleCrawler()
        news_data = news_crawler.scrape_all_topics(
            max_articles_per_topic=MAX_ARTICLES_PER_TOPIC,
            max_topics=MAX_TOPICS
        )
        
        if not news_data["posts"]:
            logger.error("❌ No news articles crawled. Cannot proceed with pipeline.")
            return
        
        news_crawler.save_news_data(news_data)
        
        # Step 3: Sentiment analysis
        logger.info("\n" + "="*50)
        logger.info("🧠 STEP 3: SENTIMENT ANALYSIS")
        logger.info("="*50)
        
        sentiment_analyzer = SentimentAnalyzer()
        news_with_sentiment = sentiment_analyzer.analyze_articles(news_data["posts"])
        
        if news_with_sentiment:
            sentiment_analyzer.save_sentiment_results(news_with_sentiment)
        
        # Step 4: Data integration and analysis
        logger.info("\n" + "="*50)
        logger.info("🔗 STEP 4: DATA INTEGRATION AND ANALYSIS")
        logger.info("="*50)
        
        data_integration = DataIntegration()
        integrated_data = data_integration.integrate_stock_news(stocks, news_with_sentiment)
        
        if integrated_data:
            correlation_results = data_integration.analyze_correlations(integrated_data)
            data_integration.save_analysis(integrated_data, correlation_results)
        
        # Generate final summary
        logger.info("\n" + "="*50)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        elapsed_time = time.time() - start_time
        summary = {
            "pipeline_execution_time_seconds": round(elapsed_time, 2),
            "stocks_crawled": len(stocks),
            "news_articles_crawled": len(news_data["posts"]),
            "articles_with_sentiment": len(news_with_sentiment) if news_with_sentiment else 0,
            "stock_news_integrations": len(integrated_data) if integrated_data else 0,
            "completed_at": get_current_timestamp(),
            "directories": {
                "base": BASE_DIR,
                "stock_data": os.path.join(BASE_DIR, "stock_information"),
                "news_data": os.path.join(BASE_DIR, "news_information"),
                "analysis": os.path.join(BASE_DIR, "analysis")
            }
        }
        
        summary_path = os.path.join(BASE_DIR, "analysis", f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 Pipeline summary saved to {summary_path}")
        logger.info(f"⏰ Total execution time: {elapsed_time:.2f} seconds")
        
        # Print key insights
        if correlation_results:
            logger.info("\n" + "="*50)
            logger.info("💡 KEY INSIGHTS")
            logger.info("="*50)
            logger.info(f"📊 Overall sentiment-price correlation: {correlation_results['overall_correlation']:.4f}")
            logger.info(f"📈 Stocks with positive sentiment & positive performance: {correlation_results['positive_correlations']['count']}")
            logger.info(f"📉 Stocks with negative sentiment & negative performance: {correlation_results['negative_correlations']['count']}")
    
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save error information
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": get_current_timestamp(),
            "step_failed": "unknown"
        }
        
        error_path = os.path.join(BASE_DIR, "logs", f"pipeline_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"❌ Error details saved to {error_path}")

# ======================
# COMMAND LINE INTERFACE
# ======================

def main():
    """Command line interface for the pipeline"""
    import argparse
    import logging

    global MAX_STOCKS, MAX_ARTICLES_PER_TOPIC, MAX_TOPICS

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Yahoo Finance Data Pipeline')
    parser.add_argument('--max-stocks', type=int, default=200, help='Maximum number of stocks to crawl')
    parser.add_argument('--max-articles-per-topic', type=int, default=15, help='Maximum articles per topic')
    parser.add_argument('--max-topics', type=int, default=8, help='Maximum number of topics to crawl')
    parser.add_argument('--test-mode', action='store_true', default=False, help='Run in test mode with reduced limits')
    parser.add_argument('--skip-sentiment', action='store_true', default=False, help='Skip sentiment analysis')

    # <-- change here to parse_known_args() to avoid errors in Colab/Jupyter
    args, unknown = parser.parse_known_args()

    if args.test_mode:
        logger.info("🧪 Running in test mode with reduced limits")
        MAX_STOCKS = 20
        MAX_ARTICLES_PER_TOPIC = 3
        MAX_TOPICS = 3
    else:
        MAX_STOCKS = args.max_stocks
        MAX_ARTICLES_PER_TOPIC = args.max_articles_per_topic
        MAX_TOPICS = args.max_topics

    logger.info(f"⚙️ Configuration: Max stocks={MAX_STOCKS}, Max articles per topic={MAX_ARTICLES_PER_TOPIC}, Max topics={MAX_TOPICS}")

    # You need to define or import run_pipeline() somewhere in your code
    run_pipeline()

if __name__ == "__main__":
    main()