import requests
from bs4 import BeautifulSoup
import time
import re
import uuid
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from dateparser import parse as date_parse
from tqdm import tqdm
import logging
from config import USER_AGENT, TOPIC_URLS, MOST_ACTIVE_STOCKS_URL, BASE_DIR
from utils import clean_text, get_current_timestamp, save_json
import os
class NewsArticleCrawler:
    """Crawler for Yahoo Finance news articles with stock symbol extraction"""
    
    def __init__(self, session=None):
        self.logger = logging.getLogger(__name__)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.most_active_stocks_url = MOST_ACTIVE_STOCKS_URL
    
    def extract_stock_symbols_from_story_section(self, story_section):
        """Extract stock symbols directly from the story section HTML"""
        stock_symbols = set()
        
        try:
            # Find all ticker wrapper elements
            ticker_wrappers = story_section.find_all('span', class_=lambda x: x and 'ticker-wrapper' in x)
            
            for wrapper in ticker_wrappers:
                symbol_span = wrapper.find('span', class_=lambda x: x and 'symbol' in x)
                if symbol_span:
                    symbol_text = clean_text(symbol_span.text.strip())
                    if symbol_text:
                        symbol = symbol_text.split()[0].strip().upper()
                        if symbol and symbol not in ['N/A', '--', '']:
                            base_symbol = re.split(r'[-_]', symbol)[0]
                            stock_symbols.add(base_symbol)
                            stock_symbols.add(symbol)
            
            # Look for fin-streamer elements with data-symbol attribute
            fin_streamers = story_section.find_all('fin-streamer', {'data-symbol': True})
            for streamer in fin_streamers:
                symbol = streamer.get('data-symbol', '').strip().upper()
                if symbol and symbol not in ['N/A', '--', '']:
                    base_symbol = re.split(r'[-_]', symbol)[0]
                    stock_symbols.add(base_symbol)
                    stock_symbols.add(symbol)
            
        except Exception as e:
            self.logger.error(f"Error extracting stock symbols from story section: {e}")
        
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
            self.logger.warning(f"Error scraping article {url}: {e}")
            return {
                "title": "",
                "author": "Yahoo Finance",
                "publish_date_iso": None,
                "publish_date_utc": None,
                "body": ""
            }
    
    def scrape_yahoo_finance_news_from_url(self, url, max_articles=20):
        """Scrape news from a specific Yahoo Finance topic URL"""
        self.logger.info(f"Scraping Yahoo Finance news from: {url}")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all story items
            story_sections = soup.find_all('section', {'data-testid': 'storyitem'})
            self.logger.info(f"Found {len(story_sections)} articles on {url}")

            posts = []
            chunks = []

            for idx, sec in enumerate(tqdm(story_sections[:max_articles], desc=f"Processing {urlparse(url).path.strip('/')}")):
                try:
                    # Extract from list view
                    headline_tag = sec.find('h3')
                    if not headline_tag:
                        continue
                    headline = clean_text(headline_tag.get_text())

                    link_tag = sec.find('a', href=True)
                    if not link_tag:
                        continue
                    href = link_tag['href']
                    article_url = urljoin("https://sg.finance.yahoo.com", href)

                    # Publisher & time
                    publish_info = sec.find(class_=re.compile(r'publishing|time', re.I))
                    publisher = "Yahoo Finance"
                    publish_date_raw = None
                    if publish_info:
                        text = clean_text(publish_info.get_text())
                        parts = text.split('•')
                        if len(parts) >= 2:
                            publisher = parts[0].strip()
                            publish_date_raw = parts[1].strip()

                    # Extract stock symbols BEFORE redirecting
                    stock_symbols = self.extract_stock_symbols_from_story_section(sec)
                    if stock_symbols:
                        self.logger.info(f"Found {len(stock_symbols)} stock symbols: {stock_symbols}")

                    # Scrape full article
                    article_data = self.scrape_article_content(article_url)
                    full_text = article_data["body"] or headline

                    # Generate post_id
                    post_id = str(uuid.uuid4())

                    # Unified publish date
                    pub_utc = article_data["publish_date_utc"] or int(datetime.now(timezone.utc).timestamp())
                    pub_iso = article_data["publish_date_iso"] or datetime.now(timezone.utc).isoformat()

                    # POST metadata
                    post = {
                        "post_id": post_id,
                        "created_utc": pub_utc,
                        "created_iso": pub_iso,
                        "subreddit": "finance",
                        "author": article_data["author"],
                        "score": 0,
                        "num_comments": 0,
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

                    # CHUNK metadata
                    emoji_count = len(re.findall(r'[^\w\s,.\'\"!?;:]', full_text))
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
                        "tickers": stock_symbols,
                        "stock_symbols": stock_symbols,
                        "url_count": len(re.findall(r'https?://', full_text)),
                        "emoji_count": emoji_count,
                        "token_count": token_count,
                        "clean_text": full_text,
                        "lang": "en",
                        "most_active_stocks_url": self.most_active_stocks_url
                    }
                    chunks.append(chunk)

                    time.sleep(1.0)

                except Exception as e:
                    self.logger.error(f"Error processing article {idx} from {url}: {e}")
                    continue

            return {"posts": posts, "chunks": chunks}
        
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return {"posts": [], "chunks": []}
    
    def scrape_all_topics(self, max_articles_per_topic=5, max_topics=5):
        """Scrape news from all topic URLs"""
        self.logger.info(f"Starting news crawl from {min(max_topics, len(TOPIC_URLS))} topics...")
        
        all_posts = []
        all_chunks = []
        topics_processed = 0
        
        for url in TOPIC_URLS[:max_topics]:
            topics_processed += 1
            self.logger.info(f"Processing topic {topics_processed}/{max_topics}: {url}")
            
            try:
                data = self.scrape_yahoo_finance_news_from_url(url, max_articles=max_articles_per_topic)
                
                if data["posts"]:
                    all_posts.extend(data["posts"])
                    all_chunks.extend(data["chunks"])
                    self.logger.info(f"Successfully scraped {len(data['posts'])} articles from {url}")
                else:
                    self.logger.warning(f"No articles found from {url}")
                
                time.sleep(2.0)
                
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {e}")
                continue
        
        self.logger.info(f"Total articles scraped: {len(all_posts)} from {topics_processed} topics")
        return {"posts": all_posts, "chunks": all_chunks}
    
    def save_news_data(self, data, filename=None):
        """Save news data to JSON files"""
        if not data["posts"]:
            self.logger.warning("No news articles to save")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yahoo_finance_news_{timestamp}"
        
        # Save full data
        json_path = os.path.join(BASE_DIR, "news_information", f"{filename}.json")
        save_json(data, json_path)
        
        # Save posts separately
        posts_path = os.path.join(BASE_DIR, "news_information", f"{filename}_posts.json")
        save_json(data["posts"], posts_path)
        
        # Save chunks separately
        chunks_path = os.path.join(BASE_DIR, "news_information", f"{filename}_chunks.json")
        save_json(data["chunks"], chunks_path)
        
        self.logger.info(f"Saved news data to {json_path}, {posts_path}, and {chunks_path}")
        
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
            # "topics_scraped": TOPIC_URLS[:max_topics],
            "most_active_stocks_url": self.most_active_stocks_url,
            "timestamp": get_current_timestamp()
        }
        
        summary_path = os.path.join(BASE_DIR, "analysis", f"news_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_json(summary, summary_path)
        
        self.logger.info(f"News summary saved to {summary_path}")
        
        return json_path
    
    def get_most_mentioned_tickers(self, posts, top_n=10):
        """Get the most mentioned tickers across all articles"""
        ticker_counts = {}
        
        for post in posts:
            for symbol in post.get("stock_symbols", []):
                ticker_counts[symbol] = ticker_counts.get(symbol, 0) + 1
        
        sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"ticker": ticker, "count": count} for ticker, count in sorted_tickers[:top_n]]