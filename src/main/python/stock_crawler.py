import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
from datetime import datetime, timezone
from urllib.parse import urljoin
from tqdm import tqdm
import logging
from config import USER_AGENT, MOST_ACTIVE_STOCKS_URL, BASE_DIR
from utils import clean_text, safe_float, get_current_timestamp, save_json

class StockDataCrawler:
    """Crawler for Yahoo Finance most active stocks data"""
    
    def __init__(self, session=None):
        self.logger = logging.getLogger(__name__)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.base_url = "https://finance.yahoo.com"
    
    def parse_stock_row(self, row):
        """Parse a single stock row from the most active stocks table"""
        try:
            # Get symbol
            symbol = "N/A"
            symbol_span = row.find('span', class_=lambda x: x and 'symbol' in x)
            if symbol_span:
                symbol_text = clean_text(symbol_span.text.strip())
                if symbol_text:
                    symbol = symbol_text.split()[0]
            
            # Get company name
            name = "N/A"
            name_cell = row.find('td', {'data-testid-cell': 'companyshortname.raw'})
            if name_cell:
                name_div = name_cell.find('div', class_=lambda x: x and 'companyName' in x)
                if name_div:
                    name = clean_text(name_div.text.strip())
                else:
                    name = clean_text(name_cell.text.strip())
            
            # Get price
            price = "N/A"
            price_streamer = row.find('fin-streamer', {'data-field': 'regularMarketPrice'})
            if price_streamer:
                price_text = clean_text(price_streamer.text.strip())
                if price_text:
                    price = price_text
            
            # Get change amount
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
            
            # Get change percent
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
            
            # Get volume
            volume = "N/A"
            volume_streamer = row.find('fin-streamer', {'data-field': 'regularMarketVolume'})
            if volume_streamer:
                volume_text = clean_text(volume_streamer.text.strip())
                if volume_text:
                    volume = volume_text
            
            # Get market cap
            market_cap = "N/A"
            market_cap_streamer = row.find('fin-streamer', {'data-field': 'marketCap'})
            if market_cap_streamer:
                market_cap_text = clean_text(market_cap_streamer.text.strip())
                if market_cap_text:
                    market_cap = market_cap_text
            
            # Get P/E ratio
            pe_ratio = "N/A"
            pe_cell = row.find('td', {'data-testid-cell': 'peratio.lasttwelvemonths'})
            if pe_cell:
                pe_text = clean_text(pe_cell.text.strip())
                if pe_text and pe_text not in ['--', '-', 'N/A']:
                    pe_ratio = pe_text
            
            # Get 52-week change
            week52_change = "N/A"
            week52_streamer = row.find('fin-streamer', {'data-field': 'fiftyTwoWeekChangePercent'})
            if week52_streamer:
                week52_span = week52_streamer.find('span', class_=lambda x: x and ('txt-positive' in x or 'txt-negative' in x))
                if week52_span:
                    week52_text = clean_text(week52_span.text.strip())
                    if week52_text:
                        week52_change = week52_text
            
            # Get 52-week range
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
            self.logger.error(f"Error parsing stock row: {e}")
            return None
    
    def crawl_most_active_stocks(self, max_stocks=200):
        """Crawl most active stocks from Yahoo Finance"""
        self.logger.info(f"Starting to crawl most active stocks (target: {max_stocks})...")
        
        all_stocks = []
        start = 0
        count = 25
        
        while len(all_stocks) < max_stocks:
            url = MOST_ACTIVE_STOCKS_URL.format(start=start, count=count)
            self.logger.info(f"Fetching URL: {url}")
            
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find rows
                rows = soup.find_all('tr', {'data-testid': 'data-table-v2-row'})
                
                self.logger.info(f"Found {len(rows)} rows with data-testid='data-table-v2-row'")
                
                if not rows:
                    self.logger.warning("No rows found with the expected structure. Trying alternative selectors...")
                    rows = soup.find_all('tr', class_=lambda x: x and 'row' in x)
                    self.logger.info(f"Found {len(rows)} rows with class containing 'row'")
                
                if not rows:
                    self.logger.error("No rows found at all. Stopping crawl.")
                    break
                
                new_stocks = []
                
                for i, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 5:
                        continue
                    
                    stock_data = self.parse_stock_row(row)
                    if stock_data:
                        new_stocks.append(stock_data)
                        self.logger.debug(f"Row {i} ({stock_data['symbol']}): ${stock_data['price']:.2f} ({stock_data['change_percent']:+.2f}%)")
                    
                    if len(new_stocks) >= count:
                        break
                
                self.logger.info(f"Processed {len(rows)} rows, found {len(new_stocks)} valid stocks")
                
                if not new_stocks:
                    self.logger.warning("No valid stocks found in this batch. Stopping crawl.")
                    break
                
                # Remove duplicates by symbol
                unique_stocks = []
                seen_symbols = set()
                for stock in new_stocks:
                    if stock['symbol'] not in seen_symbols:
                        seen_symbols.add(stock['symbol'])
                        unique_stocks.append(stock)
                
                all_stocks.extend(unique_stocks)
                self.logger.info(f"Found {len(unique_stocks)} unique stocks in this batch. Total: {len(all_stocks)}")
                
                if len(new_stocks) < count * 0.5:
                    self.logger.info("Reached end of available data. Stopping crawl.")
                    break
                
                start += count
                time.sleep(2)
                
                if len(all_stocks) >= max_stocks:
                    all_stocks = all_stocks[:max_stocks]
                    self.logger.info(f"Reached target of {max_stocks} stocks. Stopping crawl.")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error crawling stocks from {url}: {e}")
                break
        
        self.logger.info(f"Successfully crawled {len(all_stocks)} most active stocks")
        return all_stocks
    
    def save_stock_data(self, stocks, filename=None):
        """Save stock data to JSON and CSV files"""
        if not stocks:
            self.logger.warning("No stocks to save")
            return None, None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"most_active_stocks_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(BASE_DIR, "stock_information", f"{filename}.json")
        save_json(stocks, json_path)
        
        # Save as CSV
        df = pd.DataFrame(stocks)
        csv_path = os.path.join(BASE_DIR, "stock_information", f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Saved stock data to {json_path} and {csv_path}")
        
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
        save_json(summary, summary_path)
        
        self.logger.info(f"Stock summary saved to {summary_path}")
        
        return json_path, csv_path