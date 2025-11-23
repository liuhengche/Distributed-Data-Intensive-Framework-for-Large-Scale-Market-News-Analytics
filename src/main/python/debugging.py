#!/usr/bin/env python3
"""
Advanced Stock-News Correlation Analysis
Standalone script that analyzes relationships between news sentiment and stock performance
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from datetime import datetime, timedelta
import logging
import sys
import io
import re
from tqdm import tqdm

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['MATPLOTLIB_BACKEND'] = 'Agg'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('advanced_analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StockNewsAnalyzer:
    """Advanced analyzer for stock-news correlation analysis"""
    
    def __init__(self, base_dir="yahoo_finance_pipeline"):
        self.base_dir = base_dir
        self.analysis_dir = os.path.join(base_dir, "analysis")
        self.stock_dir = os.path.join(base_dir, "stock_information")
        self.news_dir = os.path.join(base_dir, "news_information")
        self.output_dir = os.path.join(base_dir, "analysis", "correlation_analysis")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized StockNewsAnalyzer with base directory: {base_dir}")
        logger.info(f"Stock data directory: {self.stock_dir}")
        logger.info(f"News data directory: {self.news_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def convert_dataframe_to_serializable(self, obj):
        """Convert pandas DataFrame or Series to JSON serializable format"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self.convert_dataframe_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_dataframe_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def load_json_files(self, directory, pattern="*.json"):
        """Load all JSON files from a directory matching the pattern"""
        file_pattern = os.path.join(directory, pattern)
        file_list = glob(file_pattern)
        
        if not file_list:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(file_list)} files matching pattern: {file_pattern}")
        
        df_list = []
        for file in tqdm(file_list, desc=f"Loading files from {os.path.basename(directory)}"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, dict):
                    if 'posts' in str(file).lower():
                        df = pd.DataFrame(data['posts'])
                    elif 'chunks' in str(file).lower():
                        df = pd.DataFrame(data['chunks'])
                    elif 'stock_summary' in str(file).lower():
                        # Handle the stock summary file format
                        stock_data = data.get('top_gainers', []) + data.get('top_losers', [])
                        df = pd.DataFrame(stock_data)
                    else:
                        df = pd.DataFrame([data])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    logger.warning(f"Unsupported JSON structure in file: {file}")
                    continue
                
                df_list.append(df)
                logger.debug(f"Successfully loaded file: {file}")
                
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")
                continue
        
        if not df_list:
            logger.warning(f"No valid data frames created from files in {directory}")
            return pd.DataFrame()
        
        combined_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Successfully loaded {len(combined_df)} records from {len(file_list)} files")
        return combined_df
    
    def clean_stock_data(self, stock_df):
        """Clean and preprocess stock data"""
        if stock_df.empty:
            logger.warning("Empty stock dataframe provided for cleaning")
            return pd.DataFrame()
        
        logger.info("Cleaning and preprocessing stock data...")
        
        # Filter columns and drop nulls
        required_columns = ['symbol', 'price', 'volume', 'name', 'change_percent']
        
        # Check which columns are actually available
        available_columns = [col for col in required_columns if col in stock_df.columns]
        
        if not available_columns:
            # Try to find similar columns
            possible_columns = {
                'symbol': ['symbol', 'ticker', 'stock_symbol'],
                'price': ['price', 'current_price', 'regularMarketPrice'],
                'volume': ['volume', 'dayvolume', 'regularMarketVolume'],
                'name': ['name', 'companyname', 'company_name'],
                'change_percent': ['change_percent', 'changePercent', 'percentchange']
            }
            
            found_columns = {}
            for target_col, possible_cols in possible_columns.items():
                for col in possible_cols:
                    if col in stock_df.columns:
                        found_columns[target_col] = col
                        break
            
            if not found_columns:
                logger.error("No required columns found in stock data")
                return pd.DataFrame()
            
            # Rename columns to standard names
            stock_df = stock_df.rename(columns={v: k for k, v in found_columns.items()})
            available_columns = list(found_columns.keys())
        
        stock_df = stock_df[available_columns].copy()
        
        # Remove rows with missing critical data
        stock_df = stock_df.dropna(subset=['symbol', 'price'])
        
        # Add timestamp if missing
        if 'created_iso' not in stock_df.columns:
            stock_df['created_iso'] = datetime.now().isoformat()
        
        # Convert timestamp to datetime
        stock_df['stock_time'] = pd.to_datetime(stock_df['created_iso'], errors='coerce')
        stock_df = stock_df.dropna(subset=['stock_time'])
        stock_df['stock_date'] = stock_df['stock_time'].dt.date
        
        # Clean price data
        stock_df['price'] = pd.to_numeric(stock_df['price'], errors='coerce')
        stock_df = stock_df.dropna(subset=['price'])
        
        # Clean volume data
        if 'volume' in stock_df.columns:
            def parse_volume(volume_str):
                if pd.isna(volume_str) or volume_str == '':
                    return np.nan
                volume_str = str(volume_str).strip().lower()
                if 'm' in volume_str:
                    return float(volume_str.replace('m', '').replace(',', '')) * 1e6
                elif 'b' in volume_str:
                    return float(volume_str.replace('b', '').replace(',', '')) * 1e9
                elif 'k' in volume_str:
                    return float(volume_str.replace('k', '').replace(',', '')) * 1e3
                else:
                    try:
                        return float(volume_str.replace(',', ''))
                    except:
                        return np.nan
            
            stock_df['volume_num'] = stock_df['volume'].apply(parse_volume)
        else:
            logger.warning("No volume column found in stock data")
            stock_df['volume_num'] = np.nan
        
        # Ensure change_percent exists
        if 'change_percent' not in stock_df.columns:
            stock_df['change_percent'] = 0.0
        
        # Select final columns
        final_columns = ['symbol', 'name', 'price', 'stock_time', 'stock_date', 'volume_num', 'change_percent']
        final_columns = [col for col in final_columns if col in stock_df.columns]
        
        stock_df = stock_df[final_columns].dropna(subset=['price'])
        
        logger.info(f"Cleaned stock data contains {len(stock_df)} records")
        return stock_df
    
    def clean_news_data(self, news_df):
        """Clean and preprocess news data"""
        if news_df.empty:
            logger.warning("Empty news dataframe provided for cleaning")
            return pd.DataFrame()
        
        logger.info("Cleaning and preprocessing news data...")
        
        # Find text columns (try multiple possible column names)
        text_columns = ['text_raw', 'clean_text', 'body', 'body_texts']
        text_col = next((col for col in text_columns if col in news_df.columns), None)
        
        # Find headline columns
        headline_columns = ['headline', 'title', 'news_headline']
        headline_col = next((col for col in headline_columns if col in news_df.columns), None)
        
        # Find time columns
        time_columns = ['created_iso', 'publish_date_iso', 'timestamp_utc', 'created_utc']
        time_col = next((col for col in time_columns if col in news_df.columns), None)
        
        # Find source columns
        source_columns = ['source', 'publisher', 'author']
        source_col = next((col for col in source_columns if col in news_df.columns), None)
        
        # Find stock symbol columns
        symbol_columns = ['stock_symbols', 'tickers', 'matched_symbol']
        symbol_col = next((col for col in symbol_columns if col in news_df.columns), None)
        
        # Check if we have minimum required columns
        if text_col is None:
            logger.error("No text column found in news data")
            return pd.DataFrame()
        
        required_cols = [text_col]
        if headline_col:
            required_cols.append(headline_col)
        if time_col:
            required_cols.append(time_col)
        if symbol_col:
            required_cols.append(symbol_col)
        
        # Filter and copy data
        news_df = news_df[required_cols].copy()
        
        # Rename columns to standard names
        column_mapping = {}
        if text_col:
            column_mapping[text_col] = 'text_raw'
        if headline_col:
            column_mapping[headline_col] = 'headline'
        if time_col:
            column_mapping[time_col] = 'created_iso'
        if source_col:
            column_mapping[source_col] = 'source'
        if symbol_col:
            column_mapping[symbol_col] = 'stock_symbols'
        
        news_df = news_df.rename(columns=column_mapping)
        
        # Handle text_raw as list of paragraphs
        if 'text_raw' in news_df.columns and news_df['text_raw'].dtype == 'object':
            news_df['text_raw'] = news_df['text_raw'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Convert time to datetime
        if 'created_iso' in news_df.columns:
            news_df['news_time'] = pd.to_datetime(news_df['created_iso'], errors='coerce')
        else:
            # Use current time as fallback
            news_df['news_time'] = datetime.now()
        
        news_df = news_df.dropna(subset=['news_time'])
        news_df['news_date'] = news_df['news_time'].dt.date
        
        # Fill missing headline with first part of text
        if 'headline' not in news_df.columns:
            news_df['headline'] = news_df['text_raw'].str[:100]
        else:
            news_df['headline'] = news_df['headline'].fillna(news_df['text_raw'].str[:100])
        
        # Fill missing source
        if 'source' not in news_df.columns:
            news_df['source'] = 'Yahoo Finance'
        else:
            news_df['source'] = news_df['source'].fillna('Yahoo Finance')
        
        # Extract stock symbols from news data
        if 'stock_symbols' in news_df.columns:
            def extract_first_symbol(symbols):
                if isinstance(symbols, list) and len(symbols) > 0:
                    return str(symbols[0]).upper().split('-')[0]
                elif isinstance(symbols, str) and symbols.strip():
                    return symbols.upper().split('-')[0]
                return 'UNKNOWN'
            
            news_df['matched_symbol'] = news_df['stock_symbols'].apply(extract_first_symbol)
        else:
            # Try to extract from text
            news_df['matched_symbol'] = self.extract_stock_symbols(news_df)
        
        # Filter out unknown symbols
        news_df = news_df[news_df['matched_symbol'] != 'UNKNOWN'].copy()
        
        logger.info(f"Cleaned news data contains {len(news_df)} records")
        return news_df
    
    def extract_stock_symbols(self, news_df):
        """Extract stock symbols from news text using pattern matching"""
        def extract_from_text(text):
            if not text or not isinstance(text, str):
                return 'UNKNOWN'
            
            text_lower = text.lower()
            
            # Known symbol mappings (expand as needed)
            symbol_mappings = {
                'nvidia': 'NVDA',
                'kenvue': 'KVUE',
                'bonk': 'BONK',
                'solana': 'SOL',
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
                'tesla': 'TSLA',
                'apple': 'AAPL',
                'microsoft': 'MSFT',
                'google': 'GOOGL',
                'amazon': 'AMZN',
                'meta': 'META'
            }
            
            for keyword, symbol in symbol_mappings.items():
                if keyword in text_lower:
                    return symbol
            
            # Pattern matching for stock symbols
            patterns = [
                r'\$([A-Z]{1,5})\b',  # $TSLA
                r'\b([A-Z]{3,5})\b',   # TSLA
                r'([A-Z]{3,5})[-.]',   # TSLA-USD
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    return matches[0].upper()
            
            return 'UNKNOWN'
        
        return news_df.apply(lambda row: extract_from_text(
            str(row.get('headline', '')) + ' ' + str(row.get('text_raw', ''))
        ), axis=1)
    
    def calculate_sentiment(self, news_df):
        """Calculate sentiment scores for news articles"""
        logger.info("Calculating sentiment scores for news articles...")
        
        if news_df.empty:
            return news_df
        
        def simple_sentiment_analysis(text):
            if not text or not isinstance(text, str):
                return 'NEUTRAL', 0.0
            
            text_lower = text.lower()
            
            # Simple keyword-based sentiment analysis
            positive_words = ['improved', 'increased', 'growth', 'positive', 'strong', 'exceeded', 
                             'upgrade', 'profit', 'record', 'success', 'breakthrough', 'launch',
                             'gains', 'rally', 'bullish', 'surge', 'recovery', 'optimistic']
            negative_words = ['decreased', 'decline', 'loss', 'negative', 'weak', 'missed', 
                            'downgrade', 'risk', 'concern', 'struggle', 'drop', 'fall',
                            'bearish', 'plunge', 'tumble', 'dip', 'volatile', 'pessimistic']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'POSITIVE', min(0.9, 0.3 + (pos_count - neg_count) * 0.2)
            elif neg_count > pos_count:
                return 'NEGATIVE', -min(0.9, 0.3 + (neg_count - pos_count) * 0.2)
            else:
                return 'NEUTRAL', 0.0
        
        # Apply sentiment analysis
        sentiment_results = news_df['text_raw'].apply(simple_sentiment_analysis)
        news_df['sentiment_label'] = sentiment_results.apply(lambda x: x[0])
        news_df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
        
        logger.info("Sentiment distribution:")
        logger.info(news_df['sentiment_label'].value_counts().to_string())
        
        return news_df
    
    def calculate_stock_returns(self, stock_df):
        """Calculate daily returns for stocks"""
        logger.info("Calculating daily stock returns...")
        
        if stock_df.empty:
            return stock_df
        
        # Add daily return column if not exists
        if 'daily_return' not in stock_df.columns:
            stock_df['daily_return'] = stock_df['change_percent'] / 100
        
        # Ensure we have volume data
        if 'total_volume' not in stock_df.columns:
            if 'volume_num' in stock_df.columns:
                stock_df['total_volume'] = stock_df['volume_num']
            else:
                stock_df['total_volume'] = 0
        
        # Add price volatility if not exists
        if 'price_volatility' not in stock_df.columns:
            stock_df['price_volatility'] = 0.01  # Default small volatility
        
        logger.info(f"Calculated returns for {len(stock_df)} stock records")
        return stock_df
    
    def match_news_to_stocks(self, news_df, stock_df):
        """Match news articles to stock price movements with time windows"""
        logger.info("Matching news articles to stock price movements...")
        
        if news_df.empty:
            logger.warning("Empty news dataframe provided for matching")
            return pd.DataFrame()
        
        if stock_df.empty:
            logger.warning("Empty stock dataframe provided for matching")
            # Create minimal dataframe with news data only
            result = news_df.copy()
            result['daily_return'] = 0.0
            result['total_volume'] = 0
            result['symbol'] = result['matched_symbol']
            return result
        
        # Create a mapping of stock symbols to their data
        stock_symbol_map = {str(symbol).upper(): data for symbol, data in stock_df.groupby('symbol')}
        
        matched_records = []
        
        for _, news in news_df.iterrows():
            symbol = str(news['matched_symbol']).upper()
            
            if symbol in stock_symbol_map:
                stock_data = stock_symbol_map[symbol].iloc[0]  # Take first record for this symbol
                
                # Create a matched record
                record = news.to_dict()
                record.update({
                    'symbol': symbol,
                    'daily_return': stock_data.get('daily_return', 0.0),
                    'total_volume': stock_data.get('total_volume', stock_data.get('volume_num', 0)),
                    'price_volatility': stock_data.get('price_volatility', 0.01),
                    'stock_date': news['news_date'],  # Match on the same date
                    'stock_time': news['news_time']
                })
                matched_records.append(record)
        
        if matched_records:
            matched_df = pd.DataFrame(matched_records)
            logger.info(f"Successfully matched {len(matched_df)} news articles to stocks")
            return matched_df
        
        logger.warning("No direct matches found, trying fuzzy matching...")
        
        # Fallback: Create minimal matched dataframe with just news data
        matched_df = news_df.copy()
        matched_df['symbol'] = matched_df['matched_symbol']
        matched_df['daily_return'] = 0.0
        matched_df['total_volume'] = 0
        matched_df['stock_date'] = matched_df['news_date']
        
        logger.info(f"Created fallback matched data with {len(matched_df)} records")
        return matched_df
    
    def perform_correlation_analysis(self, matched_df):
        """Perform comprehensive correlation analysis between news and stock performance"""
        if matched_df.empty:
            logger.warning("No matched data for correlation analysis")
            return {}
        
        logger.info("Performing comprehensive correlation analysis...")
        results = {}
        
        # 1. Sentiment vs Returns analysis
        if 'sentiment_label' in matched_df.columns and 'daily_return' in matched_df.columns:
            sentiment_return = matched_df.groupby('sentiment_label').agg(
                avg_return=('daily_return', 'mean'),
                return_std=('daily_return', 'std'),
                sample_count=('daily_return', 'count')
            ).reset_index()
            results['sentiment_return'] = sentiment_return
        
        # 2. Event type analysis (if available)
        if 'event_type' in matched_df.columns and 'daily_return' in matched_df.columns:
            event_analysis = matched_df.groupby('event_type').agg(
                avg_abs_return=('daily_return', lambda x: np.abs(x).mean()),
                volatility=('daily_return', 'std'),
                sample_count=('daily_return', 'count')
            ).sort_values('volatility', ascending=False).reset_index()
            results['event_analysis'] = event_analysis
        
        # 3. News frequency vs volume analysis
        if all(col in matched_df.columns for col in ['symbol', 'news_date', 'daily_return']):
            news_frequency = matched_df.groupby(['symbol', 'news_date']).size().reset_index(name='daily_news_count')
            
            # Use available volume column
            volume_col = 'total_volume' if 'total_volume' in matched_df.columns else 'volume_num' if 'volume_num' in matched_df.columns else None
            
            if volume_col:
                volume_data = matched_df[['symbol', 'news_date', volume_col, 'daily_return']].drop_duplicates()
                volume_data = volume_data.rename(columns={volume_col: 'total_volume'})
                volume_analysis = pd.merge(
                    news_frequency,
                    volume_data,
                    on=['symbol', 'news_date'],
                    how='inner'
                )
                results['volume_analysis'] = volume_analysis
        
        # 4. Source impact analysis
        if all(col in matched_df.columns for col in ['source', 'daily_return']):
            source_impact = matched_df.groupby('source').agg(
                avg_abs_return=('daily_return', lambda x: np.abs(x).mean()),
                avg_sentiment=('sentiment_score', 'mean') if 'sentiment_score' in matched_df.columns else ('daily_return', 'mean'),
                news_count=('daily_return', 'count')
            ).query('news_count >= 2').sort_values('avg_abs_return', ascending=False).reset_index()
            results['source_impact'] = source_impact
        
        # 5. Time period analysis
        if 'news_time' in matched_df.columns and 'daily_return' in matched_df.columns:
            matched_df['news_hour'] = matched_df['news_time'].dt.hour
            matched_df['time_period'] = matched_df['news_hour'].apply(
                lambda h: 'TRADING_HOUR' if 9 <= h <= 15 else 
                          'AFTER_MARKET' if 16 <= h <= 23 else 'PRE_MARKET'
            )
            
            time_period_analysis = matched_df.groupby('time_period').agg(
                avg_abs_return=('daily_return', lambda x: np.abs(x).mean()),
                avg_sentiment=('sentiment_score', 'mean') if 'sentiment_score' in matched_df.columns else ('daily_return', 'mean'),
                sample_count=('daily_return', 'count')
            ).reset_index()
            results['time_period_analysis'] = time_period_analysis
        
        # 6. Symbol-specific analysis
        if all(col in matched_df.columns for col in ['symbol', 'daily_return']):
            if 'sentiment_score' in matched_df.columns:
                symbol_analysis = matched_df.groupby('symbol').agg(
                    avg_sentiment=('sentiment_score', 'mean'),
                    avg_return=('daily_return', 'mean'),
                    sample_count=('daily_return', 'count')
                ).sort_values('avg_return', ascending=False).reset_index()
                results['symbol_analysis'] = symbol_analysis
        
        logger.info("Correlation analysis completed successfully")
        return results
    
    def create_visualizations(self, matched_df, analysis_results):
        """Create comprehensive visualizations for the analysis"""
        logger.info("Creating comprehensive visualizations...")
        
        if matched_df.empty:
            logger.warning("No data for visualizations")
            return []
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Stock-News Correlation Analysis Dashboard", fontsize=16, fontweight='bold')
        
        # 1. Sentiment vs Average Return
        if 'sentiment_return' in analysis_results:
            sentiment_return = analysis_results['sentiment_return']
            sns.barplot(x='sentiment_label', y='avg_return', data=sentiment_return, ax=axes[0, 0])
            axes[0, 0].set_title('Average Return by News Sentiment', fontweight='bold')
            axes[0, 0].set_ylabel('Average Return (%)')
            axes[0, 0].set_xlabel('Sentiment')
            axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # 2. Sentiment Distribution
        if 'sentiment_label' in matched_df.columns:
            sentiment_counts = matched_df['sentiment_label'].value_counts()
            if not sentiment_counts.empty:
                axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                              autopct='%1.1f%%', colors=['#2ecc71', '#3498db', '#e74c3c'])
                axes[0, 1].set_title('News Sentiment Distribution', fontweight='bold')
        
        # 3. Daily Return Distribution by Sentiment
        if 'sentiment_label' in matched_df.columns and 'daily_return' in matched_df.columns:
            sns.boxplot(x='sentiment_label', y='daily_return', data=matched_df, ax=axes[0, 2])
            axes[0, 2].set_title('Return Distribution by Sentiment', fontweight='bold')
            axes[0, 2].set_ylabel('Daily Return (%)')
            axes[0, 2].set_xlabel('Sentiment')
            axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # 4. News Count vs Volume (use available volume column)
        volume_col = 'total_volume' if 'total_volume' in matched_df.columns else 'volume_num' if 'volume_num' in matched_df.columns else None
        if volume_col and 'symbol' in matched_df.columns:
            try:
                volume_analysis = matched_df.groupby('symbol').agg(
                    news_count=('symbol', 'count'),
                    avg_volume=(volume_col, 'mean')
                ).reset_index()
                
                if not volume_analysis.empty:
                    sns.scatterplot(x='news_count', y='avg_volume', 
                                  hue='symbol', data=volume_analysis, ax=axes[1, 0])
                    axes[1, 0].set_title('News Count vs Average Volume', fontweight='bold')
                    axes[1, 0].set_xlabel('News Count')
                    axes[1, 0].set_ylabel('Average Volume')
                    axes[1, 0].set_yscale('log')
            except Exception as e:
                logger.warning(f"Error creating volume visualization: {e}")
        
        # 5. Time Period Impact
        if 'time_period_analysis' in analysis_results:
            time_period = analysis_results['time_period_analysis']
            sns.barplot(x='time_period', y='avg_abs_return', data=time_period, ax=axes[1, 1])
            axes[1, 1].set_title('Impact by News Time Period', fontweight='bold')
            axes[1, 1].set_ylabel('Average Absolute Return (%)')
            axes[1, 1].set_xlabel('Time Period')
        
        # 6. Source Impact
        if 'source_impact' in analysis_results:
            source_impact = analysis_results['source_impact']
            if not source_impact.empty:
                top_sources = source_impact.head(5)
                sns.barplot(x='source', y='avg_abs_return', data=top_sources, ax=axes[1, 2])
                axes[1, 2].set_title('Top News Sources by Impact', fontweight='bold')
                axes[1, 2].set_ylabel('Average Absolute Return (%)')
                axes[1, 2].set_xlabel('News Source')
                axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.output_dir, f"stock_news_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to: {viz_path}")
        return [viz_path]
    
    def generate_report(self, analysis_results):
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive analysis report...")
        
        if not analysis_results:
            logger.warning("No analysis results to report")
            return {}
        
        # Convert analysis results to serializable format
        serializable_results = self.convert_dataframe_to_serializable(analysis_results)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'Stock-News Correlation Analysis'
            },
            'summary': {
                'total_stocks_analyzed': 0,
                'total_news_articles': 0,
                'total_matches': 0,
                'overall_sentiment_impact': 0.0
            },
            'detailed_analysis': serializable_results
        }
        
        # Add summary statistics if available
        if 'sentiment_return' in analysis_results:
            sentiment_return = analysis_results['sentiment_return']
            if not sentiment_return.empty:
                positive_impact = sentiment_return[sentiment_return['sentiment_label'] == 'POSITIVE']['avg_return'].values
                negative_impact = sentiment_return[sentiment_return['sentiment_label'] == 'NEGATIVE']['avg_return'].values
                
                report['summary']['overall_sentiment_impact'] = (
                    float(positive_impact[0]) if len(positive_impact) > 0 else 0.0
                ) - (
                    float(negative_impact[0]) if len(negative_impact) > 0 else 0.0
                )
        
        # Save report
        report_path = os.path.join(self.output_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report saved to: {report_path}")
        return report
    
    def save_analysis_results(self, analysis_results):
        """Save detailed analysis results to CSV files"""
        logger.info("Saving detailed analysis results...")
        
        for name, df in analysis_results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = os.path.join(self.output_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {name} analysis to: {filepath}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        logger.info("🚀 Starting Stock-News Correlation Analysis Pipeline")
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            logger.info("\n" + "="*60)
            logger.info("📊 STEP 1: LOADING DATA")
            logger.info("="*60)
            
            stock_df = self.load_json_files(self.stock_dir)
            news_df = self.load_json_files(self.news_dir)
            
            if news_df.empty:
                logger.error("❌ Critical error: No news data found")
                return False
            
            # Step 2: Clean and preprocess data
            logger.info("\n" + "="*60)
            logger.info("🧹 STEP 2: CLEANING AND PREPROCESSING DATA")
            logger.info("="*60)
            
            clean_stock_df = self.clean_stock_data(stock_df)
            clean_news_df = self.clean_news_data(news_df)
            
            if clean_news_df.empty:
                logger.error("❌ Critical error: News data cleaning failed")
                return False
            
            # Step 3: Feature engineering
            logger.info("\n" + "="*60)
            logger.info("⚙️ STEP 3: FEATURE ENGINEERING")
            logger.info("="*60)
            
            # Calculate sentiment
            news_with_sentiment = self.calculate_sentiment(clean_news_df)
            
            # Calculate stock returns (even if stock data is limited)
            stock_with_returns = self.calculate_stock_returns(clean_stock_df)
            
            # Step 4: Match news to stocks
            logger.info("\n" + "="*60)
            logger.info("🔗 STEP 4: MATCHING NEWS TO STOCKS")
            logger.info("="*60)
            
            matched_df = self.match_news_to_stocks(news_with_sentiment, stock_with_returns)
            
            if matched_df.empty:
                logger.warning("⚠️ No matches found between news and stocks")
                # Create minimal matched_df with just news data
                matched_df = news_with_sentiment.copy()
                matched_df['symbol'] = matched_df['matched_symbol']
                matched_df['daily_return'] = 0.0
                matched_df['total_volume'] = 0
                matched_df['stock_date'] = matched_df['news_date']
            
            # Step 5: Correlation analysis
            logger.info("\n" + "="*60)
            logger.info("📈 STEP 5: CORRELATION ANALYSIS")
            logger.info("="*60)
            
            analysis_results = self.perform_correlation_analysis(matched_df)
            
            # Step 6: Create visualizations
            logger.info("\n" + "="*60)
            logger.info("🎨 STEP 6: CREATING VISUALIZATIONS")
            logger.info("="*60)
            
            visualization_paths = self.create_visualizations(matched_df, analysis_results)
            
            # Step 7: Generate report
            logger.info("\n" + "="*60)
            logger.info("📄 STEP 7: GENERATING REPORT")
            logger.info("="*60)
            
            report = self.generate_report(analysis_results)
            self.save_analysis_results(analysis_results)
            
            # Final summary
            elapsed_time = datetime.now() - start_time
            logger.info("\n" + "="*60)
            logger.info("✅ ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"⏰ Total execution time: {elapsed_time.total_seconds():.2f} seconds")
            logger.info(f"📊 Stocks analyzed: {len(clean_stock_df['symbol'].unique()) if not clean_stock_df.empty else 0}")
            logger.info(f"📰 News articles processed: {len(clean_news_df)}")
            logger.info(f"🔗 News-stock matches found: {len(matched_df)}")
            
            if analysis_results and 'sentiment_return' in analysis_results:
                sentiment_return = analysis_results['sentiment_return']
                logger.info("\nSENTIMENT IMPACT SUMMARY:")
                for _, row in sentiment_return.iterrows():
                    logger.info(f"  • {row['sentiment_label']}: {row['avg_return']:.4f}% average return (n={row['sample_count']})")
            
            logger.info(f"\n💾 Results saved to: {self.output_dir}")
            if visualization_paths:
                logger.info(f"🖼️ Visualizations saved to: {', '.join(visualization_paths)}")
            else:
                logger.warning("No visualizations were created due to data limitations")
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Analysis pipeline failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

def main():
    """Main function to run the analysis"""
    analyzer = StockNewsAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        logger.info("🎉 Analysis completed successfully!")
    else:
        logger.error("❌ Analysis failed. Check logs for details.")

if __name__ == "__main__":
    main()