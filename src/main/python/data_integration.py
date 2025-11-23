import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from config import BASE_DIR
from utils import get_current_timestamp, save_json
import os
class DataIntegration:
    """Integrate stock data with news sentiment analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def integrate_stock_news(self, stocks, news_with_sentiment):
        """Integrate stock data with news sentiment"""
        self.logger.info("Integrating stock data with news sentiment...")
        
        # Create stock ticker mapping
        stock_map = {stock['symbol'].upper(): stock for stock in stocks}
        
        integrated_data = []
        
        for news_item in news_with_sentiment:
            stock_symbols = news_item.get("stock_symbols", [])
            
            if stock_symbols:
                for symbol in stock_symbols:
                    clean_symbol = symbol.upper().split('-')[0]
                    
                    if clean_symbol in stock_map:
                        stock_data = stock_map[clean_symbol]
                        
                        integrated_item = {
                            "news_id": news_item.get("post_id"),
                            "stock_symbol": clean_symbol,
                            "stock_name": stock_data.get("name", ""),
                            "current_price": stock_data.get("price", 0.0),
                            "daily_change_percent": stock_data.get("change_percent", 0.0),
                            "market_cap": stock_data.get("market_cap", ""),
                            "volume": stock_data.get("volume", ""),
                            "news_headline": news_item.get("headline", ""),
                            "news_url": news_item.get("url", ""),
                            "news_sentiment": news_item.get("sentiment_label", "neutral"),
                            "sentiment_score": news_item.get("sentiment_score", 0.0),
                            "published_at": news_item.get("created_iso", ""),
                            "analysis_timestamp": get_current_timestamp()
                        }
                        integrated_data.append(integrated_item)
        
        self.logger.info(f"Integrated {len(integrated_data)} stock-news pairs")
        return integrated_data
    
    def analyze_correlations(self, integrated_data):
        """Analyze correlations between news sentiment and stock performance"""
        if not integrated_data:
            self.logger.warning("No integrated data for correlation analysis")
            return None
        
        self.logger.info("Analyzing correlations between news sentiment and stock performance...")
        
        # Convert to DataFrame
        df = pd.DataFrame(integrated_data)
        
        if df.empty:
            self.logger.warning("Empty DataFrame for correlation analysis")
            return None
        
        # Group by stock symbol
        symbol_groups = df.groupby('stock_symbol').agg({
            'sentiment_score': 'mean',
            'daily_change_percent': 'mean',
            'news_sentiment': lambda x: x.value_counts().index[0],
            'stock_name': 'first',
            'current_price': 'first',
            'market_cap': 'first'
        }).reset_index()
        
        # Calculate correlation
        if len(symbol_groups) > 1:
            correlation = symbol_groups['sentiment_score'].corr(symbol_groups['daily_change_percent'])
        else:
            correlation = 0.0
        
        # Identify significant correlations
        positive_correlations = symbol_groups[
            (symbol_groups['sentiment_score'] > 0.1) & 
            (symbol_groups['daily_change_percent'] > 0)
        ].sort_values('sentiment_score', ascending=False)
        
        negative_correlations = symbol_groups[
            (symbol_groups['sentiment_score'] < -0.1) & 
            (symbol_groups['daily_change_percent'] < 0)
        ].sort_values('sentiment_score', ascending=True)
        
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
            "top_positive_stocks": symbol_groups.nlargest(5, 'sentiment_score')[['stock_symbol', 'stock_name', 'sentiment_score', 'daily_change_percent']].to_dict('records'),
            "top_negative_stocks": symbol_groups.nsmallest(5, 'sentiment_score')[['stock_symbol', 'stock_name', 'sentiment_score', 'daily_change_percent']].to_dict('records'),
            "timestamp": get_current_timestamp()
        }
        
        self.logger.info(f"Overall sentiment-price correlation: {correlation:.4f}")
        self.logger.info(f"Positive correlations found: {len(positive_correlations)} stocks")
        self.logger.info(f"Negative correlations found: {len(negative_correlations)} stocks")
        
        return analysis
    
    def save_analysis(self, integrated_data, correlation_results, filename=None):
        """Save analysis results"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_news_analysis_{timestamp}"
        
        # Save integrated data
        integrated_path = os.path.join(BASE_DIR, "analysis", f"{filename}_integrated.json")
        save_json(integrated_data, integrated_path)
        
        # Save correlation results
        if correlation_results:
            correlation_path = os.path.join(BASE_DIR, "analysis", f"{filename}_correlations.json")
            save_json(correlation_results, correlation_path)
            
            self.logger.info(f"Correlation analysis saved to {correlation_path}")
        
        self.logger.info(f"Integrated data saved to {integrated_path}")
        return integrated_path