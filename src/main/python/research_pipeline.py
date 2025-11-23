import time
import logging
import os
from datetime import datetime
import pandas as pd

from config import BASE_DIR
from utils import setup_logging, create_success_logger, get_current_timestamp, save_json
from stock_crawler import StockDataCrawler
from news_crawler import NewsArticleCrawler
from sentiment_analyzer import SentimentAnalyzer
from data_integration import DataIntegration
from visualization import Visualizer
from advanced_analysis import AdvancedStockNewsAnalyzer

def run_research_pipeline(max_stocks=50, max_articles_per_topic=20, max_topics=10):
    """Enhanced pipeline with advanced research capabilities"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Yahoo Finance Research Pipeline")
    start_time = time.time()
    
    # Step 1: Crawl data (same as before)
    logger.info("\n" + "="*60)
    logger.info("📊 STEP 1: DATA COLLECTION")
    logger.info("="*60)
    
    stock_crawler = StockDataCrawler()
    stocks = stock_crawler.crawl_most_active_stocks(max_stocks=max_stocks)
    # Save the stock summary path when saving data
    stock_json_path, _ = stock_crawler.save_stock_data(stocks)
    # Extract the stock summary path from the filename pattern
    stock_summary_filename = os.path.basename(stock_json_path).replace('most_active_stocks_', 'stock_summary_')
    stock_summary_path = os.path.join(BASE_DIR, "analysis", stock_summary_filename)
    
    news_crawler = NewsArticleCrawler()
    news_data = news_crawler.scrape_all_topics(
        max_articles_per_topic=max_articles_per_topic,
        max_topics=max_topics
    )
    news_json_path = news_crawler.save_news_data(news_data)
    # Extract the news summary path from the filename pattern
    news_summary_filename = os.path.basename(news_json_path).replace('yahoo_finance_news_', 'news_summary_')
    news_summary_path = os.path.join(BASE_DIR, "analysis", news_summary_filename)
    
    # Step 2: Sentiment analysis
    logger.info("\n" + "="*60)
    logger.info("🧠 STEP 2: SENTIMENT ANALYSIS")
    logger.info("="*60)
    
    sentiment_analyzer = SentimentAnalyzer()
    news_with_sentiment = sentiment_analyzer.analyze_articles(news_data["posts"])
    sentiment_summary = sentiment_analyzer.save_sentiment_results(news_with_sentiment)
    
    # Step 3: Data integration
    logger.info("\n" + "="*60)
    logger.info("🔗 STEP 3: DATA INTEGRATION")
    logger.info("="*60)
    
    data_integration = DataIntegration()
    integrated_data = data_integration.integrate_stock_news(stocks, news_with_sentiment)
    correlation_results = data_integration.analyze_correlations(integrated_data)
    integrated_path = data_integration.save_analysis(integrated_data, correlation_results)
    
    # Step 4: Advanced research analysis (NEW)
    logger.info("\n" + "="*60)
    logger.info("🔬 STEP 4: ADVANCED RESEARCH ANALYSIS")
    logger.info("="*60)
    
    advanced_analyzer = AdvancedStockNewsAnalyzer()
    
    # Prepare time series data
    time_series_data = advanced_analyzer.prepare_time_series_data(integrated_data)
    
    # Granger causality analysis
    granger_results = advanced_analyzer.perform_granger_causality_analysis(time_series_data) if time_series_data is not None else None
    
    # Prediction modeling
    prediction_results = advanced_analyzer.build_prediction_model(time_series_data) if time_series_data is not None else None
    
    # Sentiment clustering
    cluster_results = advanced_analyzer.perform_sentiment_clustering(integrated_data)
    
    # Create advanced visualizations
    logger.info("\n" + "="*60)
    logger.info("🎨 STEP 5: ADVANCED VISUALIZATIONS")
    logger.info("="*60)
    
    # Time series analysis
    if time_series_data is not None:
        # Create market-wide time series
        advanced_analyzer.create_sentiment_price_time_series(time_series_data)
        
        # Create individual stock time series for top 5 stocks
        top_stocks = pd.DataFrame(integrated_data)['stock_symbol'].value_counts().head(5).index.tolist()
        for stock in top_stocks:
            advanced_analyzer.create_sentiment_price_time_series(time_series_data, stock_symbol=stock)
    
    # Sector heatmap
    advanced_analyzer.create_sector_sentiment_heatmap(integrated_data)
    
    # Network analysis
    advanced_analyzer.create_network_analysis(integrated_data)
    
    # Step 6: Generate research report
    logger.info("\n" + "="*60)
    logger.info("📝 STEP 6: RESEARCH REPORT GENERATION")
    logger.info("="*60)
    
    analysis_results = {
        'integrated_data': integrated_data,
        'correlation_results': correlation_results,
        'granger_results': granger_results
    }
    
    research_report = advanced_analyzer.generate_research_report(
        analysis_results,
        prediction_results,
        cluster_results
    )
    
    # Final summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("✅ RESEARCH PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"⏰ Total execution time: {elapsed_time:.2f} seconds")
    logger.info(f"📈 Stocks analyzed: {len(stocks)}")
    logger.info(f"📰 News articles processed: {len(news_data['posts'])}")
    logger.info(f"🔗 Integrated stock-news pairs: {len(integrated_data)}")
    
    if correlation_results:
        logger.info(f"📊 Overall sentiment-price correlation: {correlation_results['overall_correlation']:.3f}")
    
    if prediction_results:
        accuracy = prediction_results['results']['classification_accuracy']
        logger.info(f"🤖 Prediction model accuracy: {accuracy:.1%}")
    
    logger.success("🎉 Research pipeline completed successfully!")
    
    return {
        'stocks': stocks,
        'news_data': news_data,
        'integrated_data': integrated_data,
        'correlation_results': correlation_results,
        'prediction_results': prediction_results,
        'cluster_results': cluster_results,
        'research_report': research_report
    }

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(os.path.join(BASE_DIR, "logs", f"research_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    logger = create_success_logger(logger)
    
    # Run the research pipeline
    results = run_research_pipeline(
        max_stocks=50,
        max_articles_per_topic=20,
        max_topics=10
    )