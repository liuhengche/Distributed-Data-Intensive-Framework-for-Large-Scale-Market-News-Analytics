"""
Yahoo Finance Data Pipeline - Main Execution
"""

import time
import argparse
import logging
import os
from datetime import datetime

from config import LOG_FILE, DEFAULT_MAX_STOCKS, DEFAULT_MAX_ARTICLES_PER_TOPIC, DEFAULT_MAX_TOPICS, BASE_DIR
from utils import setup_logging, create_success_logger, get_current_timestamp, save_json
from stock_crawler import StockDataCrawler
from news_crawler import NewsArticleCrawler
from sentiment_analyzer import SentimentAnalyzer
from data_integration import DataIntegration
from visualization import Visualizer

def run_pipeline(max_stocks=200, max_articles_per_topic=15, max_topics=8, skip_sentiment=False):
    """Main pipeline execution function"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Yahoo Finance Data Pipeline")
    logger.info(f"Base directory: {BASE_DIR}")
    start_time = time.time()
    
    try:
        # Step 1: Crawl stock data
        logger.info("\n" + "="*50)
        logger.info("STEP 1: CRAWLING STOCK DATA")
        logger.info("="*50)
        
        stock_crawler = StockDataCrawler()
        stocks = stock_crawler.crawl_most_active_stocks(max_stocks=max_stocks)
        
        if not stocks:
            logger.error("No stock data crawled. Cannot proceed with pipeline.")
            return
        
        # Save the stock summary path when saving data
        stock_json_path, _ = stock_crawler.save_stock_data(stocks)
        # Extract the stock summary path from the filename pattern
        stock_summary_filename = os.path.basename(stock_json_path).replace('most_active_stocks_', 'stock_summary_')
        stock_summary_path = os.path.join(BASE_DIR, "analysis", stock_summary_filename)

        # Step 2: Crawl news articles
        logger.info("\n" + "="*50)
        logger.info("STEP 2: CRAWLING NEWS ARTICLES")
        logger.info("="*50)
        
        news_crawler = NewsArticleCrawler()
        news_data = news_crawler.scrape_all_topics(
            max_articles_per_topic=max_articles_per_topic,
            max_topics=max_topics
        )
        
        if not news_data["posts"]:
            logger.error("No news articles crawled. Cannot proceed with pipeline.")
            return
        
        # Save the news summary path when saving data
        news_json_path = news_crawler.save_news_data(news_data)
        # Extract the news summary path from the filename pattern
        news_summary_filename = os.path.basename(news_json_path).replace('yahoo_finance_news_', 'news_summary_')
        news_summary_path = os.path.join(BASE_DIR, "analysis", news_summary_filename)
        
        # Step 3: Sentiment analysis
        news_with_sentiment = news_data["posts"]
        sentiment_summary_path = None
        
        if not skip_sentiment:
            logger.info("\n" + "="*50)
            logger.info("STEP 3: SENTIMENT ANALYSIS")
            logger.info("="*50)
            
            sentiment_analyzer = SentimentAnalyzer()
            news_with_sentiment = sentiment_analyzer.analyze_articles(news_data["posts"])
            sentiment_summary_path = sentiment_analyzer.save_sentiment_results(news_with_sentiment)
        
        # Step 4: Data integration and analysis
        logger.info("\n" + "="*50)
        logger.info("STEP 4: DATA INTEGRATION AND ANALYSIS")
        logger.info("="*50)
        
        data_integration = DataIntegration()
        integrated_data = data_integration.integrate_stock_news(stocks, news_with_sentiment)
        
        correlation_results = None
        correlation_path = None

        if integrated_data:
            correlation_results = data_integration.analyze_correlations(integrated_data)
            output_path = data_integration.save_analysis(integrated_data, correlation_results)

            if correlation_results:
                correlation_path = output_path.replace('_integrated.json', '_correlations.json')
        
        # Step 5: Visualization
        logger.info("\n" + "="*50)
        logger.info("STEP 5: VISUALIZATION AND REPORTING")
        logger.info("="*50)
        
        visualizer = Visualizer()
        
        # Create various visualizations
        sentiment_viz_path = None
        stock_viz_path = None
        correlation_viz_path = None
        dashboard_path = None
        summary_report_path = None
        
        if sentiment_summary_path:
            sentiment_viz_path = visualizer.create_sentiment_distribution_chart(sentiment_summary_path)
        
        if stock_json_path:
            stock_viz_path = visualizer.create_stock_performance_chart(stocks)
        
        if correlation_path:
            correlation_viz_path = visualizer.create_correlation_visualization(correlation_results)
            
            # Create interactive dashboard if we have integrated data
            if os.path.exists(output_path) and os.path.exists(correlation_path):
                dashboard_path = visualizer.create_interactive_dashboard(output_path, correlation_path)
        
        # Generate final summary
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        elapsed_time = time.time() - start_time
        summary = {
            "pipeline_execution_time_seconds": round(elapsed_time, 2),
            "stocks_crawled": len(stocks),
            "news_articles_crawled": len(news_data["posts"]),
            "articles_with_sentiment": len(news_with_sentiment) if not skip_sentiment else 0,
            "stock_news_integrations": len(integrated_data) if integrated_data else 0,
            "sentiment_analysis_skipped": skip_sentiment,
            "completed_at": get_current_timestamp(),
            "directories": {
                "base": BASE_DIR,
                "stock_data": os.path.join(BASE_DIR, "stock_information"),
                "news_data": os.path.join(BASE_DIR, "news_information"),
                "analysis": os.path.join(BASE_DIR, "analysis"),
                "visualizations": os.path.join(BASE_DIR, "visualizations")
            },
            "output_files": {
                "stock_data": stock_json_path,
                "news_data": news_json_path,
                "sentiment_summary": sentiment_summary_path,
                "integrated_data": output_path if integrated_data else None,
                "correlation_analysis": correlation_path,
                "sentiment_visualization": sentiment_viz_path,
                "stock_visualization": stock_viz_path,
                "correlation_visualization": correlation_viz_path,
                "interactive_dashboard": dashboard_path
            }
        }
        
        summary_path = os.path.join(BASE_DIR, "analysis", f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_json(summary, summary_path)
        
        # Create daily summary report
        if correlation_results and stock_json_path and news_json_path and sentiment_summary_path and correlation_path:
            if correlation_results and stock_summary_path and news_summary_path and sentiment_summary_path and correlation_path:
                summary_report_path = visualizer.create_daily_summary_report(
                    summary_path,
                    stock_summary_path,  # Use the saved path
                    news_summary_path,   # Use the saved path
                    sentiment_summary_path,
                    correlation_path
                )
        
        logger.info(f"Pipeline summary saved to {summary_path}")
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        
        # Print key insights
        if correlation_results:
            logger.info("\n" + "="*50)
            logger.info("KEY INSIGHTS")
            logger.info("="*50)
            logger.info(f"Overall sentiment-price correlation: {correlation_results['overall_correlation']:.4f}")
            logger.info(f"Stocks with positive sentiment & positive performance: {correlation_results['positive_correlations']['count']}")
            logger.info(f"Stocks with negative sentiment & negative performance: {correlation_results['negative_correlations']['count']}")
            
            if correlation_results['positive_correlations']['count'] > 0:
                logger.info("\nTop Positive Correlations:")
                for stock in correlation_results['positive_correlations']['stocks'][:3]:
                    logger.info(f"  {stock['stock_symbol']} ({stock['stock_name']}): Sentiment={stock['sentiment_score']:.3f}, Change={stock['daily_change_percent']:.2f}%")
            
            if correlation_results['negative_correlations']['count'] > 0:
                logger.info("\nTop Negative Correlations:")
                for stock in correlation_results['negative_correlations']['stocks'][:3]:
                    logger.info(f"  {stock['stock_symbol']} ({stock['stock_name']}): Sentiment={stock['sentiment_score']:.3f}, Change={stock['daily_change_percent']:.2f}%")
        
        logger.success("Pipeline completed successfully!")
        return summary
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
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
        save_json(error_info, error_path)
        
        logger.info(f"Error details saved to {error_path}")
        return None

def main():
    """Command line interface for the pipeline with research capabilities"""
    # Setup logging first
    logger = setup_logging(LOG_FILE)
    logger = create_success_logger(logger)
    
    parser = argparse.ArgumentParser(description='Yahoo Finance Data Pipeline')
    parser.add_argument('--max-stocks', type=int, default=DEFAULT_MAX_STOCKS, help='Maximum number of stocks to crawl')
    parser.add_argument('--max-articles-per-topic', type=int, default=DEFAULT_MAX_ARTICLES_PER_TOPIC, help='Maximum articles per topic')
    parser.add_argument('--max-topics', type=int, default=DEFAULT_MAX_TOPICS, help='Maximum number of topics to crawl')
    parser.add_argument('--test-mode', action='store_true', default=False, help='Run in test mode with reduced limits')
    parser.add_argument('--skip-sentiment', action='store_true', default=False, help='Skip sentiment analysis')
    parser.add_argument('--research-mode', action='store_true', default=True, help='Run advanced research pipeline with predictions and deep analysis')

    args = parser.parse_args()
    
    # Update configuration based on arguments
    max_stocks = args.max_stocks
    max_articles_per_topic = args.max_articles_per_topic
    max_topics = args.max_topics
    
    if args.test_mode:
        logger.info("🧪 Running in test mode with reduced limits")
        max_stocks = 20
        max_articles_per_topic = 3
        max_topics = 3
    
    logger.info(f"⚙️ Configuration: Max stocks={max_stocks}, Max articles per topic={max_articles_per_topic}, Max topics={max_topics}")
    
    # Run appropriate pipeline
    if args.research_mode:
        logger.info("🔬 Running in RESEARCH MODE with advanced analysis and prediction capabilities")
        from research_pipeline import run_research_pipeline
        result = run_research_pipeline(
            max_stocks=max_stocks,
            max_articles_per_topic=max_articles_per_topic,
            max_topics=max_topics
        )
    else:
        result = run_pipeline(
            max_stocks=max_stocks,
            max_articles_per_topic=max_articles_per_topic,
            max_topics=max_topics,
            skip_sentiment=args.skip_sentiment
        )
    
    if result:
        logger.info("✅ Pipeline execution completed successfully!")
    else:
        logger.error("❌ Pipeline execution failed!")

if __name__ == "__main__":
    main()