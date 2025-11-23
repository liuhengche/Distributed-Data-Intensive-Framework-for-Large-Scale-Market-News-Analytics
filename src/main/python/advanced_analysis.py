import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime, timedelta
from config import BASE_DIR
from utils import get_current_timestamp, save_json, create_directory_if_not_exists
import os

class AdvancedStockNewsAnalyzer:
    """Advanced analysis of stock-news relationships with predictive capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_dir = os.path.join(BASE_DIR, "analysis", "advanced")
        create_directory_if_not_exists(self.analysis_dir)
        self.visualizations_dir = os.path.join(BASE_DIR, "visualizations", "advanced")
        create_directory_if_not_exists(self.visualizations_dir)
        
    def prepare_time_series_data(self, integrated_data, stock_historical_data=None):
        """Prepare time series data for analysis with lagged sentiment features"""
        self.logger.info("🔄 Preparing time series data for advanced analysis...")
        
        if not integrated_data:
            self.logger.warning("⚠️ No integrated data available for time series analysis")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(integrated_data)
        
        # Convert timestamp to datetime
        df['analysis_timestamp'] = pd.to_datetime(df['analysis_timestamp'])
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # Create daily aggregates
        df['date'] = df['published_at'].dt.date
        
        # Group by stock and date - use an existing column for counting news articles
        daily_data = df.groupby(['stock_symbol', 'date']).agg({
            'sentiment_score': 'mean',
            'daily_change_percent': 'mean',
            'current_price': 'last',
            'volume': 'last',
            'news_id': 'count'  # Use an existing column that exists in your data
        }).reset_index().rename(columns={'news_id': 'news_count'})
        
        # Create lagged sentiment features (1, 2, 3, 5, 7 days)
        for lag in [1, 2, 3, 5, 7]:
            daily_data[f'sentiment_lag_{lag}'] = daily_data.groupby('stock_symbol')['sentiment_score'].shift(lag)
        
        # Calculate price momentum features
        daily_data['price_change_1d'] = daily_data.groupby('stock_symbol')['current_price'].pct_change()
        daily_data['price_change_3d'] = daily_data.groupby('stock_symbol')['current_price'].pct_change(3)
        daily_data['price_change_5d'] = daily_data.groupby('stock_symbol')['current_price'].pct_change(5)
        
        # Calculate volatility features
        daily_data['volatility_5d'] = daily_data.groupby('stock_symbol')['price_change_1d'].rolling(5, min_periods=1).std().reset_index(0, drop=True)
        
        # Drop NaN values
        daily_data = daily_data.dropna(subset=[f'sentiment_lag_{lag}' for lag in [1, 2, 3, 5, 7]])
        
        self.logger.info(f"✅ Prepared time series data with {len(daily_data)} observations")
        return daily_data
    
    def perform_granger_causality_analysis(self, time_series_data):
        """Perform Granger causality test to determine if sentiment predicts price movements"""
        self.logger.info("📊 Performing Granger causality analysis...")
        
        results = {}
        
        # Group by stock symbol
        for symbol, group in time_series_data.groupby('stock_symbol'):
            if len(group) < 50:  # Need sufficient data points
                continue
            
            try:
                # Prepare data for Granger test
                test_data = group[['sentiment_score', 'price_change_1d']].dropna()
                
                if len(test_data) < 20:
                    continue
                
                # Perform Granger causality test
                gc_result = grangercausalitytests(test_data[['price_change_1d', 'sentiment_score']], maxlag=5, verbose=False)
                
                # Extract p-values for different lags
                p_values = {}
                for lag in range(1, 6):
                    p_value = gc_result[lag][0]['ssr_chi2test'][1]
                    p_values[f'lag_{lag}'] = p_value
                
                # Store results
                results[symbol] = {
                    'p_values': p_values,
                    'significant_lags': [lag for lag, p in p_values.items() if p < 0.05],
                    'sample_size': len(test_data)
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error performing Granger test for {symbol}: {e}")
                continue
        
        # Save results
        output_path = os.path.join(self.analysis_dir, f"granger_causality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_json(results, output_path)
        
        self.logger.info(f"✅ Granger causality analysis completed. Results saved to {output_path}")
        return results
    
    def build_prediction_model(self, time_series_data):
        """Build machine learning models to predict stock movements based on sentiment"""
        self.logger.info("🤖 Building prediction models...")
        
        if time_series_data is None or len(time_series_data) < 100:
            self.logger.warning("⚠️ Insufficient data for prediction modeling")
            return None
        
        # Prepare features and target
        features = ['sentiment_lag_1', 'sentiment_lag_2', 'sentiment_lag_3', 'sentiment_lag_5', 'sentiment_lag_7',
                   'volatility_5d', 'news_count']
        target = 'price_change_1d'
        
        # Remove rows with missing values
        model_data = time_series_data.dropna(subset=features + [target])
        
        if len(model_data) < 50:
            self.logger.warning("⚠️ Not enough complete data for modeling")
            return None
        
        # Create binary target (price increase/decrease)
        model_data['price_direction'] = (model_data[target] > 0).astype(int)
        
        # Split data
        train_data, test_data = train_test_split(model_data, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[features])
        X_test = scaler.transform(test_data[features])
        
        # Classification model (direction prediction)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, train_data['price_direction'])
        
        # Regression model (magnitude prediction)
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train, train_data[target])
        
        # Evaluate models
        y_pred_clf = clf.predict(X_test)
        y_pred_reg = reg.predict(X_test)
        
        accuracy = accuracy_score(test_data['price_direction'], y_pred_clf)
        mse = mean_squared_error(test_data[target], y_pred_reg)
        
        self.logger.info(f"✅ Classification model accuracy: {accuracy:.3f}")
        self.logger.info(f"✅ Regression model MSE: {mse:.6f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance_classification': clf.feature_importances_,
            'importance_regression': reg.feature_importances_
        }).sort_values('importance_classification', ascending=False)
        
        # Save model results
        model_results = {
            'classification_accuracy': accuracy,
            'regression_mse': mse,
            'feature_importance': feature_importance.to_dict('records'),
            'sample_size': len(model_data),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'timestamp': get_current_timestamp()
        }
        
        output_path = os.path.join(self.analysis_dir, f"prediction_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_json(model_results, output_path)
        
        # Create feature importance visualization
        self.create_feature_importance_chart(feature_importance)
        
        return {
            'models': {'classifier': clf, 'regressor': reg},
            'scaler': scaler,
            'results': model_results
        }
    
    def create_feature_importance_chart(self, feature_importance):
        """Create visualization of feature importance"""
        plt.figure(figsize=(12, 6))
        
        # Create subplot for classification and regression importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classification importance
        sns.barplot(x='importance_classification', y='feature', data=feature_importance, ax=ax1, palette='viridis')
        ax1.set_title('Feature Importance for Direction Prediction', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Importance Score', fontsize=12)
        
        # Regression importance
        sns.barplot(x='importance_regression', y='feature', data=feature_importance, ax=ax2, palette='viridis')
        ax2.set_title('Feature Importance for Magnitude Prediction', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance Score', fontsize=12)
        
        plt.tight_layout()
        output_path = os.path.join(self.visualizations_dir, f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Feature importance chart saved to {output_path}")
        return output_path
    
    def perform_sentiment_clustering(self, integrated_data):
        """Perform clustering analysis to identify patterns in sentiment-stock relationships"""
        self.logger.info("🔍 Performing sentiment clustering analysis...")
        
        if not integrated_data:
            return None
        
        df = pd.DataFrame(integrated_data)
        
        # Prepare data for clustering
        cluster_data = df.groupby('stock_symbol').agg({
            'sentiment_score': 'mean',
            'daily_change_percent': 'mean',
            'news_count': 'count',
            'current_price': 'mean'
        }).reset_index()
        
        # Normalize features
        features = ['sentiment_score', 'daily_change_percent', 'news_count']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data[features])
        
        # Determine optimal number of clusters
        inertias = []
        for k in range(2, 8):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Choose 4 clusters as a reasonable number
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_stocks = cluster_data[cluster_data['cluster'] == cluster_id]
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'count': len(cluster_stocks),
                'avg_sentiment': cluster_stocks['sentiment_score'].mean(),
                'avg_performance': cluster_stocks['daily_change_percent'].mean(),
                'avg_news_count': cluster_stocks['news_count'].mean(),
                'stocks': cluster_stocks['stock_symbol'].tolist()
            }
        
        # Save results
        output_path = os.path.join(self.analysis_dir, f"sentiment_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_json(cluster_analysis, output_path)
        
        # Create visualization
        self.create_cluster_visualization(cluster_data, optimal_k)
        
        return cluster_analysis
    
    def create_cluster_visualization(self, cluster_data, n_clusters):
        """Create visualization of sentiment clusters"""
        plt.figure(figsize=(15, 10))
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow'][:n_clusters]
        
        for cluster_id in range(n_clusters):
            cluster_points = cluster_data[cluster_data['cluster'] == cluster_id]
            ax.scatter(cluster_points['sentiment_score'], 
                      cluster_points['daily_change_percent'],
                      cluster_points['news_count'],
                      c=colors[cluster_id],
                      s=100,
                      alpha=0.7,
                      label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('Average Sentiment Score', fontsize=12)
        ax.set_ylabel('Average Daily Change (%)', fontsize=12)
        ax.set_zlabel('News Count', fontsize=12)
        ax.set_title('Stock Clusters Based on Sentiment-Performance Relationship', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        output_path = os.path.join(self.visualizations_dir, f"sentiment_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Cluster visualization saved to {output_path}")
        return output_path
    
    def create_sentiment_price_time_series(self, time_series_data, stock_symbol=None, days=30):
        """Create time series visualization showing sentiment and price movements"""
        if time_series_data is None:
            return None
        
        plt.figure(figsize=(15, 8))
        
        if stock_symbol:
            # Focus on specific stock
            stock_data = time_series_data[time_series_data['stock_symbol'] == stock_symbol].sort_values('date')
            if len(stock_data) < 5:
                return None
            
            # Plot last N days
            recent_data = stock_data.tail(days)
            
            fig, ax1 = plt.subplots(figsize=(15, 8))
            
            # Price line
            color = 'tab:blue'
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Stock Price ($)', color=color, fontsize=12)
            ax1.plot(recent_data['date'], recent_data['current_price'], color=color, linewidth=2.5, label='Stock Price')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Sentiment line (secondary y-axis)
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Sentiment Score', color=color, fontsize=12)
            ax2.plot(recent_data['date'], recent_data['sentiment_score'], color=color, linestyle='--', linewidth=2, label='Sentiment Score')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add sentiment annotations
            for idx, row in recent_data.iterrows():
                if abs(row['sentiment_score']) > 0.3:  # Only annotate strong sentiment
                    ax2.annotate(f"{row['sentiment_score']:.2f}", 
                               (row['date'], row['sentiment_score']),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', va='bottom',
                               fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            plt.title(f'{stock_symbol} Price and News Sentiment Over Time', fontsize=16, fontweight='bold')
            fig.tight_layout()
            
            output_path = os.path.join(self.visualizations_dir, f"{stock_symbol}_sentiment_price_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Time series visualization for {stock_symbol} saved to {output_path}")
            return output_path
        
        else:
            # Aggregate analysis across all stocks
            daily_agg = time_series_data.groupby('date').agg({
                'sentiment_score': 'mean',
                'price_change_1d': 'mean',
                'news_count': 'sum'
            }).reset_index()
            
            if len(daily_agg) < 5:
                return None
            
            # Plot correlation over time
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
            
            # Daily sentiment and price change
            ax1.plot(daily_agg['date'], daily_agg['sentiment_score'], 'b-', linewidth=2, label='Average Sentiment')
            ax1.set_ylabel('Sentiment Score', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            ax1b = ax1.twinx()
            ax1b.plot(daily_agg['date'], daily_agg['price_change_1d'] * 100, 'r-', linewidth=2, label='Avg Price Change (%)')
            ax1b.set_ylabel('Price Change (%)', color='r', fontsize=12)
            ax1b.tick_params(axis='y', labelcolor='r')
            ax1b.legend(loc='upper right')
            
            # News volume
            ax2.bar(daily_agg['date'], daily_agg['news_count'], alpha=0.7, color='green', label='News Count')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Number of News Articles', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.title('Market-Wide Sentiment, Price Movement, and News Volume', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = os.path.join(self.visualizations_dir, f"market_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Market sentiment time series saved to {output_path}")
            return output_path
    
    def create_sector_sentiment_heatmap(self, integrated_data):
        """Create heatmap showing sentiment across different sectors"""
        if not integrated_data:
            return None
        
        df = pd.DataFrame(integrated_data)
        
        # For demonstration, let's create sector-like groupings based on stock characteristics
        # In real implementation, you would have actual sector data
        df['sector'] = df['stock_symbol'].apply(lambda x: 'Technology' if any(s in x for s in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']) else
                                         'Finance' if any(s in x for s in ['JPM', 'BAC', 'WFC', 'C', 'GS']) else
                                         'Healthcare' if any(s in x for s in ['PFE', 'JNJ', 'MRK', 'ABBV', 'UNH']) else
                                         'Consumer' if any(s in x for s in ['AMZN', 'WMT', 'PG', 'KO', 'PEP']) else
                                         'Other')
        
        # Calculate average sentiment by sector and date
        sector_sentiment = df.groupby(['sector', 'date'])['sentiment_score'].mean().reset_index()
        
        # Pivot for heatmap
        heatmap_data = sector_sentiment.pivot(index='sector', columns='date', values='sentiment_score')
        
        if heatmap_data.empty or len(heatmap_data) < 2:
            return None
        
        # Create heatmap
        plt.figure(figsize=(20, 8))
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlGn', 
                   center=0,
                   cbar_kws={'label': 'Sentiment Score'})
        
        plt.title('Sector Sentiment Heatmap Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sector', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(self.visualizations_dir, f"sector_sentiment_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Sector sentiment heatmap saved to {output_path}")
        return output_path
    
    def create_network_analysis(self, integrated_data):
        """Create network analysis showing relationships between stocks and news topics"""
        if not integrated_data:
            return None
        
        df = pd.DataFrame(integrated_data)
        
        # Create network graph
        G = nx.Graph()
        
        # Add stock nodes
        for symbol in df['stock_symbol'].unique():
            G.add_node(symbol, type='stock')
        
        # Add topic nodes (based on sentiment and performance)
        topics = ['positive_performer', 'negative_performer', 'neutral_performer', 'high_sentiment', 'low_sentiment']
        for topic in topics:
            G.add_node(topic, type='topic')
        
        # Add edges based on characteristics
        for _, row in df.iterrows():
            symbol = row['stock_symbol']
            sentiment = row['sentiment_score']
            performance = row['daily_change_percent']
            
            # Connect to topic nodes
            if performance > 1 and sentiment > 0.2:
                G.add_edge(symbol, 'positive_performer', weight=abs(sentiment))
            elif performance < -1 and sentiment < -0.2:
                G.add_edge(symbol, 'negative_performer', weight=abs(sentiment))
            
            if sentiment > 0.5:
                G.add_edge(symbol, 'high_sentiment', weight=sentiment)
            elif sentiment < -0.5:
                G.add_edge(symbol, 'low_sentiment', weight=abs(sentiment))
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # Position nodes
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes
        stock_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'stock']
        topic_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'topic']
        
        nx.draw_networkx_nodes(G, pos, nodelist=stock_nodes, node_size=300, node_color='blue', alpha=0.7, label='Stocks')
        nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_size=800, node_color='red', alpha=0.9, label='Topics')
        
        # Draw edges
        edges = G.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, width=[w/2 for w in weights], alpha=0.5, edge_color='gray')
        
        # Add labels
        stock_labels = {n: n for n in stock_nodes}
        topic_labels = {n: n.replace('_', ' ').title() for n in topic_nodes}
        nx.draw_networkx_labels(G, pos, stock_labels, font_size=8, font_color='white')
        nx.draw_networkx_labels(G, pos, topic_labels, font_size=10, font_weight='bold')
        
        plt.title('Stock-Topic Network Analysis', fontsize=16, fontweight='bold')
        plt.legend(['Stocks', 'Topics'], loc='best')
        plt.axis('off')
        
        output_path = os.path.join(self.visualizations_dir, f"network_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Network analysis visualization saved to {output_path}")
        return output_path
    
    def generate_research_report(self, analysis_results, prediction_results, cluster_results):
        """Generate comprehensive research report with insights and recommendations"""
        self.logger.info("📝 Generating comprehensive research report...")
        
        report = {
            'report_metadata': {
                'generated_at': get_current_timestamp(),
                'analysis_period': 'Last 30 days',
                'data_sources': ['Yahoo Finance Stock Data', 'Yahoo Finance News Articles'],
                'sample_size': {
                    'stocks_analyzed': len(set([item['stock_symbol'] for item in analysis_results.get('integrated_data', [])])),
                    'news_articles': len(analysis_results.get('integrated_data', [])),
                    'prediction_sample_size': prediction_results.get('results', {}).get('sample_size', 0) if prediction_results else 0
                }
            },
            'executive_summary': {
                'key_findings': [],
                'market_sentiment': 'neutral',
                'prediction_accuracy': prediction_results.get('results', {}).get('classification_accuracy', 0) if prediction_results else 0
            },
            'detailed_analysis': {
                'sentiment_impact': {
                    'granger_causality': analysis_results.get('granger_results', {}),
                    'correlation_strength': analysis_results.get('correlation_results', {}).get('overall_correlation', 0)
                },
                'cluster_insights': cluster_results,
                'prediction_performance': prediction_results.get('results', {}) if prediction_results else {}
            },
            'recommendations': {
                'trading_strategies': [],
                'monitoring_alerts': [],
                'research_directions': []
            }
        }
        
        # Generate executive summary
        correlation = analysis_results.get('correlation_results', {}).get('overall_correlation', 0)
        if correlation > 0.3:
            report['executive_summary']['key_findings'].append(f"Strong positive correlation ({correlation:.3f}) between news sentiment and stock performance")
            report['executive_summary']['market_sentiment'] = 'positive'
        elif correlation < -0.3:
            report['executive_summary']['key_findings'].append(f"Strong negative correlation ({correlation:.3f}) between news sentiment and stock performance")
            report['executive_summary']['market_sentiment'] = 'negative'
        else:
            report['executive_summary']['key_findings'].append(f"Weak correlation ({correlation:.3f}) between news sentiment and stock performance")
            report['executive_summary']['market_sentiment'] = 'neutral'
        
        if prediction_results and prediction_results.get('results', {}).get('classification_accuracy', 0) > 0.6:
            accuracy = prediction_results['results']['classification_accuracy']
            report['executive_summary']['key_findings'].append(f"Sentiment-based prediction model achieved {accuracy:.1%} accuracy")
        
        # Generate recommendations
        if correlation > 0.4:
            report['recommendations']['trading_strategies'].append(
                "Consider sentiment momentum strategy: Buy stocks with consistently positive news sentiment"
            )
        elif correlation < -0.4:
            report['recommendations']['trading_strategies'].append(
                "Consider contrarian strategy: Buy stocks with negative sentiment but strong fundamentals"
            )
        
        report['recommendations']['monitoring_alerts'].append(
            "Monitor stocks with high sentiment volatility for potential trading opportunities"
        )
        
        report['recommendations']['research_directions'].append(
            "Further investigate lag effects between sentiment peaks and price movements"
        )
        
        # Save report
        output_path = os.path.join(self.analysis_dir, f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_json(report, output_path)
        
        # Generate HTML report
        self.generate_html_report(report)
        
        self.logger.info(f"✅ Research report saved to {output_path}")
        return report
    
    def generate_html_report(self, report_data):
        """Generate HTML version of the research report with embedded visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Yahoo Finance Stock-News Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .section-title {{ color: #2c3e50; font-size: 24px; font-weight: bold; margin-bottom: 15px; }}
                .subsection {{ margin-left: 20px; margin-bottom: 15px; }}
                .finding {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
                .recommendation {{ background-color: #e8f4fd; border-left: 4px solid #3498db; padding: 10px 15px; margin: 10px 0; }}
                .metric {{ text-align: center; padding: 20px; margin: 10px; border-radius: 8px; }}
                .positive {{ background-color: #d4edda; color: #155724; }}
                .negative {{ background-color: #f8d7da; color: #721c24; }}
                .neutral {{ background-color: #d1ecf1; color: #0c5460; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
                .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Yahoo Finance Stock-News Analysis Report</h1>
                <p>Generated on: {report_data['report_metadata']['generated_at']}</p>
                <p>Analysis Period: {report_data['report_metadata']['analysis_period']}</p>
            </div>
            
            <div class="section">
                <div class="section-title">Executive Summary</div>
                <div class="finding">
                    <strong>Market Sentiment:</strong> <span class="metric {report_data['executive_summary']['market_sentiment']}">
                        {report_data['executive_summary']['market_sentiment'].upper()}
                    </span>
                </div>
                
                <div class="finding">
                    <strong>Prediction Accuracy:</strong> <span class="metric positive">
                        {report_data['executive_summary']['prediction_accuracy']:.1%}
                    </span>
                </div>
                
                <div class="finding">
                    <strong>Key Findings:</strong>
                    <ul>
                        {''.join([f'<li>{finding}</li>' for finding in report_data['executive_summary']['key_findings']])}
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Sentiment-Price Relationship Analysis</div>
                
                <div class="subsection">
                    <h3>Correlation Analysis</h3>
                    <p>Overall correlation between news sentiment and stock price movements: 
                        <span class="metric {'positive' if report_data['detailed_analysis']['sentiment_impact']['correlation_strength'] > 0.3 else 'negative' if report_data['detailed_analysis']['sentiment_impact']['correlation_strength'] < -0.3 else 'neutral'}">
                            {report_data['detailed_analysis']['sentiment_impact']['correlation_strength']:.3f}
                        </span>
                    </p>
                </div>
                
                <div class="subsection">
                    <h3>Granger Causality Results</h3>
                    <p>Analysis of whether sentiment predicts price movements with statistical significance.</p>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Stock Clustering Analysis</div>
                <div class="subsection">
                    <h3>Market Segmentation</h3>
                    <p>Stocks have been grouped into {len(report_data['detailed_analysis']['cluster_insights'])} distinct clusters based on sentiment-performance patterns.</p>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Predictive Model Performance</div>
                <div class="subsection">
                    <h3>Direction Prediction Accuracy</h3>
                    <p>{report_data['detailed_analysis']['prediction_performance'].get('classification_accuracy', 0):.1%}</p>
                </div>
                
                <div class="subsection">
                    <h3>Key Predictive Features</h3>
                    <p>Most important features for predicting stock movements:</p>
                    <ul>
                        {''.join([f'<li>{feature["feature"]}: {feature["importance_classification"]:.3f}</li>' 
                        for feature in report_data['detailed_analysis']['prediction_performance'].get('feature_importance', [])[:5]])}
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Recommendations</div>
                
                <div class="subsection">
                    <h3>Trading Strategies</h3>
                    {''.join([f'<div class="recommendation">{strategy}</div>' 
                    for strategy in report_data['recommendations']['trading_strategies']])}
                </div>
                
                <div class="subsection">
                    <h3>Monitoring Alerts</h3>
                    {''.join([f'<div class="recommendation">{alert}</div>' 
                    for alert in report_data['recommendations']['monitoring_alerts']])}
                </div>
                
                <div class="subsection">
                    <h3>Research Directions</h3>
                    {''.join([f'<div class="recommendation">{direction}</div>' 
                    for direction in report_data['recommendations']['research_directions']])}
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Visualizations</div>
                <div class="chart-container">
                    <h3>Sentiment-Price Time Series</h3>
                    <p>Example visualization showing the relationship between news sentiment and stock price movements over time.</p>
                </div>
                
                <div class="chart-container">
                    <h3>Sector Sentiment Heatmap</h3>
                    <p>Heatmap showing sentiment patterns across different market sectors.</p>
                </div>
                
                <div class="chart-container">
                    <h3>Stock-Topic Network</h3>
                    <p>Network analysis revealing relationships between stocks and news topics.</p>
                </div>
            </div>
            
            <div class="footer">
                <p>This report was automatically generated by the Yahoo Finance Stock-News Analysis Pipeline</p>
                <p>Data sources: Yahoo Finance Stock Data, Yahoo Finance News Articles</p>
                <p>Disclaimer: This analysis is for informational purposes only and should not be considered as financial advice.</p>
            </div>
        </body>
        </html>
        """
        
        output_path = os.path.join(self.analysis_dir, f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"✅ HTML research report saved to {output_path}")
        return output_path