import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import logging
from datetime import datetime
from config import BASE_DIR

class Visualizer:
    """Create visualizations for stock and news data analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = os.path.join(BASE_DIR, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_sentiment_distribution_chart(self, sentiment_data, filename=None):
        """Create pie chart and bar chart for sentiment distribution"""
        if not filename:
            filename = f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load data if it's a file path
        if isinstance(sentiment_data, str):
            with open(sentiment_data, 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
        
        # Get sentiment counts
        positive = sentiment_data.get('positive_count', 0)
        negative = sentiment_data.get('negative_count', 0)
        neutral = sentiment_data.get('neutral_count', 0)
        total = positive + negative + neutral
        
        if total == 0:
            self.logger.warning("No sentiment data to visualize")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive, negative, neutral]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('News Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        x = np.arange(3)
        width = 0.6
        
        bars = ax2.bar(x, sizes, width, color=colors)
        ax2.set_ylabel('Number of Articles', fontsize=12)
        ax2.set_title('Sentiment Count by Category', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Sentiment distribution chart saved to {self.output_dir}/{filename}.png")
        return os.path.join(self.output_dir, f"{filename}.png")
    
    def create_stock_performance_chart(self, stock_data, filename=None):
        """Create charts for stock performance analysis"""
        if not filename:
            filename = f"stock_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load data if it's a file path
        if isinstance(stock_data, str):
            with open(stock_data, 'r', encoding='utf-8') as f:
                stock_data = json.load(f)
        
        if not stock_data or len(stock_data) < 2:
            self.logger.warning("Insufficient stock data for visualization")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(stock_data)
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Top Gainers vs Losers
        top_gainers = df.nlargest(5, 'change_percent')
        top_losers = df.nsmallest(5, 'change_percent')
        
        ax1.barh(top_gainers['symbol'], top_gainers['change_percent'], color='green', alpha=0.7)
        ax1.barh(top_losers['symbol'], top_losers['change_percent'], color='red', alpha=0.7)
        ax1.set_title('Top 5 Gainers & Losers', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Daily Change (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Price Distribution
        prices = df['price'].dropna()
        if not prices.empty:
            sns.histplot(prices, kde=True, ax=ax2, color='blue', alpha=0.6)
            ax2.set_title('Stock Price Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Price ($)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
        
        # 3. Market Cap vs Volume
        market_caps = df['market_cap'].str.replace('[MBT]', '', regex=True).astype(float)
        volumes = df['volume'].str.replace('[MBT]', '', regex=True).astype(float)
        
        if not market_caps.empty and not volumes.empty:
            scatter = ax3.scatter(volumes, market_caps, 
                                c=df['change_percent'], 
                                s=50, alpha=0.7, 
                                cmap='RdYlGn',
                                vmin=-5, vmax=5)
            ax3.set_title('Market Cap vs Trading Volume', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Volume', fontsize=12)
            ax3.set_ylabel('Market Cap', fontsize=12)
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            plt.colorbar(scatter, ax=ax3, label='Daily Change (%)')
        
        # 4. P/E Ratio Analysis
        pe_ratios = df[df['pe_ratio'] > 0]['pe_ratio']
        if not pe_ratios.empty and len(pe_ratios) >= 2:
            sns.boxplot(y=pe_ratios, ax=ax4, color='purple', width=0.5)
            ax4.set_title('P/E Ratio Distribution', fontsize=14, fontweight='bold')
            ax4.set_ylabel('P/E Ratio', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Stock performance charts saved to {self.output_dir}/{filename}.png")
        return os.path.join(self.output_dir, f"{filename}.png")
    
    def create_correlation_visualization(self, correlation_data, filename=None):
        """Create correlation analysis visualizations"""
        if not filename:
            filename = f"correlation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load data if it's a file path
        if isinstance(correlation_data, str):
            with open(correlation_data, 'r', encoding='utf-8') as f:
                correlation_data = json.load(f)
        
        if not correlation_data:
            self.logger.warning("No correlation data to visualize")
            return
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Overall correlation text
        correlation = correlation_data.get('overall_correlation', 0)
        correlation_text = f"Overall Sentiment-Price Correlation: {correlation:.3f}"
        
        ax1.text(0.5, 0.5, correlation_text, 
                ha='center', va='center', 
                fontsize=16, fontweight='bold',
                bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=1'))
        ax1.set_title('Correlation Coefficient', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Sentiment distribution
        sentiment_dist = correlation_data.get('sentiment_distribution', {})
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [
            sentiment_dist.get('positive', 0),
            sentiment_dist.get('negative', 0),
            sentiment_dist.get('neutral', 0)
        ]
        
        if sum(sizes) > 0:
            colors = ['#2ecc71', '#e74c3c', '#3498db']
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('News Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 3. Positive correlations
        positive_corr = correlation_data.get('positive_correlations', {}).get('stocks', [])
        if positive_corr:
            pos_df = pd.DataFrame(positive_corr)
            if len(pos_df) > 0:
                # Show top 10
                pos_df = pos_df.head(10)
                ax3.barh(pos_df['stock_symbol'], pos_df['sentiment_score'], color='green', alpha=0.7)
                ax3.set_title('Stocks with Positive Sentiment & Performance', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Average Sentiment Score', fontsize=12)
                ax3.grid(True, alpha=0.3)
        
        # 4. Negative correlations
        negative_corr = correlation_data.get('negative_correlations', {}).get('stocks', [])
        if negative_corr:
            neg_df = pd.DataFrame(negative_corr)
            if len(neg_df) > 0:
                # Show top 10
                neg_df = neg_df.head(10)
                ax4.barh(neg_df['stock_symbol'], neg_df['sentiment_score'], color='red', alpha=0.7)
                ax4.set_title('Stocks with Negative Sentiment & Performance', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Average Sentiment Score', fontsize=12)
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Correlation analysis charts saved to {self.output_dir}/{filename}.png")
        return os.path.join(self.output_dir, f"{filename}.png")
    
    def create_interactive_dashboard(self, integrated_data_path, correlation_data_path, output_filename=None):
        """Create interactive Plotly dashboard"""
        if not output_filename:
            output_filename = f"interactive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Load data
        with open(integrated_data_path, 'r', encoding='utf-8') as f:
            integrated_data = json.load(f)
        
        with open(correlation_data_path, 'r', encoding='utf-8') as f:
            correlation_data = json.load(f)
        
        if not integrated_data:
            self.logger.warning("No integrated data for interactive dashboard")
            return
        
        df = pd.DataFrame(integrated_data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment vs Stock Performance', 'Sentiment Distribution', 
                          'Top Performing Stocks', 'Stock Mentions'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Scatter plot: Sentiment vs Performance
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=df['sentiment_score'],
                    y=df['daily_change_percent'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df['current_price'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Price ($)'),
                        opacity=0.7
                    ),
                    text=df['stock_symbol'] + '<br>' + df['news_headline'].str[:50] + '...',
                    hoverinfo='text+x+y',
                    name='Stocks'
                ),
                row=1, col=1
            )
            
            # Add correlation line
            if len(df) > 1:
                z = np.polyfit(df['sentiment_score'], df['daily_change_percent'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(df['sentiment_score'].min(), df['sentiment_score'].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name=f'Correlation: {correlation_data.get("overall_correlation", 0):.3f}'
                    ),
                    row=1, col=1
                )
        
        # 2. Pie chart: Sentiment distribution
        sentiment_counts = df['news_sentiment'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker=dict(colors=['#2ecc71', '#e74c3c', '#3498db']),
                textinfo='percent+label',
                hole=0.3
            ),
            row=1, col=2
        )
        
        # 3. Bar chart: Top performing stocks
        if not df.empty:
            top_performers = df.groupby('stock_symbol').agg({
                'daily_change_percent': 'mean',
                'sentiment_score': 'mean',
                'stock_name': 'first'
            }).sort_values('daily_change_percent', ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(
                    x=top_performers.index,
                    y=top_performers['daily_change_percent'],
                    marker_color=np.where(top_performers['daily_change_percent'] > 0, 'green', 'red'),
                    text=top_performers['stock_name'],
                    hoverinfo='text+y',
                    name='Daily Change'
                ),
                row=2, col=1
            )
        
        # 4. Bar chart: Most mentioned stocks
        stock_mentions = df['stock_symbol'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=stock_mentions.index,
                y=stock_mentions.values,
                marker_color='blue',
                name='Mention Count'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text='Stock News Sentiment Analysis Dashboard',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_yaxes(title_text="Daily Change (%)", row=1, col=1)
        fig.update_xaxes(title_text="Stock Symbol", row=2, col=1)
        fig.update_yaxes(title_text="Average Daily Change (%)", row=2, col=1)
        fig.update_xaxes(title_text="Stock Symbol", row=2, col=2)
        fig.update_yaxes(title_text="Number of Mentions", row=2, col=2)
        
        # Save to HTML
        output_path = os.path.join(self.output_dir, output_filename)
        fig.write_html(output_path)
        
        self.logger.info(f"Interactive dashboard saved to {output_path}")
        return output_path
    
    def create_daily_summary_report(self, pipeline_summary, stock_summary_path, news_summary_path, sentiment_summary_path, correlation_path):
        """Create a comprehensive daily summary report with visualizations"""
        self.logger.info("Creating daily summary report...")
        
        # Load all data
        with open(pipeline_summary, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        with open(stock_summary_path, 'r', encoding='utf-8') as f:
            stock_data = json.load(f)

        with open(news_summary_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)

        with open(sentiment_summary_path, 'r', encoding='utf-8') as f:
            sentiment_data = json.load(f)

        with open(correlation_path, 'r', encoding='utf-8') as f:
            correlation_data = json.load(f)
        
        # Create figure with multiple sections
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Pipeline summary (text)
        ax1 = plt.subplot(3, 2, 1)
        summary_text = f"""
        Pipeline Summary
        ----------------
        Execution Time: {pipeline_data.get('pipeline_execution_time_seconds', 0):.2f} seconds
        Stocks Crawled: {pipeline_data.get('stocks_crawled', 0)}
        News Articles: {pipeline_data.get('news_articles_crawled', 0)}
        Integrated Pairs: {pipeline_data.get('stock_news_integrations', 0)}
        
        Key Metrics:
        • Avg Stock Change: {stock_data.get('average_change_percent', 0):.2f}%
        • Positive Sentiment: {sentiment_data.get('positive_percentage', 0):.1f}%
        • Correlation: {correlation_data.get('overall_correlation', 0):.3f}
        """
        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, 
                fontsize=10, fontfamily='monospace', 
                bbox=dict(facecolor='lightgray', alpha=0.3))
        ax1.set_title('Pipeline Summary', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Stock performance summary
        ax2 = plt.subplot(3, 2, 2)
        if 'top_gainers' in stock_data and 'top_losers' in stock_data:
            gainers = stock_data['top_gainers'][:3]
            losers = stock_data['top_losers'][:3]
            
            y = np.arange(3)
            width = 0.35
            
            ax2.barh(y - width/2, [g['change_percent'] for g in gainers], width, color='green', alpha=0.7, label='Gainers')
            ax2.barh(y + width/2, [l['change_percent'] for l in losers], width, color='red', alpha=0.7, label='Losers')
            
            ax2.set_yticks(y)
            ax2.set_yticklabels([f"{g['symbol']}" for g in gainers])
            ax2.set_xlabel('Daily Change (%)', fontsize=10)
            ax2.set_title('Top Gainers vs Losers', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Sentiment distribution
        ax3 = plt.subplot(3, 2, 3)
        if 'positive_count' in sentiment_data:
            labels = ['Positive', 'Negative', 'Neutral']
            sizes = [
                sentiment_data['positive_count'],
                sentiment_data['negative_count'], 
                sentiment_data['neutral_count']
            ]
            colors = ['#2ecc71', '#e74c3c', '#3498db']
            
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('News Sentiment Distribution', fontsize=12, fontweight='bold')
        
        # 4. Most mentioned stocks
        ax4 = plt.subplot(3, 2, 4)
        if 'most_mentioned_tickers' in news_data:
            top_mentions = news_data['most_mentioned_tickers'][:5]
            if top_mentions:
                ax4.bar([m['ticker'] for m in top_mentions], [m['count'] for m in top_mentions], 
                       color='blue', alpha=0.7)
                ax4.set_title('Most Mentioned Stocks', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Mention Count', fontsize=10)
                ax4.grid(True, alpha=0.3)
        
        # 5. Sentiment vs Performance
        ax5 = plt.subplot(3, 2, 5)
        if 'positive_correlations' in correlation_data and 'negative_correlations' in correlation_data:
            pos_count = correlation_data['positive_correlations']['count']
            neg_count = correlation_data['negative_correlations']['count']
            
            labels = ['Positive\nCorrelation', 'Negative\nCorrelation']
            values = [pos_count, neg_count]
            colors = ['green', 'red']
            
            bars = ax5.bar(labels, values, color=colors, alpha=0.7)
            ax5.set_title('Sentiment-Performance Correlations', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Number of Stocks', fontsize=10)
            ax5.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 6. Overall correlation
        ax6 = plt.subplot(3, 2, 6)
        correlation = correlation_data.get('overall_correlation', 0)
        correlation_color = 'green' if correlation > 0.3 else 'red' if correlation < -0.3 else 'blue'
        
        ax6.text(0.5, 0.5, f'Overall Correlation\n{correlation:.3f}', 
                ha='center', va='center', 
                fontsize=14, fontweight='bold',
                color=correlation_color,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=correlation_color, linewidth=2))
        ax6.set_title('Sentiment-Price Correlation', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(self.output_dir, f"daily_summary_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Daily summary report saved to {output_path}")
        return output_path