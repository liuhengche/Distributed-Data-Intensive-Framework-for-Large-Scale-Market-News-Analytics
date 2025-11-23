import logging
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime, timezone
from config import BASE_DIR
from utils import get_current_timestamp, save_json
import os 

class SentimentAnalyzer:
    """Perform sentiment analysis using FinBERT"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing sentiment analyzer...")
        
        self.use_finbert = False
        self.use_vader = False
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Check if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            self.logger.info(f"{'GPU' if device == 0 else 'CPU'} will be used for sentiment analysis")
            
            # Load FinBERT model
            self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            
            self.use_finbert = True
            self.logger.info("FinBERT sentiment analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing FinBERT: {e}")
            self.logger.info("Falling back to VADER sentiment analysis")
            
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                
                self.analyzer = SentimentIntensityAnalyzer()
                self.use_vader = True
                self.logger.info("VADER sentiment analyzer initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Error initializing VADER: {e}")
                self.logger.warning("No sentiment analysis available. Will skip sentiment analysis.")
    
    def analyze_sentiment(self, text, max_length=512):
        """Analyze sentiment of text using FinBERT or VADER as fallback"""
        if not text or len(text.strip()) < 10:
            return {"label": "neutral", "score": 0.0}
        
        try:
            if self.use_vader:
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
            
            elif self.use_finbert:
                # Use FinBERT
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    text = self.tokenizer.convert_tokens_to_string(tokens)
                
                result = self.sentiment_pipeline(text, truncation=True, max_length=max_length)[0]
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
            self.logger.warning(f"Error during sentiment analysis: {e}")
            return {"label": "neutral", "score": 0.0}
    
    def analyze_articles(self, articles):
        """Analyze sentiment for multiple articles"""
        self.logger.info(f"Analyzing sentiment for {len(articles)} articles...")
        
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
            
            if i > 0 and i % 10 == 0:
                time.sleep(1)
        
        self.logger.info("Sentiment analysis completed")
        return results
    
    def save_sentiment_results(self, results, filename=None):
        """Save sentiment analysis results"""
        if not results:
            self.logger.warning("No sentiment results to save")
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
        
        save_json(summary, filepath)
        
        self.logger.info(f"Sentiment analysis results saved to {filepath}")
        self.logger.info(f"Sentiment distribution: Positive: {len(positive)}, Negative: {len(negative)}, Neutral: {len(neutral)}")
        
        return filepath