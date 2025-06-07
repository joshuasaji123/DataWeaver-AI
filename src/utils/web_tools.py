"""
Web Tools for Multi-Agent Data Analysis

This module provides web scraping and search capabilities for agents to gather
external information and enhance their analysis with real-time data.
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import re
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class WebSearchTool:
    """
    Web search and data collection tool for agents.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.driver = None
    
    def _setup_driver(self, headless: bool = True) -> webdriver.Chrome:
        """
        Set up Chrome WebDriver with appropriate options.
        
        Args:
            headless: Whether to run browser in headless mode
            
        Returns:
            Chrome WebDriver instance
        """
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"--user-agent={self.headers['User-Agent']}")
            
            # Try to create driver
            driver = webdriver.Chrome(options=chrome_options)
            return driver
            
        except Exception as e:
            print(f"Warning: Could not set up Chrome WebDriver: {e}")
            return None
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search the web for information related to the query.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of search results with title, link, and snippet
        """
        results = []
        
        try:
            # Use DuckDuckGo search (no API key required)
            search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search results
            search_results = soup.find_all('div', class_='web-result')
            
            for i, result in enumerate(search_results[:num_results]):
                try:
                    title_elem = result.find('a', class_='result__a')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    link = title_elem.get('href') if title_elem else ""
                    
                    snippet_elem = result.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else "No description"
                    
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet,
                        'rank': i + 1
                    })
                    
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"Error in web search: {e}")
            # Fallback: return mock results
            results = [{
                'title': f"Search result for: {query}",
                'link': "https://example.com",
                'snippet': f"Information related to {query} would be found here.",
                'rank': 1
            }]
        
        return results
    
    def get_financial_data(self, symbol: str, period: str = "1y") -> Dict:
        """
        Get financial data for a stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            Dictionary with financial data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period=period)
            
            # Get basic info
            info = ticker.info
            
            # Calculate basic metrics
            current_price = hist['Close'][-1] if not hist.empty else 0
            price_change = (hist['Close'][-1] - hist['Close'][-2]) if len(hist) > 1 else 0
            percent_change = (price_change / hist['Close'][-2] * 100) if len(hist) > 1 and hist['Close'][-2] != 0 else 0
            
            # Volume analysis
            avg_volume = hist['Volume'].mean() if not hist.empty else 0
            latest_volume = hist['Volume'][-1] if not hist.empty else 0
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'percent_change': round(percent_change, 2),
                'volume': int(latest_volume),
                'avg_volume': int(avg_volume),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'historical_data': hist.to_dict('records') if not hist.empty else []
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Could not fetch data: {str(e)}",
                'current_price': 0,
                'price_change': 0,
                'percent_change': 0
            }
    
    def get_news_sentiment(self, query: str, num_articles: int = 10) -> Dict:
        """
        Get news articles and perform basic sentiment analysis.
        
        Args:
            query: Search query for news
            num_articles: Number of articles to analyze
            
        Returns:
            Dictionary with news data and sentiment analysis
        """
        try:
            # Search for news articles
            news_results = self.search_web(f"{query} news", num_articles)
            
            articles = []
            positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'increase', 'profit', 'success', 'bullish', 'strong']
            negative_words = ['bad', 'poor', 'negative', 'decline', 'decrease', 'loss', 'failure', 'bearish', 'weak', 'crisis']
            
            total_sentiment = 0
            
            for article in news_results:
                # Simple sentiment analysis
                text = (article['title'] + ' ' + article['snippet']).lower()
                
                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)
                
                sentiment_score = positive_count - negative_count
                sentiment = 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
                
                articles.append({
                    'title': article['title'],
                    'snippet': article['snippet'],
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score,
                    'link': article['link']
                })
                
                total_sentiment += sentiment_score
            
            overall_sentiment = 'positive' if total_sentiment > 0 else 'negative' if total_sentiment < 0 else 'neutral'
            
            return {
                'query': query,
                'total_articles': len(articles),
                'overall_sentiment': overall_sentiment,
                'overall_sentiment_score': total_sentiment,
                'articles': articles,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'query': query,
                'error': f"Could not analyze news sentiment: {str(e)}",
                'overall_sentiment': 'neutral',
                'articles': []
            }
    
    def get_competitor_analysis(self, company: str, industry: str) -> Dict:
        """
        Get basic competitor analysis information.
        
        Args:
            company: Company name to analyze
            industry: Industry sector
            
        Returns:
            Dictionary with competitor information
        """
        try:
            # Search for competitor information
            competitors_query = f"{company} competitors {industry}"
            search_results = self.search_web(competitors_query, 10)
            
            # Extract potential competitor names (simple keyword extraction)
            competitor_keywords = []
            for result in search_results:
                text = result['title'] + ' ' + result['snippet']
                # Look for company-like words (capitalized, common endings)
                words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Co))?\b', text)
                competitor_keywords.extend(words)
            
            # Count frequency and filter
            from collections import Counter
            word_counts = Counter(competitor_keywords)
            
            # Filter out common words and the original company
            exclude_words = {'The', 'This', 'That', 'Company', 'Business', 'Market', 'Industry', company}
            potential_competitors = [word for word, count in word_counts.most_common(10) 
                                   if word not in exclude_words and count > 1]
            
            return {
                'company': company,
                'industry': industry,
                'potential_competitors': potential_competitors[:5],
                'search_results': search_results,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'company': company,
                'industry': industry,
                'error': f"Could not perform competitor analysis: {str(e)}",
                'potential_competitors': []
            }
    
    def get_market_trends(self, topic: str) -> Dict:
        """
        Get market trends and analysis for a specific topic.
        
        Args:
            topic: Topic or industry to analyze
            
        Returns:
            Dictionary with market trend information
        """
        try:
            # Search for market trends
            trends_query = f"{topic} market trends 2024 analysis"
            search_results = self.search_web(trends_query, 8)
            
            # Extract key metrics and trends
            trend_indicators = {
                'growth': 0,
                'decline': 0,
                'stable': 0,
                'emerging': 0
            }
            
            key_insights = []
            
            for result in search_results:
                text = (result['title'] + ' ' + result['snippet']).lower()
                
                # Look for trend indicators
                if any(word in text for word in ['growth', 'growing', 'increase', 'rising', 'expanding']):
                    trend_indicators['growth'] += 1
                elif any(word in text for word in ['decline', 'falling', 'decrease', 'shrinking']):
                    trend_indicators['decline'] += 1
                elif any(word in text for word in ['stable', 'steady', 'consistent']):
                    trend_indicators['stable'] += 1
                elif any(word in text for word in ['emerging', 'new', 'innovative', 'disrupting']):
                    trend_indicators['emerging'] += 1
                
                # Extract potential insights
                if len(result['snippet']) > 50:
                    key_insights.append(result['snippet'])
            
            # Determine overall trend
            max_trend = max(trend_indicators, key=trend_indicators.get)
            
            return {
                'topic': topic,
                'overall_trend': max_trend,
                'trend_indicators': trend_indicators,
                'key_insights': key_insights[:3],
                'search_results': search_results,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'topic': topic,
                'error': f"Could not analyze market trends: {str(e)}",
                'overall_trend': 'unknown',
                'trend_indicators': {}
            }
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

class SportsDataTool:
    """
    Tool for gathering sports statistics and performance data.
    """
    
    def __init__(self):
        self.web_tool = WebSearchTool()
    
    def get_team_stats(self, team_name: str, sport: str, season: str = "2024") -> Dict:
        """
        Get team statistics and performance data.
        
        Args:
            team_name: Name of the team
            sport: Sport type (basketball, football, soccer, etc.)
            season: Season year
            
        Returns:
            Dictionary with team statistics
        """
        try:
            # Search for team statistics
            query = f"{team_name} {sport} statistics {season}"
            search_results = self.web_tool.search_web(query, 5)
            
            # Extract performance indicators from search results
            performance_data = {
                'team_name': team_name,
                'sport': sport,
                'season': season,
                'search_results': search_results,
                'performance_indicators': self._extract_performance_metrics(search_results),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return performance_data
            
        except Exception as e:
            return {
                'team_name': team_name,
                'sport': sport,
                'error': f"Could not fetch team stats: {str(e)}"
            }
    
    def _extract_performance_metrics(self, search_results: List[Dict]) -> Dict:
        """Extract performance metrics from search results."""
        metrics = {
            'wins': 0,
            'losses': 0,
            'ranking_mentions': 0,
            'performance_keywords': []
        }
        
        performance_words = ['wins', 'victories', 'championships', 'rankings', 'performance', 'statistics']
        
        for result in search_results:
            text = (result['title'] + ' ' + result['snippet']).lower()
            
            # Look for performance keywords
            for word in performance_words:
                if word in text:
                    metrics['performance_keywords'].append(word)
            
            # Simple win/loss extraction (basic regex)
            wins = re.findall(r'(\d+)\s*wins?', text)
            losses = re.findall(r'(\d+)\s*losses?', text)
            
            if wins:
                metrics['wins'] = max(metrics['wins'], int(wins[0]))
            if losses:
                metrics['losses'] = max(metrics['losses'], int(losses[0]))
        
        return metrics

# Factory function for creating appropriate tools
def create_web_tool(tool_type: str):
    """
    Factory function to create appropriate web tools.
    
    Args:
        tool_type: Type of tool to create ('search', 'financial', 'sports')
        
    Returns:
        Appropriate tool instance
    """
    if tool_type == 'search':
        return WebSearchTool()
    elif tool_type == 'financial':
        return WebSearchTool()  # Financial capabilities are built into WebSearchTool
    elif tool_type == 'sports':
        return SportsDataTool()
    else:
        return WebSearchTool()  # Default to general web search 