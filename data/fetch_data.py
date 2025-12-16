"""
Data Fetching and Caching Module

This module provides functions for fetching historical price data
from yfinance and caching it locally to avoid redundant API calls.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .asset_list import FULL_ASSET_UNIVERSE, SP500_TOP_100, TEST_ASSETS


# Default cache directory
CACHE_DIR = Path(__file__).parent / "cache"


def get_cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    """Get the cache file path for a ticker and date range."""
    cache_key = f"{ticker}_{start_date}_{end_date}.pkl"
    return CACHE_DIR / cache_key


def fetch_ticker_data(ticker: str, 
                      start_date: str = "2010-01-01", 
                      end_date: str = "2024-12-31",
                      use_cache: bool = True,
                      verbose: bool = False,
                      max_retries: int = 3,
                      retry_delay: float = 2.0) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data for a single ticker.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    use_cache : bool
        Whether to use cached data if available.
    verbose : bool
        Whether to print status messages.
    max_retries : int
        Maximum number of retry attempts on failure.
    retry_delay : float
        Seconds to wait between retries.
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame with OHLCV data and 'Date' column, or None if fetch failed.
        Columns: Date, Open, High, Low, Close, Volume, Adj Close
    """
    import time
    import warnings
    warnings.filterwarnings('ignore')
    
    # Check cache first
    cache_path = get_cache_path(ticker, start_date, end_date)
    
    if use_cache and cache_path.exists():
        if verbose:
            print(f"Loading {ticker} from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Fetch from yfinance with retry logic
    for attempt in range(max_retries):
        try:
            if verbose and attempt > 0:
                print(f"  Retry {attempt + 1}/{max_retries} for {ticker}...")
            
            # Use yf.download instead of Ticker.history for better reliability
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                auto_adjust=False,
                progress=False,
                threads=False  # Single thread to avoid rate limiting
            )
            
            if df.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                if verbose:
                    print(f"No data for {ticker}")
                return None
            
            # Handle multi-level columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten multi-level columns - just take the first level (Price type)
                df.columns = df.columns.get_level_values(0)
            
            # Reset index to get Date as a column
            df = df.reset_index()
            
            # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Remove timezone info if present
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            # Cache the result
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)
            
            # Small delay to avoid rate limiting for next request
            time.sleep(0.2)
            
            return df
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            if verbose:
                print(f"Error fetching {ticker}: {e}")
            return None
    
    return None


def fetch_multiple(tickers: List[str],
                   start_date: str = "2010-01-01",
                   end_date: str = "2024-12-31",
                   use_cache: bool = True,
                   verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple tickers.
    
    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    start_date, end_date : str
        Date range.
    use_cache : bool
        Whether to use cached data.
    verbose : bool
        Whether to show progress bar.
    
    Returns
    -------
    dict
        Dictionary mapping ticker -> DataFrame.
        Failed tickers are excluded from the result.
    """
    results = {}
    
    iterator = tqdm(tickers, desc="Fetching data") if verbose else tickers
    
    for ticker in iterator:
        df = fetch_ticker_data(ticker, start_date, end_date, use_cache, verbose=False)
        if df is not None:
            results[ticker] = df
        elif verbose:
            tqdm.write(f"Skipping {ticker} - no data")
    
    if verbose:
        print(f"\nSuccessfully fetched {len(results)}/{len(tickers)} tickers")
    
    return results


def fetch_all(start_date: str = "2010-01-01",
              end_date: str = "2024-12-31",
              use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for the full asset universe (100+ assets).
    
    This may take a while on first run but will be cached.
    
    Returns
    -------
    dict
        Dictionary mapping ticker -> DataFrame.
    """
    print(f"Fetching {len(FULL_ASSET_UNIVERSE)} assets from {start_date} to {end_date}...")
    return fetch_multiple(FULL_ASSET_UNIVERSE, start_date, end_date, use_cache)


def fetch_test_assets(start_date: str = "2010-01-01",
                      end_date: str = "2024-12-31",
                      use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for a small test subset (10 assets).
    
    Use this for quick testing before running full experiments.
    """
    print(f"Fetching {len(TEST_ASSETS)} test assets...")
    return fetch_multiple(TEST_ASSETS, start_date, end_date, use_cache)


def fetch_sp500_top100(start_date: str = "2010-01-01",
                       end_date: str = "2024-12-31",
                       use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for top 100 S&P 500 companies.
    """
    print(f"Fetching top {len(SP500_TOP_100)} S&P 500 companies...")
    return fetch_multiple(SP500_TOP_100, start_date, end_date, use_cache)


def generate_descriptive_statistics(data: Dict[str, pd.DataFrame],
                                    output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate descriptive statistics for the fetched data.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping ticker -> DataFrame.
    output_path : str, optional
        Path to save the statistics CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with descriptive statistics for each ticker.
    """
    stats_list = []
    
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        
        returns = df['Close'].pct_change().dropna()
        
        stats = {
            'ticker': ticker,
            'start_date': df['Date'].min().strftime('%Y-%m-%d'),
            'end_date': df['Date'].max().strftime('%Y-%m-%d'),
            'n_observations': len(df),
            'n_trading_days': len(df),
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'sharpe_approx': returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0,
            'total_return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) if len(df) > 0 else 0,
            'avg_volume': df['Volume'].mean() if 'Volume' in df.columns else 0,
        }
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    if output_path:
        stats_df.to_csv(output_path, index=False)
        print(f"Saved descriptive statistics to {output_path}")
    
    return stats_df


def clear_cache():
    """Clear all cached data files."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")
    else:
        print("No cache to clear.")


if __name__ == "__main__":
    # Quick test
    print("Testing data fetching...")
    data = fetch_test_assets()
    stats = generate_descriptive_statistics(data)
    print(stats)
