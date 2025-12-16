"""
Data Module

Provides data fetching and asset universe definitions.
"""

from .asset_list import (
    SP500_TOP_100,
    SECTOR_ETFS,
    INTERNATIONAL_ETFS,
    BOND_ETFS,
    COMMODITY_ETFS,
    FULL_ASSET_UNIVERSE,
    TEST_ASSETS,
    get_assets_by_sector
)
from .fetch_data import (
    fetch_ticker_data,
    fetch_multiple,
    fetch_all,
    fetch_test_assets,
    fetch_sp500_top100,
    generate_descriptive_statistics,
    clear_cache
)

__all__ = [
    'SP500_TOP_100',
    'SECTOR_ETFS', 
    'INTERNATIONAL_ETFS',
    'BOND_ETFS',
    'COMMODITY_ETFS',
    'FULL_ASSET_UNIVERSE',
    'TEST_ASSETS',
    'get_assets_by_sector',
    'fetch_ticker_data',
    'fetch_multiple',
    'fetch_all',
    'fetch_test_assets',
    'fetch_sp500_top100',
    'generate_descriptive_statistics',
    'clear_cache'
]
