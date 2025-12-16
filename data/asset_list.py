"""
Asset Universe Definitions

This module defines the asset universe for the expanded empirical study.
Contains top 100 S&P 500 companies by market cap plus sector ETFs,
international ETFs, bonds, and commodities.
"""

# Top 100 S&P 500 by market cap (as of 2024)
SP500_TOP_100 = [
    # Top 10 - Mega caps
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Alphabet Class A
    'AMZN',   # Amazon
    'NVDA',   # NVIDIA
    'META',   # Meta Platforms
    'BRK-B',  # Berkshire Hathaway
    'TSLA',   # Tesla
    'UNH',    # UnitedHealth
    'LLY',    # Eli Lilly
    
    # 11-20
    'JPM',    # JPMorgan Chase
    'V',      # Visa
    'XOM',    # Exxon Mobil
    'JNJ',    # Johnson & Johnson
    'MA',     # Mastercard
    'AVGO',   # Broadcom
    'PG',     # Procter & Gamble
    'HD',     # Home Depot
    'CVX',    # Chevron
    'MRK',    # Merck
    
    # 21-30
    'COST',   # Costco
    'ABBV',   # AbbVie
    'PEP',    # PepsiCo
    'KO',     # Coca-Cola
    'ADBE',   # Adobe
    'WMT',    # Walmart
    'MCD',    # McDonald's
    'CRM',    # Salesforce
    'CSCO',   # Cisco
    'BAC',    # Bank of America
    
    # 31-40
    'TMO',    # Thermo Fisher
    'ACN',    # Accenture
    'NFLX',   # Netflix
    'AMD',    # AMD
    'LIN',    # Linde
    'ABT',    # Abbott Labs
    'ORCL',   # Oracle
    'DHR',    # Danaher
    'INTC',   # Intel
    'DIS',    # Disney
    
    # 41-50
    'WFC',    # Wells Fargo
    'VZ',     # Verizon
    'PM',     # Philip Morris
    'INTU',   # Intuit
    'CMCSA',  # Comcast
    'NKE',    # Nike
    'RTX',    # RTX Corp (Raytheon)
    'TXN',    # Texas Instruments
    'IBM',    # IBM
    'QCOM',   # Qualcomm
    
    # 51-60
    'NOW',    # ServiceNow
    'SPGI',   # S&P Global
    'HON',    # Honeywell
    'AMGN',   # Amgen
    'GE',     # General Electric
    'CAT',    # Caterpillar
    'AMAT',   # Applied Materials
    'PFE',    # Pfizer
    'LOW',    # Lowe's
    'UNP',    # Union Pacific
    
    # 61-70
    'GS',     # Goldman Sachs
    'BKNG',   # Booking Holdings
    'ISRG',   # Intuitive Surgical
    'ELV',    # Elevance Health
    'MS',     # Morgan Stanley
    'MDT',    # Medtronic
    'T',      # AT&T
    'SYK',    # Stryker
    'ADP',    # Automatic Data Processing
    'BLK',    # BlackRock
    
    # 71-80
    'VRTX',   # Vertex Pharma
    'DE',     # Deere
    'GILD',   # Gilead
    'C',      # Citigroup
    'REGN',   # Regeneron
    'ZTS',    # Zoetis
    'ADI',    # Analog Devices
    'LRCX',   # Lam Research
    'SCHW',   # Charles Schwab
    'MMC',    # Marsh McLennan
    
    # 81-90
    'CB',     # Chubb
    'MO',     # Altria
    'PANW',   # Palo Alto Networks
    'SLB',    # Schlumberger
    'BMY',    # Bristol-Myers
    'ETN',    # Eaton
    'FI',     # Fiserv
    'CME',    # CME Group
    'SNPS',   # Synopsys
    'KLAC',   # KLA Corp
    
    # 91-100
    'CDNS',   # Cadence Design
    'BSX',    # Boston Scientific
    'CI',     # Cigna
    'DUK',    # Duke Energy
    'EOG',    # EOG Resources
    'SO',     # Southern Company
    'ICE',    # Intercontinental Exchange
    'MU',     # Micron
    'PNC',    # PNC Financial
    'USB',    # U.S. Bancorp
]

# Sector ETFs (Select Sector SPDRs)
SECTOR_ETFS = [
    'XLF',    # Financial Select
    'XLK',    # Technology Select
    'XLV',    # Health Care Select
    'XLE',    # Energy Select
    'XLI',    # Industrial Select
    'XLY',    # Consumer Discretionary
    'XLP',    # Consumer Staples
    'XLU',    # Utilities Select
    'XLB',    # Materials Select
    'XLRE',   # Real Estate Select
    'XLC',    # Communication Services
]

# International ETFs
INTERNATIONAL_ETFS = [
    'EFA',    # iShares MSCI EAFE (Developed Markets ex-US)
    'EEM',    # iShares MSCI Emerging Markets
    'VEU',    # Vanguard All-World ex-US
]

# Bond ETFs
BOND_ETFS = [
    'TLT',    # iShares 20+ Year Treasury
    'IEF',    # iShares 7-10 Year Treasury
    'AGG',    # iShares Core US Aggregate Bond
    'BND',    # Vanguard Total Bond Market
]

# Commodity ETFs
COMMODITY_ETFS = [
    'GLD',    # SPDR Gold
    'SLV',    # iShares Silver
    'USO',    # US Oil Fund
    'DBA',    # Agriculture ETF
]

# Full asset universe for the study
FULL_ASSET_UNIVERSE = (
    SP500_TOP_100 + 
    SECTOR_ETFS + 
    INTERNATIONAL_ETFS + 
    BOND_ETFS + 
    COMMODITY_ETFS
)

# Default subset for testing
TEST_ASSETS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech mega caps
    'JPM', 'BAC', 'GS',  # Financials
    'XOM', 'CVX',  # Energy
]


def get_assets_by_sector(sector):
    """
    Get assets by sector.
    
    Parameters
    ----------
    sector : str
        One of: 'tech', 'finance', 'healthcare', 'energy', 'consumer',
        'industrial', 'etf', 'bond', 'commodity', 'international'
    
    Returns
    -------
    list of str
        Ticker symbols in that sector.
    """
    sectors = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'AVGO', 
                 'ADBE', 'CRM', 'CSCO', 'AMD', 'ORCL', 'INTC', 'TXN', 
                 'QCOM', 'NOW', 'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MU'],
        'finance': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 
                    'BLK', 'SCHW', 'C', 'PNC', 'USB', 'SPGI', 'MMC', 'CB', 'ICE'],
        'healthcare': ['UNH', 'LLY', 'JNJ', 'MRK', 'ABBV', 'PFE', 'TMO', 
                       'ABT', 'DHR', 'AMGN', 'ISRG', 'ELV', 'MDT', 'SYK', 
                       'VRTX', 'GILD', 'REGN', 'ZTS', 'BMY', 'BSX', 'CI'],
        'energy': ['XOM', 'CVX', 'SLB', 'EOG'],
        'consumer': ['COST', 'PEP', 'KO', 'WMT', 'MCD', 'PG', 'HD', 'LOW', 
                     'NKE', 'NFLX', 'DIS', 'BKNG'],
        'industrial': ['CAT', 'DE', 'HON', 'UNP', 'GE', 'RTX', 'ETN', 'LIN'],
        'etf_sector': SECTOR_ETFS,
        'etf_international': INTERNATIONAL_ETFS,
        'etf_bond': BOND_ETFS,
        'etf_commodity': COMMODITY_ETFS,
    }
    
    if sector not in sectors:
        raise ValueError(f"Unknown sector: {sector}. Available: {list(sectors.keys())}")
    
    return sectors[sector]
