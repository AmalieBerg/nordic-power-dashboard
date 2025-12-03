# Nordic Power Price Dashboard

**Production-ready dashboard for analyzing Nordic electricity prices with GARCH volatility forecasting**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project demonstrates production-quality skills in:
- **Data Engineering**: Automated data pipelines with ENTSO-E API
- **Database Design**: SQLite with proper schema, indexes, and UPSERT logic
- **Quantitative Finance**: GARCH volatility forecasting for power prices
- **Web Development**: Interactive Streamlit dashboard
- **DevOps**: GitHub Actions for automated data updates

**Target users:** Quantitative analysts, power traders, and energy companies in Nordic markets

---

## 📊 Features

### Current (Week 2 - Complete)
- ✅ Production-ready ENTSO-E API client with retry logic
- ✅ SQLite database for price storage
- ✅ Configuration management with environment variables
- ✅ Error handling and logging
- ✅ Support for all Norwegian bidding zones (NO1-NO5)

### Upcoming
- ⏳ GARCH(1,1) volatility forecasting (Week 3)
- ⏳ Interactive Streamlit dashboard (Week 4)
- ⏳ Automated data updates via GitHub Actions (Week 2-3)
- ⏳ Price spike detection and alerts

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- ENTSO-E API token (free registration at https://transparency.entsoe.eu/)

### Installation

```bash
# 1. Clone or download this project
cd nordic-power-dashboard

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API token
# Create .env file with:
ENTSOE_API_TOKEN=your-token-here

# 5. Test everything works
python src/utils/config.py
python src/data/entsoe_client.py
python src/data/database.py
```

---

## 📁 Project Structure

```
nordic-power-dashboard/
├── src/                          # Source code
│   ├── data/                     # Data ingestion and storage
│   │   ├── entsoe_client.py     # API client with retry logic
│   │   ├── database.py          # SQLite database layer
│   │   └── fetcher.py           # Orchestration (coming soon)
│   ├── models/                   # Forecasting models
│   │   └── garch_model.py       # GARCH volatility (Week 3)
│   ├── dashboard/                # Streamlit app
│   │   └── app.py               # Dashboard UI (Week 4)
│   └── utils/                    # Utilities
│       └── config.py            # Configuration management
│
├── data/                         # Database storage
│   └── prices.db                # SQLite database (created automatically)
│
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit tests
│
├── .env                          # Environment variables (create this!)
├── .gitignore                   # Git exclusions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 💻 Usage Examples

### Fetch and Store Prices

```python
from src.data import EntsoeAPIClient, PriceDatabase
from src.utils import Config
import pandas as pd

# Initialize
client = EntsoeAPIClient(api_key=Config.ENTSOE_API_TOKEN)
db = PriceDatabase()

# Fetch 30 days of Bergen prices
end = pd.Timestamp.now(tz='Europe/Oslo')
start = end - pd.Timedelta(days=30)
prices = client.fetch_day_ahead_prices('NO_2', start, end)

# Store in database
db.insert_prices('NO_2', prices)

# Query back
stored_prices = db.get_prices('NO_2', start, end)
print(f"Mean price: {stored_prices.mean():.2f} EUR/MWh")
```

### Database Statistics

```python
from src.data import PriceDatabase

db = PriceDatabase()
stats = db.get_database_stats()

print(f"Total records: {stats['total_records']}")
print(f"Database size: {stats['file_size_mb']:.2f} MB")
print(f"Zones: {list(stats['records_per_zone'].keys())}")
```

---

## 🧪 Testing

```bash
# Test API client
python src/data/entsoe_client.py

# Test database layer
python src/data/database.py

# Test configuration
python src/utils/config.py

# Run unit tests (coming soon)
pytest tests/
```

---

## 📈 Development Roadmap

### Week 1: Planning ✅ (Complete)
- [x] User stories and requirements
- [x] Technical architecture design
- [x] ENTSO-E API research and testing

### Week 2: Data Pipeline 🔨 (In Progress)
- [x] Production API client
- [x] Configuration management
- [x] Database layer
- [ ] Data fetcher orchestration
- [ ] GitHub Actions automation

### Week 3: Forecasting ⏳ (Planned)
- [ ] GARCH(1,1) volatility model
- [ ] Backtesting framework
- [ ] Model evaluation metrics

### Week 4: Dashboard ⏳ (Planned)
- [ ] Streamlit multi-page app
- [ ] Current prices view
- [ ] Historical charts
- [ ] Volatility forecasts
- [ ] Production deployment

---

## 🔧 Configuration

All configuration is managed through environment variables in `.env`:

```bash
# Required
ENTSOE_API_TOKEN=your-token-here

# Optional (defaults provided)
HISTORICAL_YEARS=2
UPDATE_FREQUENCY_HOURS=6
```

See `src/utils/config.py` for all available settings.

---

## 📊 Data Sources

- **Price Data**: ENTSO-E Transparency Platform (free API)
- **Coverage**: All Norwegian bidding zones (NO1-NO5)
- **Frequency**: Hourly day-ahead prices
- **Historical**: 2+ years available
- **Update**: Published daily at ~12:42 CET

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👤 Author

**Amalie Berg**
- LinkedIn: [linkedin.com/in/amalie-berg](https://linkedin.com/in/amalie-berg)
- Location: Bergen, Norway
- Focus: Quantitative analysis in Nordic energy trading

---

## 🙏 Acknowledgments

- ENTSO-E for providing free access to European power market data
- `entsoe-py` library maintainers
- Norwegian School of Economics (NHH)

---

## 📞 Contact

For questions about this project, reach out via LinkedIn or GitHub issues.

---

**Last updated:** November 2024  
**Status:** Active development (Week 2 of 4)
