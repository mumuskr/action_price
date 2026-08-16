# Raw market data

Market data files are local research inputs and are not committed to Git.

Current market-data source:

- Provider: Interactive Brokers historical market data through TWS
- Frequency: 1 minute
- Market: US stocks
- Price source: trades
- Session: regular trading hours only
- Adjustment: IBKR `TRADES` bars, without a separate adjustment step

Validated Parquet files are written under `data/processed/`. IBKR contract and
request provenance is stored in each Parquet schema's metadata.

Downloading requires a logged-in TWS or IB Gateway session and the appropriate
historical market-data permission. Account credentials are never stored by this
project.
