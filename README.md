# Inline Arbitrage (Live) - GB to East via ENTSO-E

This script fetches day-ahead prices from the ENTSO-E Transparency Platform for
GB, FR, NL, DE-LU, PL, CZ, then computes simple route netbacks for:

- `GB->FR->DE->PL`
- `GB->NL->DE->PL`

Key behavior:

- Uses real ENTSO-E area EIC codes, not UI labels like `BZN|GB`
- Queries by local market day for each zone, then converts timestamps to UTC
- Converts GB prices from GBP to EUR using a user-supplied FX rate
- Scores routes leg-by-leg and blocks a route when any border leg is underwater
- Uses synthetic data only when you ask for it with `--dry`

## Quick start

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then put your ENTSO-E token in it:

   ```
   ENTOS_E_TOKEN=YOUR_TOKEN_HERE
   GBP_EUR=1.17
   ```

   Either `ENTOS_E_TOKEN` or `ENTSOE_TOKEN` works for the token.
   Either `GBP_EUR` or `GBP_EUR_RATE` works for the FX rate.

3. Run a real pull for one market day:

   ```bash
   python inline_arbitrage_live.py --date 2025-09-20 --save out.csv
   ```

4. Or run synthetic demo data explicitly:

   ```bash
   python inline_arbitrage_live.py --date 2025-09-20 --dry
   ```

## Notes

- ENTSO-E day-ahead prices are a local market-day product, not a generic rolling UTC feed.
- GB prices are published in GBP on ENTSO-E, so an FX rate is required to compare them
  against EUR-denominated continental markets.
- Some zones may publish sub-hourly prices. The script normalizes sub-hourly data to
  hourly means before calculating the simple route netbacks.
- Route scores are heuristics based on per-leg spreads, losses, and fees. They are
  stricter than a simple `PL - GB` spread because a negative intermediate leg blocks
  the route for that hour.

## Next steps

- Add physical flow and ATC checks
- Calibrate fee and capacity costs with auction history
- Add imbalance-risk premia and intraday updates
- Persist data and build a dashboard for live monitoring
