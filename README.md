# Inline Arbitrage (Live) - GB via Elexon, Continent via ENTSO-E

This script fetches GB market-index prices from Elexon MID and continental
day-ahead prices from the ENTSO-E Transparency Platform for FR, NL, DE-LU, PL, CZ,
then computes simple route netbacks for:

- `GB->FR->DE->PL`
- `GB->NL->DE->PL`

Key behavior:

- Uses Elexon MID for GB and real ENTSO-E area EIC codes for the continental zones
- Queries by local market day for each zone, then converts timestamps to UTC
- Converts GB prices from GBP to EUR using a user-supplied FX rate
- Scores routes leg-by-leg and blocks a route when any border leg is underwater
- Uses synthetic data only when you ask for it with `--dry`

## Quick start

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then put your API credentials and FX rate in it:

   ```
   ENTOS_E_TOKEN=YOUR_TOKEN_HERE
   BMRS_API_KEY=YOUR_BMRS_API_KEY_HERE
   GBP_EUR=1.17
   ```

   `BMRS_API_KEY` is read exactly by that name if you have an Iris/Elexon key.
   Either `ENTOS_E_TOKEN` or `ENTSOE_TOKEN` works for the ENTSO-E token.
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

- GB prices come from the Elexon market-index feed and are published in GBP, so an FX rate
  is required to compare them against EUR-denominated continental markets.
- Continental ENTSO-E day-ahead prices are a local market-day product, not a generic rolling UTC feed.
- The GB feed is half-hourly and is normalized to hourly means before route calculation.
- The default GB provider is `APXMIDP`, which is the only provider returning non-zero live
  rows in the sample query I checked on March 10, 2026.
- If Elexon changes authentication behavior, the script will send `BMRS_API_KEY` when present.
- Some continental zones may publish sub-hourly prices. The script normalizes sub-hourly data to
  hourly means before calculating the simple route netbacks.
- Route scores are heuristics based on per-leg spreads, losses, and fees. They are
  stricter than a simple `PL - GB` spread because a negative intermediate leg blocks
  the route for that hour.

## Next steps

- Add physical flow and ATC checks
- Calibrate fee and capacity costs with auction history
- Add imbalance-risk premia and intraday updates
- Persist data and build a dashboard for live monitoring
