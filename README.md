
# Inline Arbitrage (Live) – GB → East via ENTSO‑E

This script fetches **day‑ahead prices** from the ENTSO‑E Transparency Platform for:
GB, FR, NL, DE‑LU, PL, CZ, then computes simple route netbacks for GB→FR→DE→PL and GB→NL→DE→PL.

## Quick start

1. Get a **free ENTSO‑E API token** (Transparency Platform account → API key).
2. Put it in a `.env` file next to the script:
   ```
   ENTOS_E_TOKEN=YOUR_TOKEN_HERE
   ```
   (Either `ENTOS_E_TOKEN` or `ENTSOE_TOKEN` works.)
3. Run:
   ```bash
   python inline_arbitrage_live.py --date 2025-09-20 --save out.csv
   ```

If the token or network is missing, the script **falls back to synthetic** data so you can still test the pipeline.

## What it does

- Pulls DA prices for GB, FR, NL, DE‑LU, PL, CZ (hourly)
- Computes **netbacks** using simple losses (HVDC 2% each, AC 1% each) and fee placeholders
- Outputs a CSV with per‑hour best route & **EXPORT/HOLD** signal

## Next steps (optional, production)

- Add **ENTSO‑E physical flows/ATC** checks (border saturation flags)
- Calibrate fee/capacity costs with **JAO** auction history
- Include **imbalance risk** premia and **intraday** updates
- Persist data & build a dashboard (e.g., Streamlit) for live monitoring
