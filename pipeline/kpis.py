import pandas as pd

def calculate_kpis(df: pd.DataFrame) -> dict:
    kpis = {}

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    try:
        # Burn rate = average monthly net cash outflow
        if "net_cash_flow" in df.columns:
            negative_flows = df[df["net_cash_flow"] < 0]["net_cash_flow"]
            kpis["burn_rate"] = abs(negative_flows.mean()) if not negative_flows.empty else 0
        elif "total_expenses" in df.columns and "total_revenue" in df.columns:
            df["net"] = df["total_revenue"] - df["total_expenses"]
            negative = df[df["net"] < 0]["net"]
            kpis["burn_rate"] = abs(negative.mean()) if not negative.empty else 0
        else:
            kpis["burn_rate"] = None

        # Cash / burn = runway in months
        if "cash_balance" in df.columns and kpis.get("burn_rate"):
            latest_cash = df["cash_balance"].dropna().iloc[-1]
            kpis["runway_months"] = round(latest_cash / kpis["burn_rate"], 1) if kpis["burn_rate"] > 0 else None
            kpis["cash_balance"] = latest_cash
        else:
            kpis["runway_months"] = None
            kpis["cash_balance"] = None

        # Gross margin
        if "total_revenue" in df.columns and "cogs" in df.columns:
            total_rev = df["total_revenue"].sum()
            total_cogs = df["cogs"].sum()
            kpis["gross_margin_pct"] = round(((total_rev - total_cogs) / total_rev) * 100, 1) if total_rev else None
        else:
            kpis["gross_margin_pct"] = None

        # EBITDA (simple: revenue - opex, no D&A breakdown at this stage)
        if "total_revenue" in df.columns and "total_expenses" in df.columns:
            kpis["ebitda"] = (df["total_revenue"] - df["total_expenses"]).sum()
        else:
            kpis["ebitda"] = None

        # Revenue trend (last 3 months vs prior 3)
        if "total_revenue" in df.columns and len(df) >= 6:
            recent = df["total_revenue"].iloc[-3:].mean()
            prior = df["total_revenue"].iloc[-6:-3].mean()
            kpis["revenue_trend_pct"] = round(((recent - prior) / prior) * 100, 1) if prior else None
        else:
            kpis["revenue_trend_pct"] = None

    except Exception as e:
        kpis["error"] = str(e)

    return kpis