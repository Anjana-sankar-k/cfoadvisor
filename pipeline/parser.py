import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = ["month", "total_revenue", "total_expenses"]

def parse_file(uploaded_file) -> pd.DataFrame | None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None

        df.columns = [c.strip().lower() for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.info("Your file must include at least: month, total_revenue, total_expenses")
            return None

        # Parse month column
        df["month"] = pd.to_datetime(df["month"], infer_datetime_format=True, errors="coerce")
        df = df.sort_values("month").reset_index(drop=True)

        # Fill numeric NaNs with 0
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return None


def df_to_text_chunks(df: pd.DataFrame) -> list[str]:
    chunks = []
    for _, row in df.iterrows():
        month_str = row["month"].strftime("%B %Y") if pd.notnull(row["month"]) else "Unknown month"
        parts = [f"In {month_str}:"]

        if "total_revenue" in row:
            parts.append(f"total revenue was {row['total_revenue']:,.0f}")
        if "total_expenses" in row:
            parts.append(f"total expenses were {row['total_expenses']:,.0f}")
        if "cogs" in row:
            parts.append(f"cost of goods sold was {row['cogs']:,.0f}")
        if "net_cash_flow" in row:
            parts.append(f"net cash flow was {row['net_cash_flow']:,.0f}")
        if "cash_balance" in row:
            parts.append(f"cash balance was {row['cash_balance']:,.0f}")
        if "salaries" in row:
            parts.append(f"salaries were {row['salaries']:,.0f}")
        if "rent" in row:
            parts.append(f"rent was {row['rent']:,.0f}")
        if "marketing" in row:
            parts.append(f"marketing spend was {row['marketing']:,.0f}")

        chunks.append(" ".join(parts) + ".")

    # Also add a summary chunk
    if "total_revenue" in df.columns:
        total_rev = df["total_revenue"].sum()
        total_exp = df["total_expenses"].sum()
        summary = (
            f"Financial summary across {len(df)} months: "
            f"total revenue was {total_rev:,.0f}, "
            f"total expenses were {total_exp:,.0f}, "
            f"net position was {total_rev - total_exp:,.0f}."
        )
        chunks.append(summary)

    return chunks