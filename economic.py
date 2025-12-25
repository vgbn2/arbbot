import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, Any, Optional
from fredapi import Fred

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MacroQuantETL")

# --- UTILITIES ---
def fetch_with_retry(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for exponential backoff retries."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    wait = backoff_factor ** attempt
                    logger.warning(f"Error in {func.__name__}: {e}. Retrying in {wait}s (Attempt {attempt}/{max_retries})")
                    time.sleep(wait)
            logger.error(f"Failed {func.__name__} after {max_retries} attempts.")
            raise
        return wrapper
    return decorator

class DataTransformer:
    """Factory for specific economic data transformations."""
    
    @staticmethod
    def calculate_yoy_index(series: pd.Series) -> pd.Series:
        """Converts Index levels to Year-over-Year % Change."""
        # (Latest / 12-months-ago - 1) * 100
        return series.pct_change(periods=12) * 100

    @staticmethod
    def calculate_mom_diff(series: pd.Series) -> pd.Series:
        """Converts Levels to Month-over-Month Difference (Flow)."""
        return series.diff(periods=1)

class MacroDataPipeline:
    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            logger.critical("FRED_API_KEY environment variable is missing.")
            raise ValueError("Please set FRED_API_KEY in your environment variables.")
        
        self.fred = Fred(api_key=self.api_key)
        self.results = []

        # Configuration: Indicator Name -> {FRED Code, Transformation Method}
        self.fred_config = {
            "GDP Growth QoQ": {"code": "A191RL1Q225SBEA", "transform": "rate", "impact": "direct", "scale": 20.0},
            "CPI YoY": {"code": "CPIAUCSL", "transform": "yoy_index", "impact": "direct", "scale": 30.0},
            "PPI YoY": {"code": "PPIACO", "transform": "yoy_index", "impact": "direct", "scale": 30.0},
            "Core PCE YoY": {"code": "PCEPILFE", "transform": "yoy_index", "impact": "direct", "scale": 30.0},
            "Wage Growth YoY": {"code": "CES0500000003", "transform": "yoy_index", "impact": "direct", "scale": 30.0},
            "Unemployment Rate": {"code": "UNRATE", "transform": "rate", "impact": "inverse", "scale": 20.0},
            "Initial Jobless Claims": {"code": "ICSA", "transform": "rate", "impact": "inverse", "scale": 0.0005},
            "Job Openings (JOLTS)": {"code": "JTSJOL", "transform": "level", "impact": "direct", "scale": 0.05},
            "Nonfarm Payrolls": {"code": "PAYEMS", "transform": "mom_diff", "impact": "direct", "scale": 0.05},
            "Fed Funds Rate": {"code": "FEDFUNDS", "transform": "rate", "impact": "direct", "scale": 40.0},
            "10Y Treasury Yield": {"code": "DGS10", "transform": "rate", "impact": "direct", "scale": 20.0},
            "10Y-2Y Yield Spread": {"code": "T10Y2Y", "transform": "rate", "impact": "direct", "scale": 20.0},
        }

    @fetch_with_retry()
    def _fetch_fred_series(self, code: str) -> pd.Series:
        return self.fred.get_series(code)

    def _analyze_usd_impact(self, actual: float, previous: float, impact_type: str, scale: float = 1.0) -> tuple[int, str]:
        """Calculates -10 to 10 score and grade."""
        diff = actual - previous
        if impact_type == "inverse":
            diff = -diff
            
        raw_score = diff * scale
        score = int(max(min(round(raw_score), 10), -10))
        
        if score == 0: grade = "Neutral"
        elif 1 <= score <= 3: grade = "Mildly Bullish"
        elif 4 <= score <= 7: grade = "Bullish"
        elif 8 <= score <= 10: grade = "Very Bullish"
        elif -3 <= score <= -1: grade = "Mildly Bearish"
        elif -7 <= score <= -4: grade = "Bearish"
        else: grade = "Very Bearish"
        
        return score, grade

    def _process_series(self, name: str, series: pd.Series, transform_type: str, impact_type: str = "direct", scale: float = 1.0):
        # 1. Transform
        if transform_type == "yoy_index":
            transformed = DataTransformer.calculate_yoy_index(series)
        elif transform_type == "mom_diff":
            transformed = DataTransformer.calculate_mom_diff(series)
        else:
            transformed = series

        # Drop NaNs created by shifting
        transformed = transformed.dropna()

        if transformed.empty:
            logger.warning(f"{name}: Series empty after transformation.")
            return

        # 2. Extract Latest & Previous
        latest_date = transformed.index[-1]
        latest_val = float(transformed.iloc[-1])
        prev_val = float(transformed.iloc[-2]) if len(transformed) > 1 else 0.0

        # 3. Stale Check (45 days)
        is_stale = (datetime.now() - latest_date).days > 45
        status = "Stale" if is_stale else "Fresh"
        if is_stale:
            logger.warning(f"{name} is stale! Latest: {latest_date.date()}")

        # 4. Naive Moving Average (formerly Forecast)
        naive_ma = round((latest_val + prev_val) / 2, 2)

        # 5. USD Impact Analysis
        usd_score, usd_grade = self._analyze_usd_impact(latest_val, prev_val, impact_type, scale)

        self.results.append({
            "Indicator": name,
            "Date": latest_date,
            "Actual": round(latest_val, 2),
            "Previous": round(prev_val, 2),
            "naive_moving_avg": naive_ma,
            "Status": status,
            "USD_Score": usd_score,
            "USD_Grade": usd_grade
        })
        logger.info(f"Processed {name}: {latest_val:.2f} ({latest_date.date()})")

    def run_fred_extraction(self):
        for name, config in self.fred_config.items():
            try:
                raw_data = self._fetch_fred_series(config["code"])
                self._process_series(
                    name, 
                    raw_data, 
                    config["transform"], 
                    config.get("impact", "direct"),
                    config.get("scale", 1.0)
                )
            except Exception as e:
                logger.error(f"Failed to process FRED series {name}: {e}")

    def export(self, filename="economic_data.csv"):
        if not self.results:
            logger.warning("No results to export.")
            return

        df = pd.DataFrame(self.results)
        
        # Enforce Types
        df["Date"] = pd.to_datetime(df["Date"])
        df["Actual"] = df["Actual"].astype(float)
        df["Previous"] = df["Previous"].astype(float)
        df["naive_moving_avg"] = df["naive_moving_avg"].astype(float)
        
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info(f"Pipeline finished. Data saved to {filename}")

    def export_to_excel(self, filename="economic_data.xlsx"):
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        
        # Enforce Types & Format for Excel
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df["Actual"] = df["Actual"].astype(float)
        df["Previous"] = df["Previous"].astype(float)
        df["naive_moving_avg"] = df["naive_moving_avg"].astype(float)

        try:
            df.to_excel(filename, index=False, sheet_name="MacroData")
            logger.info(f"Pipeline finished. Data saved to {filename}")
        except ImportError:
            logger.error("Excel export failed. Please install openpyxl: pip install openpyxl")

    def _run_cycle(self):
        """Helper to run the full extraction and export process."""
        logger.info("Executing scheduled update...")
        self.run_fred_extraction()
        self.export()
        self.export_to_excel()

    def run_continuously(self):
        """Runs the pipeline continuously, triggering after major US data releases."""
        logger.info("Starting continuous scheduler...")
        logger.info("Targeting US Data Releases (08:30 AM ET & 10:00 AM ET)")
        logger.info("Checks UTC times: 12:35, 13:35, 14:05, 15:05 to handle DST.")

        while True:
            now = datetime.now(timezone.utc)
            
            # Check for 8:30 AM ET release window (approx 12:30 or 13:30 UTC)
            # We run at minute 35 to give API 5 mins to update
            if now.hour in [12, 13] and now.minute == 35:
                self._run_cycle()
                time.sleep(61) # Sleep > 1 min to avoid double trigger
            
            # Check for 10:00 AM ET release window (approx 14:00 or 15:00 UTC)
            # We run at minute 05
            elif now.hour in [14, 15] and now.minute == 5:
                self._run_cycle()
                time.sleep(61)
            
            else:
                time.sleep(10) # Check every 10 seconds

if __name__ == "__main__":
    try:
        pipeline = MacroDataPipeline()
        pipeline._run_cycle() # Run once immediately
        pipeline.run_continuously() # Enter loop
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        sys.exit(1)
