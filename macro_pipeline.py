import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass
from fredapi import Fred

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_logger(name):
    return logging.getLogger(name)

logger = get_logger("MacroEngine")

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

@dataclass
class RegionConfig:
    region_name: str
    currency_pair: str
    currency_role: Literal["domestic", "foreign"]  # 'domestic' = US, 'foreign' = Others
    local_currency: str
    target_hours_utc: List[int]
    indicators: Dict[str, Dict[str, Any]]
    output_filename: str

class DataTransformer:
    """Factory for specific economic data transformations."""
    
    @staticmethod
    def calculate_yoy_index(series: pd.Series, periods: int = 12) -> pd.Series:
        """Converts Index levels to Year-over-Year % Change."""
        return series.pct_change(periods=periods) * 100

    @staticmethod
    def calculate_mom_diff(series: pd.Series) -> pd.Series:
        """Converts Levels to Month-over-Month Difference (Flow)."""
        return series.diff(periods=1)

class MacroDataPipeline:
    def __init__(self, config: RegionConfig):
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            logger.critical("FRED_API_KEY environment variable is missing.")
            raise ValueError("Please set FRED_API_KEY in your environment variables.")
        
        self.fred = Fred(api_key=self.api_key)
        self.config = config
        self.results = []
        self.fred_config = self.config.indicators
        self.last_run_time = 0

    @fetch_with_retry()
    def _fetch_fred_series(self, code: str) -> pd.Series:
        return self.fred.get_series(code)

    def _analyze_local_impact(self, actual: float, previous: float, impact_type: str, scale: float = 1.0) -> tuple[int, str]:
        """Calculates -10 to 10 score and grade for the local currency."""
        diff = actual - previous
        if impact_type == "inverse":
            diff = -diff
            
        local_score = diff * scale
        
        score = int(max(min(round(local_score), 10), -10))
        
        if score == 0: grade = "Neutral"
        elif 1 <= score <= 3: grade = "Mildly Bullish"
        elif 4 <= score <= 7: grade = "Bullish"
        elif 8 <= score <= 10: grade = "Very Bullish"
        elif -3 <= score <= -1: grade = "Mildly Bearish"
        elif -7 <= score <= -4: grade = "Bearish"
        else: grade = "Very Bearish"
        
        return score, grade

    def _process_series(self, name: str, series: pd.Series, transform_type: str, impact_type: str = "direct", scale: float = 1.0):
        if transform_type == "yoy_index":
            transformed = DataTransformer.calculate_yoy_index(series, periods=self.fred_config[name].get("periods", 12))
        elif transform_type == "mom_diff":
            transformed = DataTransformer.calculate_mom_diff(series)
        else:
            transformed = series

        transformed = transformed.dropna()

        if transformed.empty:
            logger.warning(f"{name}: Series empty after transformation.")
            return None

        latest_date = transformed.index[-1]
        latest_val = float(transformed.iloc[-1])
        prev_val = float(transformed.iloc[-2]) if len(transformed) > 1 else 0.0

        is_stale = (datetime.now() - latest_date).days > 45
        status = "Stale" if is_stale else "Fresh"
        if is_stale:
            logger.warning(f"{name} is stale! Latest: {latest_date.date()}")

        naive_ma = round((latest_val + prev_val) / 2, 2)
        score, grade = self._analyze_local_impact(latest_val, prev_val, impact_type, scale)

        result = {
            "Indicator": name,
            "Date": latest_date,
            "Actual": round(latest_val, 2),
            "Previous": round(prev_val, 2),
            "naive_moving_avg": naive_ma,
            "Status": status,
            f"{self.config.local_currency}_Score": score,
            f"{self.config.local_currency}_Grade": grade
        }
        logger.info(f"Processed {name}: {latest_val:.2f} ({latest_date.date()})")
        return result

    def run_fred_extraction(self):
        self.results = []
        logger.info(f"Buffer cleared. Starting new data cycle for {self.config.region_name}.")
        self.last_run_time = time.time()
        new_results = []
        for name, config in self.fred_config.items():
            try:
                raw_data = self._fetch_fred_series(config["code"])
                item = self._process_series(name, raw_data, config["transform"], config.get("impact", "direct"), config.get("scale", 1.0))
                if item:
                    new_results.append(item)
            except Exception as e:
                logger.error(f"Failed to process FRED series {name}: {e}")
        
        self.results = new_results
        return self.results

    def export(self):
        fname_csv = f"{self.config.output_filename}.csv"
        fname_xlsx = f"{self.config.output_filename}.xlsx"
        
        if not self.results: return
        df = pd.DataFrame(self.results)
        
        # Sort by Indicator for consistent output
        df.sort_values(by="Indicator", inplace=True)
        
        # Ensure no duplicates in the export
        df.drop_duplicates(subset=['Indicator'], keep='last', inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df.to_csv(fname_csv, index=False, encoding="utf-8-sig", mode='w')
        try:
            df.to_excel(fname_xlsx, index=False, sheet_name="MacroData")
        except ImportError: pass
        except Exception as e:
            logger.error(f"Failed to save Excel (file open?): {e}")
        logger.info(f"Data saved to {fname_csv} and {fname_xlsx}")

    def _run_cycle(self):
        """Executes a full extraction and export cycle."""
        self.run_fred_extraction()
        self.export()

    def run_continuously(self):
        logger.info(f"Starting scheduler for {self.config.region_name} (Hours UTC: {self.config.target_hours_utc})")
        last_window = None
        while True:
            now = datetime.now(timezone.utc)
            if now.hour in self.config.target_hours_utc and (now.minute == 5 or now.minute == 35):
                current_window = (now.day, now.hour, now.minute)
                if current_window != last_window:
                    # Prevent double-run if started during the window
                    if time.time() - self.last_run_time > 300:
                        self._run_cycle()
                    else:
                        logger.info(f"Skipping scheduled run for {self.config.region_name} (Startup run was recent)")
                    last_window = current_window
                time.sleep(10)
            else: time.sleep(10)
