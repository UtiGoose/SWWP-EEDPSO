"""
Sliding-Window Weighted Popularity (SWWP) model.

Implements dual-scale temporal modeling:
  - Short-term Trend Window: captures recent popularity via exponential decay.
  - Long-term Periodic Window: captures recurring patterns by matching
    temporal context (time segment, weekday).

The model also computes a Purchase Heat indicator H(tau) that quantifies
the absolute activity level of the current temporal context for guiding
EEDPSO initialization.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import timedelta


class SlidingWindowWeightedPopularity:
    """Dual-Scale Sliding-Window Weighted Popularity recommender."""

    def __init__(self, interactions_df, short_window=14, alpha=0.7,
                 trend_decay=0.9, n_candidates=2000):
        """
        Parameters
        ----------
        interactions_df : pd.DataFrame
            Must contain columns ['item_id', 'timestamp'].
        short_window : int
            Length of the short-term trend window in days (Delta_S).
        alpha : float
            Fusion weight for trend vs. periodic scores.
        trend_decay : float
            Exponential decay factor for the trend channel (lambda_fast).
        n_candidates : int
            Number of top items to retain in the candidate pool.
        """
        self.short_window = short_window
        self.alpha = alpha
        self.trend_decay = trend_decay
        self.n_candidates = n_candidates
        self.window_cache = {}

        df = interactions_df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["time_segment"] = df["datetime"].dt.hour // 4
        df["weekday"] = df["datetime"].dt.weekday
        self._raw = df
        self._daily = self._aggregate_daily(df)
        self._global_pop = df["item_id"].value_counts()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_daily(df):
        """Aggregate interactions to (date, time_segment, item_id) level."""
        agg = (df.groupby([pd.Grouper(key="datetime", freq="D"),
                           "time_segment", "item_id"])
               .size().reset_index(name="count"))
        agg["weekday"] = agg["datetime"].dt.weekday
        return agg

    # ------------------------------------------------------------------
    # Precomputation (Algorithm 1 in the paper)
    # ------------------------------------------------------------------

    def precompute(self, n_recent_days=30):
        """Precompute SWWP scores for recent evaluation dates."""
        daily = self._daily
        dates_to_process = sorted(daily["datetime"].dt.date.unique())[-n_recent_days:]
        segments = daily["time_segment"].unique()

        for date in dates_to_process:
            ts_date = pd.Timestamp(date)
            trend_start = ts_date - timedelta(days=self.short_window)

            trend_slice = daily[(daily["datetime"] >= trend_start)
                                & (daily["datetime"] < ts_date)]
            period_slice = daily[daily["datetime"] < trend_start]

            for seg in segments:
                target_wd = ts_date.weekday()

                # --- Trend Score (Eq. 6) ---
                trend_scores = defaultdict(float)
                seg_data = trend_slice[trend_slice["time_segment"] == seg]
                for _, row in seg_data.iterrows():
                    days_ago = (ts_date - row["datetime"]).days
                    trend_scores[int(row["item_id"])] += (
                        row["count"] * self.trend_decay ** days_ago)

                # --- Periodic Score (Eq. 7) ---
                period_scores = defaultdict(float)
                matched = period_slice[
                    (period_slice["weekday"] == target_wd)
                    & (period_slice["time_segment"] == seg)]
                for _, row in matched.iterrows():
                    period_scores[int(row["item_id"])] += row["count"]

                # --- Hybrid Fusion (Eq. 8) ---
                max_t = max(trend_scores.values(), default=1.0)
                max_p = max(period_scores.values(), default=1.0)
                all_items = set(trend_scores) | set(period_scores)

                fused = {}
                for item_id in all_items:
                    s_t = trend_scores.get(item_id, 0.0) / max_t
                    s_p = period_scores.get(item_id, 0.0) / max_p
                    fused[item_id] = self.alpha * s_t + (1 - self.alpha) * s_p

                key = (date, seg, target_wd)
                self.window_cache[key] = fused

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(self, timestamp, n_items=None):
        """Return (item_list, score_list) ranked by SWWP score."""
        n_items = n_items or self.n_candidates
        dt = pd.to_datetime(timestamp)
        key = (dt.date(), dt.hour // 4, dt.weekday())

        if key in self.window_cache:
            scores = self.window_cache[key]
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = [(iid, cnt) for iid, cnt
                      in zip(self._global_pop.index, self._global_pop.values)]

        items = [i for i, _ in ranked[:n_items]]
        counts = [c for _, c in ranked[:n_items]]

        # Backfill with global popularity if needed
        if len(items) < n_items:
            existing = set(items)
            for iid in self._global_pop.index:
                if iid not in existing:
                    items.append(iid)
                    counts.append(float(self._global_pop[iid]))
                    if len(items) >= n_items:
                        break

        return items[:n_items], counts[:n_items]

    # ------------------------------------------------------------------
    # Purchase Heat Estimation (Eq. 10-13)
    # ------------------------------------------------------------------

    @staticmethod
    def purchase_heat(dt):
        """Compute the Purchase Heat indicator H(tau)."""
        hour, weekday, month = dt.hour, dt.weekday(), dt.month
        segment = hour // 4

        # Base heat B_s (Eq. 11)
        base = {0: 0.5, 1: 0.6, 2: 0.6, 3: 0.7, 4: 0.7, 5: 0.8}[segment]

        # Weekday factor F_d (Eq. 12)
        if weekday in (5, 6):
            wd_factor = 1.2
        elif weekday == 4:
            wd_factor = 1.1
        else:
            wd_factor = 1.0

        # Monthly factor F_m (Eq. 13)
        if month in (11, 12):
            mo_factor = 1.15
        elif month in (6, 7):
            mo_factor = 1.1
        else:
            mo_factor = 1.0

        return min(base * wd_factor * mo_factor, 0.9)
