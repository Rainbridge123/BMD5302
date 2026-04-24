import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


QUESTIONNAIRE = [
    {
        "question_id": "q1",
        "construct": "preference",
        "weight": 0.10,
        "title": "What is your planned investment horizon?",
        "options": {
            "A": {"text": "Less than 1 year", "score": 1},
            "B": {"text": "1-3 years", "score": 2},
            "C": {"text": "3-5 years", "score": 3},
            "D": {"text": "More than 5 years", "score": 4},
        },
    },
    {
        "question_id": "q2",
        "construct": "preference",
        "weight": 0.08,
        "title": "How would you describe your investment experience?",
        "options": {
            "A": {"text": "No experience beyond bank deposits", "score": 1},
            "B": {"text": "Experience with low-risk products such as bonds or money market funds", "score": 2},
            "C": {"text": "Experience with mutual funds or balanced products", "score": 3},
            "D": {"text": "Experience with stocks, derivatives, or other high-risk assets", "score": 4},
        },
    },
    {
        "question_id": "q3",
        "construct": "capacity",
        "weight": 0.10,
        "title": "What is your current financial situation?",
        "options": {
            "A": {"text": "Unstable income and little savings", "score": 1},
            "B": {"text": "Stable income but limited surplus", "score": 2},
            "C": {"text": "Stable income with moderate savings", "score": 3},
            "D": {"text": "Strong financial position with significant investable assets", "score": 4},
        },
    },
    {
        "question_id": "q4",
        "construct": "capacity",
        "weight": 0.10,
        "title": "What proportion of your total assets will be invested?",
        "options": {
            "A": {"text": "More than 80%", "score": 1},
            "B": {"text": "50%-80%", "score": 2},
            "C": {"text": "20%-50%", "score": 3},
            "D": {"text": "Less than 20%", "score": 4},
        },
    },
    {
        "question_id": "q5",
        "construct": "preference",
        "weight": 0.15,
        "title": "If your portfolio drops by 15% in a short period, what would you do?",
        "options": {
            "A": {"text": "Sell all investments", "score": 1},
            "B": {"text": "Sell part of the portfolio", "score": 2},
            "C": {"text": "Hold and wait", "score": 3},
            "D": {"text": "Buy more", "score": 4},
        },
    },
    {
        "question_id": "q6",
        "construct": "preference",
        "weight": 0.12,
        "title": "Which return-risk trade-off do you prefer?",
        "options": {
            "A": {"text": "Stable return with no loss", "score": 1},
            "B": {"text": "Moderate return with small fluctuations", "score": 2},
            "C": {"text": "Higher return with acceptable losses", "score": 3},
            "D": {"text": "Very high return despite large potential losses", "score": 4},
        },
    },
    {
        "question_id": "q7",
        "construct": "preference",
        "weight": 0.15,
        "title": "What is the maximum loss you can tolerate?",
        "options": {
            "A": {"text": "Less than 10%", "score": 1},
            "B": {"text": "10%-20%", "score": 2},
            "C": {"text": "20%-30%", "score": 3},
            "D": {"text": "More than 30%", "score": 4},
        },
    },
    {
        "question_id": "q8",
        "construct": "preference",
        "weight": 0.08,
        "title": "What is your primary investment goal?",
        "options": {
            "A": {"text": "Capital preservation", "score": 1},
            "B": {"text": "Stable income", "score": 2},
            "C": {"text": "Balanced growth", "score": 3},
            "D": {"text": "Aggressive capital appreciation", "score": 4},
        },
    },
    {
        "question_id": "q9",
        "construct": "capacity",
        "weight": 0.05,
        "title": "How stable is your income source?",
        "options": {
            "A": {"text": "Very unstable", "score": 1},
            "B": {"text": "Somewhat unstable", "score": 2},
            "C": {"text": "Stable", "score": 3},
            "D": {"text": "Very stable", "score": 4},
        },
    },
    {
        "question_id": "q10",
        "construct": "preference",
        "weight": 0.07,
        "title": "How do you react to market volatility?",
        "options": {
            "A": {"text": "I feel very uncomfortable and avoid risk", "score": 1},
            "B": {"text": "I prefer to reduce exposure", "score": 2},
            "C": {"text": "I can tolerate fluctuations", "score": 3},
            "D": {"text": "I see volatility as an opportunity", "score": 4},
        },
    },
]


POLICIES = {
    "no_short": {
        "title": "No Short Sales",
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "gross_limit": 1.0,
    },
    "unconstrained_short": {
        "title": "Unconstrained Short Sales",
        "lower_bound": None,
        "upper_bound": None,
        "gross_limit": None,
    },
    "bounded_short": {
        "title": "Bounded Short Sales",
        "lower_bound": -0.20,
        "upper_bound": 0.35,
        "gross_limit": 1.40,
    },
}


BOOTSTRAP_ITERATIONS = 250
ROLLING_WINDOW = 36
CALIBRATION_GRID = np.round(np.arange(1.00, 15.00 + 0.001, 0.01), 2)
RISK_AVERSION_BANDS = [
    {"category": "Very Conservative", "min_score": 1.00, "max_score": 1.75, "A_low": 8.0, "A_high": 12.0},
    {"category": "Conservative", "min_score": 1.75, "max_score": 2.50, "A_low": 5.0, "A_high": 8.0},
    {"category": "Balanced", "min_score": 2.50, "max_score": 3.25, "A_low": 3.0, "A_high": 5.0},
    {"category": "Growth", "min_score": 3.25, "max_score": 3.625, "A_low": 1.5, "A_high": 3.0},
    {"category": "Aggressive", "min_score": 3.625, "max_score": 4.00, "A_low": 0.5, "A_high": 1.5},
]
VOLATILITY_BAND_SPECS = [
    ("3%-13%", 0.03, 0.13),
    ("4%-14%", 0.04, 0.14),
    ("5%-15%", 0.05, 0.15),
]
BOUNDED_SHORT_REGIMES = [
    {"regime": "conservative", "lower_bound": -0.15, "upper_bound": 0.30, "gross_limit": 1.20},
    {"regime": "baseline", "lower_bound": -0.20, "upper_bound": 0.35, "gross_limit": 1.40},
    {"regime": "liberal", "lower_bound": -0.25, "upper_bound": 0.40, "gross_limit": 1.60},
]
SLEEVE_SHORT_LABELS = {
    "money market/cash": "Cash",
    "short-duration bond": "Short Bond",
    "global investment-grade bond": "IG Bond",
    "higher-yield bond": "HY Bond",
    "singapore equity": "SG Equity",
    "u.s. equity": "US Equity",
    "global developed equity": "World Equity",
    "asia ex-japan equity": "Asia ex-JP",
    "global REIT": "REIT",
    "gold/real-asset hedge": "Gold",
}
FRONTIER_LABEL_OFFSETS = {
    "Cash": (8, -12),
    "Short Bond": (8, 4),
    "IG Bond": (8, -10),
    "HY Bond": (8, 6),
    "SG Equity": (-50, 8),
    "US Equity": (8, -14),
    "World Equity": (-62, -10),
    "Asia ex-JP": (-54, 8),
    "REIT": (8, -16),
    "Gold": (8, 8),
}
PORTFOLIO_PLOT_LABELS = {
    "Equal Weight Benchmark": "Equal Weight",
    "GMVP no_short": "GMVP\nNo Short",
    "GMVP bounded_short": "GMVP\nBounded",
    "Optimal Portfolio no_short": "Optimal\nNo Short",
    "Resampled Portfolio no_short": "Resampled\nNo Short",
    "Optimal Portfolio bounded_short": "Optimal\nBounded",
}
ASSET_PLOT_COLORS = {
    "Cash": "#2a9d8f",
    "Short Bond": "#4cc9f0",
    "IG Bond": "#4895ef",
    "HY Bond": "#4361ee",
    "SG Equity": "#f4a261",
    "US Equity": "#e76f51",
    "World Equity": "#ef476f",
    "Asia ex-JP": "#ff7f50",
    "REIT": "#bc4749",
    "Gold": "#ffb703",
}
PORTFOLIO_BAR_COLORS = {
    "Equal Weight Benchmark": "#f4a261",
    "GMVP no_short": "#2a9d8f",
    "GMVP bounded_short": "#4cc9f0",
    "Optimal Portfolio no_short": "#1d3557",
    "Resampled Portfolio no_short": "#457b9d",
    "Optimal Portfolio bounded_short": "#e76f51",
}
PLOT_THEME = {
    "paper": "#fff9f1",
    "panel": "#fdf2e1",
    "grid": "#d9c7a5",
    "navy": "#1d3557",
    "teal": "#2a9d8f",
    "orange": "#e76f51",
    "gold": "#ffb703",
    "berry": "#ef476f",
    "ink": "#25324a",
    "soft_white": "#fffdf8",
    "short_red": "#d1495b",
    "short_edge": "#8d1b2a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BMD5302 Part 1 and Part 2 portfolio engine.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--prices", default=str(root / "data" / "fsmone_prices.csv"))
    parser.add_argument("--fund-meta", default=str(root / "data" / "fsmone_fund_universe.csv"))
    parser.add_argument("--answers", default=str(root / "data" / "questionnaire_answers.json"))
    parser.add_argument("--outputs", default=str(root / "outputs"))
    parser.add_argument("--submission-mode", choices=["draft", "final"], default="draft")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_questionnaire_weights() -> None:
    total = sum(question["weight"] for question in QUESTIONNAIRE)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Questionnaire weights must sum to 1.0; got {total}.")
    constructs = {question["construct"] for question in QUESTIONNAIRE}
    if constructs != {"preference", "capacity"}:
        raise ValueError(f"Questionnaire constructs must be preference/capacity; got {sorted(constructs)}.")


def load_fund_metadata(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "selection_rationale" in df.columns and "selection_reason" not in df.columns:
        df = df.rename(columns={"selection_rationale": "selection_reason"})

    required = {
        "fund_name",
        "sleeve",
        "share_class_currency",
        "selection_status",
        "fund_house",
        "fsmone_code",
        "isin",
        "share_class_type",
        "inception_date",
        "aum_sgd_m",
        "annual_fee_pct",
        "selection_reason",
        "selection_evidence",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Fund metadata is missing columns: {sorted(missing)}")

    optional_defaults = {
        "selection_rule": "",
        "rejected_alternatives": "",
        "tie_break_reason": "",
        "evidence_reference": "",
        "proxy_source": "",
        "notes": "",
    }
    for column, default in optional_defaults.items():
        if column not in df.columns:
            df[column] = default

    if len(df) != 10:
        raise ValueError("Fund metadata must contain exactly 10 funds.")
    if df["fund_name"].duplicated().any():
        raise ValueError("Fund names must be unique.")
    if df["sleeve"].duplicated().any():
        raise ValueError("Each sleeve must appear exactly once.")

    df["selection_status"] = df["selection_status"].astype(str).str.lower().str.strip()
    if not set(df["selection_status"]).issubset({"final", "proxy"}):
        raise ValueError("selection_status must be either 'final' or 'proxy'.")

    df["share_class_currency"] = df["share_class_currency"].astype(str).str.upper().str.replace("_", "-").str.strip()
    allowed_currencies = {"SGD", "SGD-HEDGED"}
    if not set(df["share_class_currency"]).issubset(allowed_currencies):
        raise ValueError("All share class currencies must be SGD or SGD-HEDGED.")

    df["inception_date"] = pd.to_datetime(df["inception_date"], errors="coerce")
    if df["inception_date"].isnull().any():
        raise ValueError("inception_date must be a valid date for every fund.")

    for numeric_column in ["aum_sgd_m", "annual_fee_pct"]:
        df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")
        if df[numeric_column].isnull().any():
            raise ValueError(f"{numeric_column} must be numeric for every fund.")
        if (df[numeric_column] < 0).any():
            raise ValueError(f"{numeric_column} must be non-negative.")

    text_columns = [
        "fund_name",
        "sleeve",
        "fund_house",
        "fsmone_code",
        "isin",
        "share_class_type",
        "selection_reason",
        "selection_evidence",
        "selection_rule",
        "rejected_alternatives",
        "tie_break_reason",
        "evidence_reference",
        "proxy_source",
        "notes",
    ]
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    return df


def load_price_data(filepath: str, expected_funds: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(filepath)
    if "date" not in raw.columns:
        raise ValueError("Price file must contain a date column.")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["date"].isnull().any():
        raise ValueError("Price file contains invalid dates.")
    raw = raw.sort_values("date").set_index("date")
    if raw.index.duplicated().any():
        raise ValueError("Price file contains duplicated dates.")
    if not raw.index.is_monotonic_increasing:
        raise ValueError("Dates must be sorted ascending.")

    missing = [name for name in expected_funds if name not in raw.columns]
    extra = [name for name in raw.columns if name not in expected_funds]
    if missing or extra:
        raise ValueError(f"Price columns must match fund metadata exactly. Missing={missing}; Extra={extra}")

    panel = raw.loc[:, expected_funds].apply(pd.to_numeric, errors="coerce")
    for column in panel.columns:
        valid = panel[column].dropna()
        if (valid <= 0).any():
            raise ValueError(f"All prices must be strictly positive for {column}.")

    aligned = panel.dropna(how="any")
    if aligned.empty:
        raise ValueError("Aligned price panel is empty after dropping incomplete months.")
    if len(aligned) < 60:
        raise ValueError("At least 60 monthly observations are required after alignment.")

    common_start = aligned.index.min().date().isoformat()
    common_end = aligned.index.max().date().isoformat()
    coverage_rows = []
    for fund_name in expected_funds:
        series = panel[fund_name]
        valid = series.dropna()
        coverage_rows.append(
            {
                "fund_name": fund_name,
                "first_available_date": "" if valid.empty else valid.index.min().date().isoformat(),
                "last_available_date": "" if valid.empty else valid.index.max().date().isoformat(),
                "original_observations": int(valid.shape[0]),
                "missing_count": int(series.isnull().sum()),
                "usable_after_alignment": int(aligned.shape[0]),
                "common_window_start": common_start,
                "common_window_end": common_end,
            }
        )
    return aligned, pd.DataFrame(coverage_rows)


def submission_blockers(metadata: pd.DataFrame, submission_mode: str) -> List[str]:
    blockers: List[str] = []
    proxy_count = int((metadata["selection_status"] != "final").sum())
    if proxy_count:
        blockers.append(f"{proxy_count} fund entries are still marked as proxy instead of final FSMOne funds.")

    critical_text_columns = [
        "fsmone_code",
        "selection_reason",
        "selection_evidence",
        "rejected_alternatives",
        "tie_break_reason",
        "evidence_reference",
    ]
    for column in critical_text_columns:
        if (metadata[column].astype(str).str.strip() == "").any():
            blockers.append(f"At least one fund is missing {column}.")

    zero_fee_count = int((metadata["annual_fee_pct"] <= 0).sum())
    if zero_fee_count:
        blockers.append(f"{zero_fee_count} funds still have zero or missing annual fees.")

    invalid_currency = ~metadata["share_class_currency"].isin({"SGD", "SGD-HEDGED"})
    if invalid_currency.any():
        blockers.append("At least one fund uses a share-class currency outside SGD / SGD-HEDGED.")

    if submission_mode == "final" and blockers:
        raise ValueError("Final submission mode failed readiness checks: " + "; ".join(blockers))
    return blockers


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    returns_df = price_df.pct_change().dropna()
    if returns_df.empty:
        raise ValueError("Not enough observations to compute monthly returns.")
    return returns_df


def annualized_mean_returns(returns_df: pd.DataFrame, periods_per_year: int = 12) -> pd.Series:
    return returns_df.mean() * periods_per_year


def annualized_covariance(returns_df: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    return returns_df.cov() * periods_per_year


def covariance_psd_diagnostics(cov_matrix: pd.DataFrame, floor: float = 1e-8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = cov_matrix.to_numpy(dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    raw_min = float(eigenvalues.min())
    raw_max = float(eigenvalues.max())
    raw_psd_pass = bool(raw_min >= -1e-10)
    repair_applied = not raw_psd_pass
    if repair_applied:
        repaired_eigenvalues = np.maximum(eigenvalues, floor)
        repaired_values = (eigenvectors * repaired_eigenvalues) @ eigenvectors.T
        repaired_values = (repaired_values + repaired_values.T) / 2.0
    else:
        repaired_values = values.copy()
    repaired_cov = pd.DataFrame(repaired_values, index=cov_matrix.index, columns=cov_matrix.columns)

    psd_df = pd.DataFrame(
        [
            {
                "raw_min_eigenvalue": raw_min,
                "raw_max_eigenvalue": raw_max,
                "raw_psd_pass": raw_psd_pass,
                "repair_applied": repair_applied,
                "eigenvalue_floor": floor,
                "repaired_min_eigenvalue": float(np.linalg.eigvalsh(repaired_values).min()),
            }
        ]
    )
    note = (
        "Raw covariance matrix was already positive semidefinite."
        if not repair_applied
        else "Raw covariance matrix required numerical repair; eigenvalues below the floor were clipped before optimization."
    )
    repair_note_df = pd.DataFrame([{"note": note}])
    return repaired_cov, psd_df, repair_note_df


def asset_points(mean_returns: pd.Series, cov_matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    sleeve_lookup = metadata.set_index("fund_name")["sleeve"].to_dict()
    return pd.DataFrame(
        {
            "fund_name": mean_returns.index,
            "short_label": [SLEEVE_SHORT_LABELS[sleeve_lookup[name]] for name in mean_returns.index],
            "expected_return": mean_returns.values,
            "volatility": np.sqrt(np.clip(np.diag(cov_matrix.values), 0.0, None)),
        }
    )


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    return float(np.dot(weights, mean_returns))


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(weights.T @ cov_matrix @ weights)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.sqrt(max(portfolio_variance(weights, cov_matrix), 0.0)))


def gross_exposure(weights: np.ndarray) -> float:
    return float(np.sum(np.abs(weights)))


def long_exposure(weights: np.ndarray) -> float:
    return float(np.sum(np.clip(weights, 0.0, None)))


def short_exposure(weights: np.ndarray) -> float:
    return float(np.sum(np.abs(np.clip(weights, None, 0.0))))


def utility_from_stats(expected_return: float, variance: float, A: float) -> float:
    return float(expected_return - 0.5 * A * variance)


def utility_value(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, A: float) -> float:
    return utility_from_stats(portfolio_return(weights, mean_returns), portfolio_variance(weights, cov_matrix), A)


def weighted_long_fee_pct(weights: np.ndarray, annual_fee_pct: np.ndarray) -> float:
    long_weights = np.clip(weights, 0.0, None)
    if np.isclose(long_weights.sum(), 0.0):
        return 0.0
    normalized_long_weights = long_weights / long_weights.sum()
    return float(np.dot(normalized_long_weights, annual_fee_pct))


def borrow_cost_drag(weights: np.ndarray, annual_borrow_cost: float) -> float:
    return float(short_exposure(weights) * annual_borrow_cost)


def realized_portfolio_returns(weights: np.ndarray, returns_df: pd.DataFrame) -> pd.Series:
    return returns_df @ weights


def max_drawdown(weights: np.ndarray, returns_df: pd.DataFrame) -> float:
    realized = realized_portfolio_returns(weights, returns_df)
    cumulative = (1.0 + realized).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    return float(drawdown.min())


def _weight_sum_constraint() -> Dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _target_return_constraint(mean_returns: np.ndarray, target_return: float) -> Dict:
    return {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return}


def _gross_exposure_constraint(limit: float) -> Dict:
    return {"type": "ineq", "fun": lambda w: limit - np.sum(np.abs(w))}


def _bounds(policy_name: str, n_assets: int) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
    policy = POLICIES[policy_name]
    if policy["lower_bound"] is None and policy["upper_bound"] is None:
        return None
    return [(policy["lower_bound"], policy["upper_bound"])] * n_assets


def _constraints(policy_name: str, mean_returns: Optional[np.ndarray] = None, target_return: Optional[float] = None) -> List[Dict]:
    policy = POLICIES[policy_name]
    constraints = [_weight_sum_constraint()]
    if target_return is not None and mean_returns is not None:
        constraints.append(_target_return_constraint(mean_returns, target_return))
    if policy["gross_limit"] is not None and policy["gross_limit"] > 1.0:
        constraints.append(_gross_exposure_constraint(policy["gross_limit"]))
    return constraints


def _initial_weights(n_assets: int) -> np.ndarray:
    return np.repeat(1.0 / n_assets, n_assets)


def _solve(
    objective,
    x0: np.ndarray,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]],
    constraints: List[Dict],
    label: str,
) -> np.ndarray:
    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1500, "ftol": 1e-8},
    )
    if not result.success:
        raise ValueError(f"{label} optimization failed: {result.message}")
    return result.x


def serialize_top_positions(weights: np.ndarray, fund_names: List[str], positive: bool) -> str:
    items = []
    for fund_name, weight in zip(fund_names, weights):
        if positive and weight > 1e-8:
            items.append((fund_name, weight))
        if not positive and weight < -1e-8:
            items.append((fund_name, weight))
    items.sort(key=lambda item: item[1], reverse=positive)
    top_items = items[:3]
    if not top_items:
        return "N/A"
    return "; ".join(f"{name} ({weight:.2%})" for name, weight in top_items)


def portfolio_row(
    portfolio_name: str,
    weights: np.ndarray,
    fund_names: List[str],
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    annual_fee_pct: Optional[np.ndarray] = None,
    returns_df: Optional[pd.DataFrame] = None,
    A: Optional[float] = None,
    target_return: Optional[float] = None,
) -> Dict:
    expected_return = portfolio_return(weights, mean_returns)
    variance = portfolio_variance(weights, cov_matrix)
    row = {
        "portfolio_name": portfolio_name,
        "expected_return": expected_return,
        "variance": variance,
        "volatility": portfolio_volatility(weights, cov_matrix),
        "net_exposure": float(np.sum(weights)),
        "long_exposure": long_exposure(weights),
        "gross_exposure": gross_exposure(weights),
        "short_exposure": short_exposure(weights),
        "target_return": np.nan if target_return is None else float(target_return),
        "top_long_positions": serialize_top_positions(weights, fund_names, positive=True),
        "top_short_positions": serialize_top_positions(weights, fund_names, positive=False),
        "client_ready": bool(np.all(weights >= -1e-8)),
    }
    if annual_fee_pct is None:
        row["weighted_long_fee_pct"] = np.nan
    else:
        row["weighted_long_fee_pct"] = weighted_long_fee_pct(weights, annual_fee_pct)
    row["utility"] = np.nan if A is None else utility_from_stats(expected_return, variance, A)
    row["max_drawdown"] = np.nan if returns_df is None else max_drawdown(weights, returns_df)
    for fund_name, weight in zip(fund_names, weights):
        row[fund_name] = float(weight)
    return row


def extract_weights(row: Dict, fund_names: List[str]) -> np.ndarray:
    return np.array([float(row[fund_name]) for fund_name in fund_names], dtype=float)


def solve_gmvp(
    policy_name: str,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    A: Optional[float] = None,
) -> Dict:
    mu = mean_returns.to_numpy(dtype=float)
    sigma = cov_matrix.to_numpy(dtype=float)
    weights = _solve(
        objective=lambda w: portfolio_variance(w, sigma),
        x0=_initial_weights(len(mu)),
        bounds=_bounds(policy_name, len(mu)),
        constraints=_constraints(policy_name),
        label=f"GMVP {policy_name}",
    )
    return portfolio_row(
        portfolio_name=f"GMVP {policy_name}",
        weights=weights,
        fund_names=list(mean_returns.index),
        mean_returns=mu,
        cov_matrix=sigma,
        annual_fee_pct=annual_fee_pct.to_numpy(dtype=float),
        returns_df=returns_df,
        A=A,
    )


def solve_return_extreme(policy_name: str, mean_returns: pd.Series, maximize: bool = True) -> Tuple[np.ndarray, float]:
    mu = mean_returns.to_numpy(dtype=float)

    def objective(w):
        value = portfolio_return(w, mu)
        return -value if maximize else value

    weights = _solve(
        objective=objective,
        x0=_initial_weights(len(mu)),
        bounds=_bounds(policy_name, len(mu)),
        constraints=_constraints(policy_name),
        label=f"Return extreme {policy_name}",
    )
    return weights, portfolio_return(weights, mu)


def solve_target_return_portfolio(
    policy_name: str,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    target_return: float,
) -> Dict:
    mu = mean_returns.to_numpy(dtype=float)
    sigma = cov_matrix.to_numpy(dtype=float)
    weights = _solve(
        objective=lambda w: portfolio_variance(w, sigma),
        x0=_initial_weights(len(mu)),
        bounds=_bounds(policy_name, len(mu)),
        constraints=_constraints(policy_name, mean_returns=mu, target_return=target_return),
        label=f"Target return {policy_name}",
    )
    return portfolio_row(
        portfolio_name=f"Target Return {policy_name}",
        weights=weights,
        fund_names=list(mean_returns.index),
        mean_returns=mu,
        cov_matrix=sigma,
        annual_fee_pct=annual_fee_pct.to_numpy(dtype=float),
        returns_df=returns_df,
        target_return=target_return,
    )


def generate_frontier(
    policy_name: str,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    n_points: int = 80,
    upper_target_return: Optional[float] = None,
) -> pd.DataFrame:
    gmvp = solve_gmvp(policy_name, mean_returns, cov_matrix, annual_fee_pct, returns_df)
    start_return = gmvp["expected_return"]
    if upper_target_return is None:
        if policy_name == "unconstrained_short":
            raise ValueError("An explicit upper_target_return is required for the unconstrained-short frontier plot segment.")
        _, max_return = solve_return_extreme(policy_name, mean_returns, maximize=True)
        upper_target_return = max(max_return, start_return)
    targets = np.linspace(start_return, max(float(upper_target_return), start_return), n_points)
    rows = []
    for target in targets:
        try:
            row = solve_target_return_portfolio(policy_name, mean_returns, cov_matrix, annual_fee_pct, returns_df, float(target))
            row["policy"] = policy_name
            rows.append(row)
        except ValueError:
            continue
    frontier = pd.DataFrame(rows)
    if frontier.empty:
        raise ValueError(f"No efficient frontier points were generated for {policy_name}.")
    return frontier.sort_values(["volatility", "expected_return"]).reset_index(drop=True)


def solve_optimal_portfolio(
    policy_name: str,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    A: float,
    initial_weights: Optional[np.ndarray] = None,
) -> Dict:
    mu = mean_returns.to_numpy(dtype=float)
    sigma = cov_matrix.to_numpy(dtype=float)

    def objective(w):
        return -(portfolio_return(w, mu) - 0.5 * A * portfolio_variance(w, sigma))

    weights = _solve(
        objective=objective,
        x0=_initial_weights(len(mu)) if initial_weights is None else initial_weights,
        bounds=_bounds(policy_name, len(mu)),
        constraints=_constraints(policy_name),
        label=f"Optimal portfolio {policy_name}",
    )
    return portfolio_row(
        portfolio_name=f"Optimal Portfolio {policy_name}",
        weights=weights,
        fund_names=list(mean_returns.index),
        mean_returns=mu,
        cov_matrix=sigma,
        annual_fee_pct=annual_fee_pct.to_numpy(dtype=float),
        returns_df=returns_df,
        A=A,
    )


def equal_weight_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    A: float,
) -> Dict:
    fund_names = list(mean_returns.index)
    weights = _initial_weights(len(fund_names))
    return portfolio_row(
        portfolio_name="Equal Weight Benchmark",
        weights=weights,
        fund_names=fund_names,
        mean_returns=mean_returns.to_numpy(dtype=float),
        cov_matrix=cov_matrix.to_numpy(dtype=float),
        annual_fee_pct=annual_fee_pct.to_numpy(dtype=float),
        returns_df=returns_df,
        A=A,
    )


def load_answers(filepath: str) -> Dict[str, str]:
    with open(filepath, "r", encoding="utf-8") as handle:
        answers = json.load(handle)
    if not isinstance(answers, dict):
        raise TypeError("Questionnaire answers must be a JSON object.")
    return answers


def construct_weight_totals() -> Dict[str, float]:
    totals: Dict[str, float] = {"preference": 0.0, "capacity": 0.0}
    for question in QUESTIONNAIRE:
        totals[question["construct"]] += question["weight"]
    return totals


def target_volatility_from_score(score: float, lower_bound: float = 0.04, upper_bound: float = 0.14) -> float:
    return lower_bound + ((score - 1.0) / 3.0) * (upper_bound - lower_bound)


def risk_aversion_band_from_preference(preference_score: float) -> Dict[str, float | str]:
    bounded_score = float(np.clip(preference_score, 1.0, 4.0))
    for band in RISK_AVERSION_BANDS:
        if band["min_score"] <= bounded_score < band["max_score"]:
            return band
    return RISK_AVERSION_BANDS[-1]


def risk_aversion_from_preference_capacity(preference_score: float, capacity_score: float) -> Tuple[float, Dict[str, float | str]]:
    band = risk_aversion_band_from_preference(preference_score)
    bounded_capacity = float(np.clip(capacity_score, 1.0, 4.0))
    A_low = float(band["A_low"])
    A_high = float(band["A_high"])
    A = A_high - ((bounded_capacity - 1.0) / 3.0) * (A_high - A_low)
    return float(A), band


def investor_type_from_target_vol(target_volatility: float) -> str:
    if target_volatility < 0.07:
        return "Conservative"
    if target_volatility < 0.11:
        return "Balanced"
    return "Growth"


def score_questionnaire(
    answers: Dict[str, str],
    lower_vol_bound: float = 0.04,
    upper_vol_bound: float = 0.14,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    totals = construct_weight_totals()
    rows = []
    construct_scores = {"preference": 0.0, "capacity": 0.0}
    total_raw_score = 0.0

    for question in QUESTIONNAIRE:
        qid = question["question_id"]
        if qid not in answers:
            raise ValueError(f"Missing answer for {qid}.")
        selected_option = str(answers[qid]).strip().upper()
        if selected_option not in question["options"]:
            raise ValueError(f"Invalid option {selected_option} for {qid}.")
        option = question["options"][selected_option]
        normalized_construct_weight = question["weight"] / totals[question["construct"]]
        construct_contribution = option["score"] * normalized_construct_weight
        construct_scores[question["construct"]] += construct_contribution
        total_raw_score += option["score"]
        rows.append(
            {
                "question_id": qid,
                "question": question["title"],
                "construct": question["construct"],
                "selected_option": selected_option,
                "selected_text": option["text"],
                "raw_score": option["score"],
                "question_weight": question["weight"],
                "normalized_construct_weight": normalized_construct_weight,
                "construct_contribution": construct_contribution,
            }
        )

    preference_score = construct_scores["preference"]
    capacity_score = construct_scores["capacity"]
    target_vol_pref = target_volatility_from_score(preference_score, lower_vol_bound, upper_vol_bound)
    target_vol_cap = target_volatility_from_score(capacity_score, lower_vol_bound, upper_vol_bound)
    final_target_vol = min(target_vol_pref, target_vol_cap)
    binding_construct = "capacity" if target_vol_cap <= target_vol_pref else "preference"
    risk_aversion_A, risk_band = risk_aversion_from_preference_capacity(preference_score, capacity_score)

    construct_summary_df = pd.DataFrame(
        [
            {
                "construct": "preference",
                "question_count": int(sum(question["construct"] == "preference" for question in QUESTIONNAIRE)),
                "raw_weight_sum": totals["preference"],
                "normalized_score": preference_score,
                "target_volatility": target_vol_pref,
                "volatility_band": f"{lower_vol_bound:.0%}-{upper_vol_bound:.0%}",
                "interpretation": "Subjective risk appetite and tolerance for market losses.",
            },
            {
                "construct": "capacity",
                "question_count": int(sum(question["construct"] == "capacity" for question in QUESTIONNAIRE)),
                "raw_weight_sum": totals["capacity"],
                "normalized_score": capacity_score,
                "target_volatility": target_vol_cap,
                "volatility_band": f"{lower_vol_bound:.0%}-{upper_vol_bound:.0%}",
                "interpretation": "Objective capacity to bear risk given income, assets, and financial resilience.",
            },
        ]
    )

    profile = {
        "total_raw_score": total_raw_score,
        "risk_preference_score": preference_score,
        "risk_capacity_score": capacity_score,
        "target_vol_pref": target_vol_pref,
        "target_vol_cap": target_vol_cap,
        "final_target_vol": final_target_vol,
        "final_risk_score": min(preference_score, capacity_score),
        "risk_category": risk_band["category"],
        "binding_construct": binding_construct,
        "risk_aversion_A": risk_aversion_A,
        "risk_aversion_lower_A": risk_band["A_low"],
        "risk_aversion_upper_A": risk_band["A_high"],
        "volatility_band_lower": lower_vol_bound,
        "volatility_band_upper": upper_vol_bound,
    }
    return pd.DataFrame(rows), construct_summary_df, profile


def build_a_calibration_grid(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    fund_names = list(mean_returns.index)
    rows = []
    previous_weights: Optional[np.ndarray] = None
    for A in CALIBRATION_GRID:
        row = solve_optimal_portfolio(
            "no_short",
            mean_returns,
            cov_matrix,
            annual_fee_pct,
            returns_df,
            float(A),
            initial_weights=previous_weights,
        )
        previous_weights = extract_weights(row, fund_names)
        row["A"] = float(A)
        rows.append(row)
    return pd.DataFrame(rows)


def select_calibration_row(calibration_grid: pd.DataFrame, target_volatility: float) -> pd.Series:
    scored = calibration_grid.copy()
    scored["distance_to_target_vol"] = (scored["volatility"] - target_volatility).abs()
    scored = scored.sort_values(["distance_to_target_vol", "A"], ascending=[True, False]).reset_index(drop=True)
    return scored.iloc[0]


def calibration_summary(questionnaire_profile: Dict, selected_row: pd.Series) -> Dict:
    calibrated_A = float(selected_row["A"])
    calibrated_vol = float(selected_row["volatility"])
    return {
        "total_raw_score": questionnaire_profile["total_raw_score"],
        "risk_preference_score": questionnaire_profile["risk_preference_score"],
        "risk_capacity_score": questionnaire_profile["risk_capacity_score"],
        "target_vol_pref": questionnaire_profile["target_vol_pref"],
        "target_vol_cap": questionnaire_profile["target_vol_cap"],
        "final_target_vol": questionnaire_profile["final_target_vol"],
        "binding_construct": questionnaire_profile["binding_construct"],
        "risk_aversion_A": calibrated_A,
        "calibrated_portfolio_volatility": calibrated_vol,
        "calibration_gap": abs(calibrated_vol - questionnaire_profile["final_target_vol"]),
        "investor_type": investor_type_from_target_vol(questionnaire_profile["final_target_vol"]),
    }


def direct_risk_aversion_summary(questionnaire_profile: Dict, optimal_row: Dict) -> Dict:
    portfolio_volatility = float(optimal_row["volatility"])
    return {
        "total_raw_score": questionnaire_profile["total_raw_score"],
        "risk_preference_score": questionnaire_profile["risk_preference_score"],
        "risk_capacity_score": questionnaire_profile["risk_capacity_score"],
        "target_vol_pref": questionnaire_profile["target_vol_pref"],
        "target_vol_cap": questionnaire_profile["target_vol_cap"],
        "final_target_vol": questionnaire_profile["final_target_vol"],
        "final_risk_score": questionnaire_profile["final_risk_score"],
        "risk_category": questionnaire_profile["risk_category"],
        "binding_construct": questionnaire_profile["binding_construct"],
        "risk_aversion_A": questionnaire_profile["risk_aversion_A"],
        "risk_aversion_lower_A": questionnaire_profile["risk_aversion_lower_A"],
        "risk_aversion_upper_A": questionnaire_profile["risk_aversion_upper_A"],
        "calibrated_portfolio_volatility": portfolio_volatility,
        "calibration_gap": abs(portfolio_volatility - questionnaire_profile["final_target_vol"]),
        "calibration_method": "preference_band_capacity_adjusted_A",
        "investor_type": investor_type_from_target_vol(questionnaire_profile["final_target_vol"]),
    }


def a_sensitivity_tables(calibration_grid: pd.DataFrame, fund_names: List[str], calibrated_A: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scenario_specs = [
        ("A - 10%", max(1.0, round(calibrated_A * 0.90, 2))),
        ("Calibrated A", round(calibrated_A, 2)),
        ("A + 10%", min(15.0, round(calibrated_A * 1.10, 2))),
    ]
    chosen_rows = []
    for label, target_A in scenario_specs:
        selected = calibration_grid.iloc[(calibration_grid["A"] - target_A).abs().argsort()].iloc[0]
        chosen_rows.append(
            {
                "scenario": label,
                "target_A": target_A,
                "selected_A": float(selected["A"]),
                "expected_return": float(selected["expected_return"]),
                "volatility": float(selected["volatility"]),
                "utility": float(selected["utility"]),
                "weighted_long_fee_pct": float(selected["weighted_long_fee_pct"]),
            }
        )

    summary_df = pd.DataFrame(chosen_rows)
    base = calibration_grid.iloc[(calibration_grid["A"] - round(calibrated_A, 2)).abs().argsort()].iloc[0]
    change_rows = []
    for row in chosen_rows:
        selected = calibration_grid.iloc[(calibration_grid["A"] - row["selected_A"]).abs().argsort()].iloc[0]
        for fund_name in fund_names:
            change_rows.append(
                {
                    "scenario": row["scenario"],
                    "fund_name": fund_name,
                    "weight": float(selected[fund_name]),
                    "delta_vs_calibrated": float(selected[fund_name] - base[fund_name]),
                }
            )
    return summary_df, pd.DataFrame(change_rows)


def volatility_band_sensitivity(
    answers: Dict[str, str],
    calibration_grid: pd.DataFrame,
    fund_names: List[str],
) -> pd.DataFrame:
    rows = []
    for label, lower_bound, upper_bound in VOLATILITY_BAND_SPECS:
        _, _, profile = score_questionnaire(answers, lower_vol_bound=lower_bound, upper_vol_bound=upper_bound)
        selected = calibration_grid.iloc[(calibration_grid["A"] - profile["risk_aversion_A"]).abs().argsort()].iloc[0]
        rows.append(
            {
                "volatility_band": label,
                "volatility_band_lower": lower_bound,
                "volatility_band_upper": upper_bound,
                "binding_construct": profile["binding_construct"],
                "final_risk_score": float(profile["final_risk_score"]),
                "risk_category": profile["risk_category"],
                "target_volatility": float(profile["final_target_vol"]),
                "calibrated_A": float(profile["risk_aversion_A"]),
                "direct_A": float(profile["risk_aversion_A"]),
                "risk_aversion_lower_A": float(profile["risk_aversion_lower_A"]),
                "risk_aversion_upper_A": float(profile["risk_aversion_upper_A"]),
                "selected_grid_A": float(selected["A"]),
                "expected_return": float(selected["expected_return"]),
                "volatility": float(selected["volatility"]),
                "utility": float(selected["utility"]),
                "weighted_long_fee_pct": float(selected["weighted_long_fee_pct"]),
                "top_three_weights": serialize_top_positions(
                    np.array([float(selected[fund_name]) for fund_name in fund_names], dtype=float),
                    fund_names,
                    positive=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def resampled_no_short_portfolio(
    bootstrap_detailed_df: pd.DataFrame,
    fund_names: List[str],
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    A: float,
) -> Dict:
    average_weights = bootstrap_detailed_df[fund_names].mean().to_numpy(dtype=float)
    if np.allclose(average_weights.sum(), 0.0):
        raise ValueError("Bootstrap average weights sum to zero; cannot build resampled portfolio.")
    average_weights = np.clip(average_weights, 0.0, None)
    average_weights = average_weights / average_weights.sum()
    return portfolio_row(
        portfolio_name="Resampled Portfolio no_short",
        weights=average_weights,
        fund_names=fund_names,
        mean_returns=mean_returns.to_numpy(dtype=float),
        cov_matrix=cov_matrix.to_numpy(dtype=float),
        annual_fee_pct=annual_fee_pct.to_numpy(dtype=float),
        returns_df=returns_df,
        A=A,
    )


def portfolio_mechanism_summary(
    metadata: pd.DataFrame,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    optimal_no_short: Dict,
    optimal_bounded_short: Dict,
) -> pd.DataFrame:
    role_map = {
        "money market/cash": "cash anchor / GMVP magnet",
        "short-duration bond": "long-only ballast",
        "global investment-grade bond": "diversifying credit ballast",
        "higher-yield bond": "credit beta sleeve",
        "singapore equity": "regional equity growth engine",
        "u.s. equity": "global growth engine",
        "global developed equity": "broad equity proxy / crowded sleeve",
        "asia ex-japan equity": "regional equity diversifier",
        "global REIT": "listed property beta / short candidate",
        "gold/real-asset hedge": "inflation hedge / diversifier",
    }
    rows = []
    for _, row in metadata.iterrows():
        fund_name = row["fund_name"]
        rows.append(
            {
                "fund_name": fund_name,
                "sleeve": row["sleeve"],
                "annualized_mean_return": float(mean_returns[fund_name]),
                "annualized_volatility": float(np.sqrt(max(cov_matrix.loc[fund_name, fund_name], 0.0))),
                "corr_with_infinity_us_500": float(correlation_matrix.loc[fund_name, "Infinity US 500 Stock Index SGD"]),
                "corr_with_fidelity_world": float(correlation_matrix.loc[fund_name, "Fidelity World A-ACC-SGD"]),
                "corr_with_blackrock_gold": float(correlation_matrix.loc[fund_name, "Blackrock World Gold Fund A2 SGD-H"]),
                "weight_optimal_no_short": float(optimal_no_short[fund_name]),
                "weight_optimal_bounded_short": float(optimal_bounded_short[fund_name]),
                "role_tag": role_map[row["sleeve"]],
            }
        )
    return pd.DataFrame(rows)


def solve_optimal_portfolio_custom_bounds(
    portfolio_name: str,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    A: float,
    lower_bound: float,
    upper_bound: float,
    gross_limit: float,
) -> Dict:
    mu = mean_returns.to_numpy(dtype=float)
    sigma = cov_matrix.to_numpy(dtype=float)

    def objective(w):
        return -(portfolio_return(w, mu) - 0.5 * A * portfolio_variance(w, sigma))

    constraints = [_weight_sum_constraint()]
    if gross_limit > 1.0:
        constraints.append(_gross_exposure_constraint(gross_limit))

    weights = _solve(
        objective=objective,
        x0=_initial_weights(len(mu)),
        bounds=[(lower_bound, upper_bound)] * len(mu),
        constraints=constraints,
        label=portfolio_name,
    )
    return portfolio_row(
        portfolio_name=portfolio_name,
        weights=weights,
        fund_names=list(mean_returns.index),
        mean_returns=mu,
        cov_matrix=sigma,
        annual_fee_pct=annual_fee_pct.to_numpy(dtype=float),
        returns_df=returns_df,
        A=A,
    )


def bounded_short_parameter_sensitivity(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    A: float,
) -> pd.DataFrame:
    rows = []
    for regime in BOUNDED_SHORT_REGIMES:
        portfolio = solve_optimal_portfolio_custom_bounds(
            portfolio_name=f"Bounded-short {regime['regime']}",
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            annual_fee_pct=annual_fee_pct,
            returns_df=returns_df,
            A=A,
            lower_bound=regime["lower_bound"],
            upper_bound=regime["upper_bound"],
            gross_limit=regime["gross_limit"],
        )
        weights = extract_weights(portfolio, list(mean_returns.index))
        utility_after_total_cost = utility_from_stats(
            float(portfolio["expected_return"]) - borrow_cost_drag(weights, 0.03),
            float(portfolio["variance"]),
            A,
        )
        rows.append(
            {
                "regime": regime["regime"],
                "long_cap": regime["upper_bound"],
                "short_cap": abs(regime["lower_bound"]),
                "gross_cap": regime["gross_limit"],
                "expected_return": float(portfolio["expected_return"]),
                "volatility": float(portfolio["volatility"]),
                "max_drawdown": float(portfolio["max_drawdown"]),
                "utility": float(portfolio["utility"]),
                "utility_after_3pct_borrow_cost": float(utility_after_total_cost),
                "weighted_long_fee_pct": float(portfolio["weighted_long_fee_pct"]),
                "gross_exposure": float(portfolio["gross_exposure"]),
                "short_exposure": float(portfolio["short_exposure"]),
                "top_long_positions": portfolio["top_long_positions"],
                "top_short_positions": portfolio["top_short_positions"],
                "respects_regime_bounds": bool(
                    float(portfolio["gross_exposure"]) <= regime["gross_limit"] + 1e-8
                    and (weights >= regime["lower_bound"] - 1e-8).all()
                    and (weights <= regime["upper_bound"] + 1e-8).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def archetype_checks(calibration_grid: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "Low Risk": {question["question_id"]: "A" for question in QUESTIONNAIRE},
        "Balanced": {
            "q1": "C",
            "q2": "C",
            "q3": "C",
            "q4": "C",
            "q5": "C",
            "q6": "B",
            "q7": "B",
            "q8": "C",
            "q9": "C",
            "q10": "C",
        },
        "Growth": {question["question_id"]: "D" for question in QUESTIONNAIRE},
    }
    rows = []
    for name, answers in scenarios.items():
        _, _, profile = score_questionnaire(answers)
        rows.append(
            {
                "scenario": name,
                "risk_preference_score": profile["risk_preference_score"],
                "risk_capacity_score": profile["risk_capacity_score"],
                "final_risk_score": profile["final_risk_score"],
                "risk_category": profile["risk_category"],
                "final_target_vol": profile["final_target_vol"],
                "risk_aversion_A": float(profile["risk_aversion_A"]),
                "investor_type": investor_type_from_target_vol(profile["final_target_vol"]),
            }
        )
    return pd.DataFrame(rows)


def benchmark_table(portfolios: List[Dict]) -> pd.DataFrame:
    benchmark = pd.DataFrame(portfolios)
    columns = [
        "portfolio_name",
        "expected_return",
        "weighted_long_fee_pct",
        "variance",
        "volatility",
        "max_drawdown",
        "utility",
        "net_exposure",
        "long_exposure",
        "gross_exposure",
        "short_exposure",
        "client_ready",
        "top_long_positions",
        "top_short_positions",
    ]
    return benchmark[columns].copy()


def empirical_stress_scenarios(returns_df: pd.DataFrame, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sleeve_to_fund = metadata.set_index("sleeve")["fund_name"].to_dict()
    growth_fund = sleeve_to_fund["global developed equity"]
    ig_fund = sleeve_to_fund["global investment-grade bond"]
    gold_fund = sleeve_to_fund["gold/real-asset hedge"]
    min_selected_months = 6

    growth_threshold = returns_df[growth_fund].quantile(0.10)
    rates_threshold = returns_df[ig_fund].quantile(0.10)
    gold_threshold = returns_df[gold_fund].quantile(0.75)
    ig_lower_threshold = returns_df[ig_fund].quantile(0.25)

    growth_mask = returns_df[growth_fund] <= growth_threshold
    rates_mask = returns_df[ig_fund] <= rates_threshold
    inflation_mask = (returns_df[gold_fund] >= gold_threshold) & (returns_df[ig_fund] <= ig_lower_threshold)
    inflation_definition = (
        f"Average monthly returns when {gold_fund} was in its top quartile ({gold_threshold:.2%} or higher) "
        f"and {ig_fund} was in its bottom quartile ({ig_lower_threshold:.2%} or lower)."
    )
    if int(inflation_mask.sum()) < min_selected_months:
        rank_score = returns_df[gold_fund].rank(pct=True) - returns_df[ig_fund].rank(pct=True)
        selected_index = rank_score.sort_values(ascending=False).head(min_selected_months).index
        inflation_mask = returns_df.index.isin(selected_index)
        inflation_definition = (
            f"Average monthly returns during the top {min_selected_months} months ranked by high {gold_fund} performance "
            f"and weak {ig_fund} performance after the quartile intersection produced fewer than {min_selected_months} observations."
        )

    scenarios = [
        (
            "Growth Shock",
            growth_mask,
            f"Average monthly returns during months when {growth_fund} was in its worst decile ({growth_threshold:.2%} or lower).",
            "Proxy for a negative earnings-growth and risk-appetite state.",
        ),
        (
            "Rates Up",
            rates_mask,
            f"Average monthly returns during months when {ig_fund} was in its worst decile ({rates_threshold:.2%} or lower).",
            "Proxy for a discount-rate shock that hurts duration-sensitive assets.",
        ),
        (
            "Inflation Resurgence",
            inflation_mask,
            inflation_definition,
            "Proxy for a state in which nominal bonds weaken while real-asset hedges strengthen.",
        ),
    ]

    scenario_rows = []
    assumption_rows = []
    for name, mask, definition, proxy_text in scenarios:
        selected = returns_df.loc[mask]
        if selected.empty:
            raise ValueError(f"No observations were selected for the empirical stress scenario: {name}")
        mean_row = {"scenario": name}
        mean_row.update(selected.mean().to_dict())
        scenario_rows.append(mean_row)
        assumption_rows.append(
            {
                "scenario": name,
                "trigger_definition": definition,
                "economic_proxy_interpretation": proxy_text,
                "selected_month_count": int(selected.shape[0]),
                "first_selected_month": selected.index.min().date().isoformat(),
                "last_selected_month": selected.index.max().date().isoformat(),
            }
        )
    return pd.DataFrame(scenario_rows), pd.DataFrame(assumption_rows)


def stress_test_table(
    benchmark_df: pd.DataFrame,
    scenario_asset_returns: pd.DataFrame,
    metadata: pd.DataFrame,
    fund_names: List[str],
) -> pd.DataFrame:
    fund_to_sleeve = metadata.set_index("fund_name")["sleeve"].to_dict()
    rows = []
    for _, scenario in scenario_asset_returns.iterrows():
        scenario_name = scenario["scenario"]
        scenario_vector = {fund_name: float(scenario[fund_name]) for fund_name in fund_names}
        for _, portfolio in benchmark_df.iterrows():
            contributions = {
                fund_name: float(portfolio[fund_name]) * scenario_vector[fund_name]
                for fund_name in fund_names
            }
            biggest_loss_fund = min(contributions, key=contributions.get)
            biggest_hedge_fund = max(contributions, key=contributions.get)
            rows.append(
                {
                    "scenario": scenario_name,
                    "portfolio_name": portfolio["portfolio_name"],
                    "scenario_return": float(sum(contributions.values())),
                    "client_ready": bool(portfolio["client_ready"]),
                    "gross_exposure": float(portfolio["gross_exposure"]),
                    "biggest_loss_contributor": fund_to_sleeve[biggest_loss_fund],
                    "biggest_hedge_sleeve": fund_to_sleeve[biggest_hedge_fund],
                }
            )
    return pd.DataFrame(rows)


def rolling_stability_tables(
    returns_df: pd.DataFrame,
    annual_fee_pct: pd.Series,
    A: float,
    fund_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    previous_weights: Optional[np.ndarray] = None
    for end in range(ROLLING_WINDOW, len(returns_df) + 1):
        window_returns = returns_df.iloc[end - ROLLING_WINDOW : end]
        mean_returns = annualized_mean_returns(window_returns)
        cov_matrix = annualized_covariance(window_returns)
        repaired_cov, _, _ = covariance_psd_diagnostics(cov_matrix)
        row = solve_optimal_portfolio(
            "no_short",
            mean_returns,
            repaired_cov,
            annual_fee_pct,
            window_returns,
            A,
            initial_weights=previous_weights,
        )
        previous_weights = extract_weights(row, fund_names)
        row["window_start"] = window_returns.index.min().date().isoformat()
        row["window_end"] = window_returns.index.max().date().isoformat()
        rows.append(row)

    detailed = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "window_count": int(detailed.shape[0]),
                "mean_expected_return": float(detailed["expected_return"].mean()),
                "std_expected_return": float(detailed["expected_return"].std(ddof=0)),
                "mean_volatility": float(detailed["volatility"].mean()),
                "std_volatility": float(detailed["volatility"].std(ddof=0)),
                "mean_utility": float(detailed["utility"].mean()),
                "std_utility": float(detailed["utility"].std(ddof=0)),
            }
        ]
    )
    return detailed, summary


def bootstrap_stability_tables(
    returns_df: pd.DataFrame,
    annual_fee_pct: pd.Series,
    A: float,
    fund_names: List[str],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    previous_weights: Optional[np.ndarray] = None
    for iteration in range(1, iterations + 1):
        sample_idx = rng.integers(0, len(returns_df), size=len(returns_df))
        sampled_returns = returns_df.iloc[sample_idx].reset_index(drop=True)
        mean_returns = annualized_mean_returns(sampled_returns)
        cov_matrix = annualized_covariance(sampled_returns)
        repaired_cov, _, _ = covariance_psd_diagnostics(cov_matrix)
        row = solve_optimal_portfolio(
            "no_short",
            mean_returns,
            repaired_cov,
            annual_fee_pct,
            sampled_returns,
            A,
            initial_weights=previous_weights,
        )
        previous_weights = extract_weights(row, fund_names)
        row["iteration"] = iteration
        rows.append(row)

    detailed = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "iteration_count": int(detailed.shape[0]),
                "mean_expected_return": float(detailed["expected_return"].mean()),
                "std_expected_return": float(detailed["expected_return"].std(ddof=0)),
                "mean_volatility": float(detailed["volatility"].mean()),
                "std_volatility": float(detailed["volatility"].std(ddof=0)),
                "mean_utility": float(detailed["utility"].mean()),
                "std_utility": float(detailed["utility"].std(ddof=0)),
            }
        ]
    )
    return detailed, summary


def weight_summary_table(detailed_df: pd.DataFrame, fund_names: List[str], prefix: str) -> pd.DataFrame:
    rows = []
    for fund_name in fund_names:
        rows.append(
            {
                "fund_name": fund_name,
                f"{prefix}_mean_weight": float(detailed_df[fund_name].mean()),
                f"{prefix}_std_weight": float(detailed_df[fund_name].std(ddof=0)),
                f"{prefix}_min_weight": float(detailed_df[fund_name].min()),
                f"{prefix}_max_weight": float(detailed_df[fund_name].max()),
            }
        )
    return pd.DataFrame(rows)


def borrow_cost_sensitivity(
    optimal_no_short: Dict,
    optimal_bounded_short: Dict,
    A: float,
) -> pd.DataFrame:
    base_rows = [optimal_no_short, optimal_bounded_short]
    rows = []
    for borrow_rate in [0.00, 0.01, 0.02, 0.03]:
        for portfolio in base_rows:
            net_return = float(portfolio["expected_return"]) - float(portfolio["short_exposure"]) * borrow_rate
            rows.append(
                {
                    "portfolio_name": portfolio["portfolio_name"],
                    "borrow_cost_rate": borrow_rate,
                    "weighted_long_fee_pct": float(portfolio["weighted_long_fee_pct"]),
                    "short_exposure": float(portfolio["short_exposure"]),
                    "borrow_cost_drag": float(portfolio["short_exposure"]) * borrow_rate,
                    "expected_return_after_borrow_cost": net_return,
                    "utility_after_borrow_cost": utility_from_stats(net_return, float(portfolio["variance"]), A),
                }
            )
    sensitivity_df = pd.DataFrame(rows)
    gap_rows = []
    for borrow_rate, group in sensitivity_df.groupby("borrow_cost_rate"):
        no_short_utility = float(group.loc[group["portfolio_name"] == "Optimal Portfolio no_short", "utility_after_borrow_cost"].iloc[0])
        for _, row in group.iterrows():
            gap_rows.append(
                {
                    **row.to_dict(),
                    "utility_gap_vs_no_short": float(row["utility_after_borrow_cost"] - no_short_utility),
                }
            )
    return pd.DataFrame(gap_rows)


def fund_selection_evidence_table(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata.loc[
        :,
        [
            "sleeve",
            "fund_name",
            "rejected_alternatives",
            "tie_break_reason",
            "evidence_reference",
            "selection_status",
        ],
    ].rename(columns={"fund_name": "chosen_fund"})


def robustness_table(
    returns_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    fund_names: List[str],
    A: float,
) -> pd.DataFrame:
    first_cutoff = pd.Timestamp("2023-12-31")
    if returns_df.index.min() > first_cutoff or returns_df.index.max() <= first_cutoff:
        midpoint = len(returns_df) // 2
        period_map = {"Subperiod 1": returns_df.iloc[:midpoint], "Subperiod 2": returns_df.iloc[midpoint:]}
    else:
        period_map = {"2021-2023": returns_df.loc[returns_df.index <= first_cutoff], "2024-2025": returns_df.loc[returns_df.index > first_cutoff]}

    rows = []
    for period_name, period_returns in period_map.items():
        mean_returns = annualized_mean_returns(period_returns)
        cov_matrix = annualized_covariance(period_returns)
        repaired_cov, _, _ = covariance_psd_diagnostics(cov_matrix)
        mu = mean_returns.to_numpy(dtype=float)
        sigma = repaired_cov.to_numpy(dtype=float)
        for _, portfolio in benchmark_df.iterrows():
            weights = portfolio[fund_names].to_numpy(dtype=float)
            expected_return = portfolio_return(weights, mu)
            variance = portfolio_variance(weights, sigma)
            rows.append(
                {
                    "period": period_name,
                    "portfolio_name": portfolio["portfolio_name"],
                    "expected_return": expected_return,
                    "volatility": np.sqrt(max(variance, 0.0)),
                    "utility": utility_from_stats(expected_return, variance, A),
                    "gross_exposure": float(np.sum(np.abs(weights))),
                }
            )
    return pd.DataFrame(rows)


def validation_summary(
    metadata: pd.DataFrame,
    price_df: pd.DataFrame,
    frontier_no_short: pd.DataFrame,
    frontier_unconstrained_short: pd.DataFrame,
    frontier_bounded_short: pd.DataFrame,
    optimal_no_short: Dict,
    optimal_unconstrained_short: Dict,
    optimal_bounded_short: Dict,
    resampled_no_short: Dict,
    archetypes: pd.DataFrame,
    fund_names: List[str],
    psd_check: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    bounded_short_sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    no_short_weights = frontier_no_short[fund_names]
    unconstrained_short_weights = frontier_unconstrained_short[fund_names]
    bounded_short_weights = frontier_bounded_short[fund_names]
    checks = [
        {
            "check": "All funds are final FSMOne funds",
            "status": bool((metadata["selection_status"] == "final").all()),
            "details": f"{int((metadata['selection_status'] == 'final').sum())}/10 final",
        },
        {
            "check": "Exactly 10 funds in metadata",
            "status": len(metadata) == 10,
            "details": str(len(metadata)),
        },
        {
            "check": "At least 60 aligned monthly observations",
            "status": len(price_df) >= 60,
            "details": str(len(price_df)),
        },
        {
            "check": "No missing prices in aligned panel",
            "status": int(price_df.isnull().sum().sum()) == 0,
            "details": str(int(price_df.isnull().sum().sum())),
        },
        {
            "check": "Covariance PSD status disclosed",
            "status": True,
            "details": "repair_applied=" + str(bool(psd_check["repair_applied"].iloc[0])),
        },
        {
            "check": "No-short frontier respects bounds",
            "status": bool(((no_short_weights >= -1e-8) & (no_short_weights <= 1.0 + 1e-8)).all().all()),
            "details": "0 <= w_i <= 1",
        },
        {
            "check": "Unconstrained-short frontier weights sum to one",
            "status": bool(np.allclose(unconstrained_short_weights.sum(axis=1), 1.0, atol=1e-6)),
            "details": "sum(w)=1 with no asset-level caps",
        },
        {
            "check": "Bounded-short frontier respects bounds",
            "status": bool(((bounded_short_weights >= -0.20 - 1e-8) & (bounded_short_weights <= 0.35 + 1e-8)).all().all()),
            "details": "-0.20 <= w_i <= 0.35",
        },
        {
            "check": "Bounded-short frontier respects gross limit",
            "status": bool((frontier_bounded_short["gross_exposure"] <= 1.40 + 1e-8).all()),
            "details": "gross <= 1.40",
        },
        {
            "check": "Optimal no-short weights sum to one",
            "status": np.isclose(sum(optimal_no_short[fund] for fund in fund_names), 1.0),
            "details": f"{sum(optimal_no_short[fund] for fund in fund_names):.8f}",
        },
        {
            "check": "Optimal unconstrained-short weights sum to one",
            "status": np.isclose(sum(optimal_unconstrained_short[fund] for fund in fund_names), 1.0),
            "details": f"{sum(optimal_unconstrained_short[fund] for fund in fund_names):.8f}",
        },
        {
            "check": "Optimal bounded-short weights sum to one",
            "status": np.isclose(sum(optimal_bounded_short[fund] for fund in fund_names), 1.0),
            "details": f"{sum(optimal_bounded_short[fund] for fund in fund_names):.8f}",
        },
        {
            "check": "Resampled no-short weights sum to one",
            "status": np.isclose(sum(resampled_no_short[fund] for fund in fund_names), 1.0),
            "details": f"{sum(resampled_no_short[fund] for fund in fund_names):.8f}",
        },
        {
            "check": "Calibrated A is monotonic across archetypes",
            "status": bool(archetypes["risk_aversion_A"].is_monotonic_decreasing),
            "details": "Low Risk A > Balanced A > Growth A",
        },
        {
            "check": "Fee treatment disclosed as NAV-based",
            "status": "weighted_long_fee_pct" in benchmark_df.columns and "net_expected_return_after_fee" not in benchmark_df.columns,
            "details": "Historical NAV/chart returns are used directly; annual_fee_pct is retained as disclosure only.",
        },
        {
            "check": "Bounded-short sensitivity regimes respect configured caps",
            "status": bool(bounded_short_sensitivity_df["respects_regime_bounds"].all()),
            "details": "All conservative / baseline / liberal regimes satisfy their own caps.",
        },
    ]
    return pd.DataFrame(checks)


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    asset_points_df: pd.DataFrame,
    gmvp_row: Dict,
    optimal_row: Optional[Dict],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    fig.patch.set_facecolor(PLOT_THEME["paper"])
    ax.set_facecolor(PLOT_THEME["panel"])
    ax.fill_between(
        frontier_df["volatility"],
        frontier_df["expected_return"],
        frontier_df["expected_return"].min(),
        color=PLOT_THEME["gold"],
        alpha=0.12,
        zorder=1,
    )
    ax.plot(
        frontier_df["volatility"],
        frontier_df["expected_return"],
        linewidth=3.0,
        color=PLOT_THEME["navy"],
        label="Efficient Frontier",
        zorder=2,
    )
    point_colors = [ASSET_PLOT_COLORS.get(label, PLOT_THEME["teal"]) for label in asset_points_df["short_label"]]
    ax.scatter(
        asset_points_df["volatility"],
        asset_points_df["expected_return"],
        s=65,
        alpha=0.9,
        color=point_colors,
        edgecolor=PLOT_THEME["soft_white"],
        linewidth=1.1,
        label="Individual Funds",
        zorder=3,
    )
    for _, point in asset_points_df.iterrows():
        dx, dy = FRONTIER_LABEL_OFFSETS.get(point["short_label"], (6, 6))
        ax.annotate(
            point["short_label"],
            (point["volatility"], point["expected_return"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=PLOT_THEME["ink"],
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": PLOT_THEME["soft_white"],
                "edgecolor": ASSET_PLOT_COLORS.get(point["short_label"], PLOT_THEME["teal"]),
                "linewidth": 1.0,
                "alpha": 0.96,
            },
        )
    ax.scatter(
        gmvp_row["volatility"],
        gmvp_row["expected_return"],
        marker="X",
        s=200,
        color=PLOT_THEME["orange"],
        edgecolor=PLOT_THEME["soft_white"],
        linewidth=1.0,
        label="GMVP",
        zorder=5,
    )
    if optimal_row is not None:
        ax.scatter(
            optimal_row["volatility"],
            optimal_row["expected_return"],
            marker="D",
            s=140,
            color=PLOT_THEME["teal"],
            edgecolor=PLOT_THEME["soft_white"],
            linewidth=1.0,
            label="Investor Optimum",
            zorder=5,
        )
    ax.set_xlabel("Annualized Volatility", color=PLOT_THEME["ink"], fontweight="bold")
    ax.set_ylabel("Annualized Expected Return", color=PLOT_THEME["ink"], fontweight="bold")
    ax.set_title(title, color=PLOT_THEME["navy"], fontweight="bold", pad=14)
    ax.grid(color=PLOT_THEME["grid"], linestyle="--", linewidth=0.8, alpha=0.55)
    ax.tick_params(colors=PLOT_THEME["ink"])
    legend = ax.legend(frameon=True, facecolor=PLOT_THEME["soft_white"], edgecolor=PLOT_THEME["grid"])
    for text in legend.get_texts():
        text.set_color(PLOT_THEME["ink"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_weights(portfolio: Dict, fund_names: List[str], display_labels: List[str], title: str, save_path: Path) -> None:
    values = [portfolio[fund_name] for fund_name in fund_names]
    colors = [ASSET_PLOT_COLORS.get(label, PLOT_THEME["teal"]) if value >= 0 else PLOT_THEME["short_red"] for label, value in zip(display_labels, values)]
    edge_colors = [PLOT_THEME["navy"] if value >= 0 else PLOT_THEME["short_edge"] for value in values]
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(PLOT_THEME["paper"])
    ax.set_facecolor(PLOT_THEME["panel"])
    bars = ax.bar(display_labels, values, color=colors, edgecolor=edge_colors, linewidth=1.1, zorder=3)
    ax.axhline(0.0, color=PLOT_THEME["ink"], linewidth=1.0)
    y_padding = max(0.03, max(abs(value) for value in values) * 0.18)
    ax.set_ylim(min(values) - y_padding, max(values) + y_padding)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.01 if value >= 0 else -0.01),
            f"{value:.0%}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            color=PLOT_THEME["ink"],
            fontweight="bold",
        )
    ax.set_ylabel("Portfolio Weight", color=PLOT_THEME["ink"], fontweight="bold")
    ax.set_title(title, color=PLOT_THEME["navy"], fontweight="bold", pad=14)
    ax.grid(axis="y", color=PLOT_THEME["grid"], linestyle="--", linewidth=0.8, alpha=0.55, zorder=1)
    ax.tick_params(colors=PLOT_THEME["ink"])
    plt.xticks(rotation=28, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_correlation_heatmap(returns_df: pd.DataFrame, display_labels: List[str], save_path: Path) -> None:
    corr = returns_df.corr()
    fig, ax = plt.subplots(figsize=(9.2, 7.3))
    fig.patch.set_facecolor(PLOT_THEME["paper"])
    ax.set_facecolor(PLOT_THEME["panel"])
    image = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(display_labels)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = float(corr.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=PLOT_THEME["soft_white"] if abs(value) > 0.45 else PLOT_THEME["ink"],
                fontweight="bold" if i == j else "normal",
            )
    ax.set_title("Fund Return Correlation Heatmap", color=PLOT_THEME["navy"], fontweight="bold", pad=14)
    ax.tick_params(colors=PLOT_THEME["ink"])
    ax.set_xticks(np.arange(-0.5, len(corr.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(corr.index), 1), minor=True)
    ax.grid(which="minor", color=PLOT_THEME["soft_white"], linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=PLOT_THEME["ink"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_utility_comparison(benchmark_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PLOT_THEME["paper"])
    ax.set_facecolor(PLOT_THEME["panel"])
    labels = [PORTFOLIO_PLOT_LABELS.get(name, name) for name in benchmark_df["portfolio_name"]]
    colors = [PORTFOLIO_BAR_COLORS.get(name, PLOT_THEME["teal"]) for name in benchmark_df["portfolio_name"]]
    bars = ax.bar(labels, benchmark_df["utility"], color=colors, edgecolor=PLOT_THEME["soft_white"], linewidth=1.1, zorder=3)
    ax.grid(axis="y", color=PLOT_THEME["grid"], linestyle="--", linewidth=0.8, alpha=0.55, zorder=1)
    for bar, value in zip(bars, benchmark_df["utility"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.0015,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PLOT_THEME["ink"],
            fontweight="bold",
        )
    ax.set_ylabel("Mean-Variance Utility", color=PLOT_THEME["ink"], fontweight="bold")
    ax.set_title("Utility Comparison Across Key Portfolios", color=PLOT_THEME["navy"], fontweight="bold", pad=14)
    ax.tick_params(colors=PLOT_THEME["ink"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary_json(
    filepath: Path,
    metadata: pd.DataFrame,
    price_df: pd.DataFrame,
    investor_profile: Dict,
    gmvp_no_short: Dict,
    gmvp_unconstrained_short: Dict,
    gmvp_bounded_short: Dict,
    optimal_no_short: Dict,
    optimal_unconstrained_short: Dict,
    optimal_bounded_short: Dict,
    blockers: List[str],
    submission_mode: str,
    psd_check: pd.DataFrame,
) -> None:
    proxy_warning = bool((metadata["selection_status"] != "final").any())
    summary = {
        "proxy_warning": proxy_warning,
        "submission_mode": submission_mode,
        "submission_ready": not blockers,
        "remaining_blockers": blockers,
        "real_fund_count": int((metadata["selection_status"] == "final").sum()),
        "date_start": str(price_df.index.min().date()),
        "date_end": str(price_df.index.max().date()),
        "observation_count": int(len(price_df)),
        "currency_policy": "All funds are expected to be SGD or SGD-hedged share classes.",
        "questionnaire_method": (
            "Questionnaire answers are separated into risk-preference and risk-capacity constructs. "
            "Risk preference maps the investor into a risk-appetite category and A interval; risk capacity then selects the final A within that interval, with weaker capacity moving toward the more conservative upper-A endpoint."
        ),
        "solver_method": "SLSQP with equal-weight initialization and ftol=1e-10.",
        "covariance_psd_repair_applied": bool(psd_check["repair_applied"].iloc[0]),
        "recommended_policy": "no_short",
        "strict_short_sales_policy": "unconstrained_short",
        "research_extension_policy": "bounded_short",
        "fee_treatment": "Historical FSMOne NAV/chart returns are treated as already net of fund-level expenses; annual_fee_pct is retained as disclosure only.",
        "investor_profile": investor_profile,
        "gmvp_no_short": gmvp_no_short,
        "gmvp_unconstrained_short": gmvp_unconstrained_short,
        "gmvp_bounded_short": gmvp_bounded_short,
        "optimal_no_short": optimal_no_short,
        "optimal_unconstrained_short": optimal_unconstrained_short,
        "optimal_bounded_short": optimal_bounded_short,
    }
    filepath.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    validate_questionnaire_weights()
    args = parse_args()

    root = Path(args.outputs)
    tables_dir = root / "tables"
    plots_dir = root / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    metadata = load_fund_metadata(args.fund_meta)
    blockers = submission_blockers(metadata, args.submission_mode)
    fund_names = metadata["fund_name"].tolist()
    sleeve_lookup = metadata.set_index("fund_name")["sleeve"].to_dict()
    price_df, coverage_df = load_price_data(args.prices, fund_names)
    returns_df = compute_returns(price_df)
    annual_fee_pct = metadata.set_index("fund_name")["annual_fee_pct"].reindex(fund_names)

    mean_returns = annualized_mean_returns(returns_df)
    raw_cov_matrix = annualized_covariance(returns_df)
    opt_cov_matrix, psd_check_df, repair_note_df = covariance_psd_diagnostics(raw_cov_matrix)
    correlation_matrix = returns_df.corr()
    asset_points_df = asset_points(mean_returns, opt_cov_matrix, metadata)
    display_labels = [SLEEVE_SHORT_LABELS[sleeve_lookup[fund_name]] for fund_name in fund_names]

    answers = load_answers(args.answers)
    questionnaire_df, construct_summary_df, questionnaire_profile = score_questionnaire(answers)
    A = float(questionnaire_profile["risk_aversion_A"])
    optimal_no_short = solve_optimal_portfolio("no_short", mean_returns, opt_cov_matrix, annual_fee_pct, returns_df, A=A)
    investor_profile = direct_risk_aversion_summary(questionnaire_profile, optimal_no_short)
    calibration_grid_df = build_a_calibration_grid(mean_returns, opt_cov_matrix, annual_fee_pct, returns_df)
    selected_calibration = calibration_grid_df.iloc[(calibration_grid_df["A"] - A).abs().argsort()].iloc[0].copy()
    selected_calibration["direct_A"] = A
    selected_calibration["distance_to_direct_A"] = abs(float(selected_calibration["A"]) - A)

    gmvp_no_short = solve_gmvp("no_short", mean_returns, opt_cov_matrix, annual_fee_pct, returns_df, A=A)
    gmvp_unconstrained_short = solve_gmvp("unconstrained_short", mean_returns, opt_cov_matrix, annual_fee_pct, returns_df, A=A)
    gmvp_bounded_short = solve_gmvp("bounded_short", mean_returns, opt_cov_matrix, annual_fee_pct, returns_df, A=A)
    optimal_unconstrained_short = solve_optimal_portfolio(
        "unconstrained_short",
        mean_returns,
        opt_cov_matrix,
        annual_fee_pct,
        returns_df,
        A=A,
    )
    optimal_bounded_short = solve_optimal_portfolio("bounded_short", mean_returns, opt_cov_matrix, annual_fee_pct, returns_df, A=A)
    unconstrained_frontier_upper = max(
        float(mean_returns.max()) * 1.15,
        float(optimal_unconstrained_short["expected_return"]) * 1.05,
        float(optimal_bounded_short["expected_return"]) * 1.15,
    )
    frontier_no_short = generate_frontier("no_short", mean_returns, opt_cov_matrix, annual_fee_pct, returns_df)
    frontier_unconstrained_short = generate_frontier(
        "unconstrained_short",
        mean_returns,
        opt_cov_matrix,
        annual_fee_pct,
        returns_df,
        upper_target_return=unconstrained_frontier_upper,
    )
    frontier_bounded_short = generate_frontier("bounded_short", mean_returns, opt_cov_matrix, annual_fee_pct, returns_df)
    equal_weight = equal_weight_portfolio(mean_returns, opt_cov_matrix, annual_fee_pct, returns_df, A=A)
    vol_band_sensitivity_df = volatility_band_sensitivity(answers, calibration_grid_df, fund_names)

    bootstrap_detailed_df, bootstrap_summary_df = bootstrap_stability_tables(returns_df, annual_fee_pct, A, fund_names)
    bootstrap_weight_summary_df = weight_summary_table(bootstrap_detailed_df, fund_names, "bootstrap")
    resampled_no_short = resampled_no_short_portfolio(
        bootstrap_detailed_df,
        fund_names,
        mean_returns,
        opt_cov_matrix,
        annual_fee_pct,
        returns_df,
        A,
    )
    portfolio_rows = [equal_weight, gmvp_no_short, gmvp_bounded_short, optimal_no_short, resampled_no_short, optimal_bounded_short]
    benchmark_df = pd.DataFrame(portfolio_rows)

    scenario_asset_returns, stress_assumptions_df = empirical_stress_scenarios(returns_df, metadata)
    stress_df = stress_test_table(benchmark_df, scenario_asset_returns, metadata, fund_names)
    robustness_df = robustness_table(returns_df, benchmark_df, fund_names, A)
    rolling_detailed_df, rolling_summary_df = rolling_stability_tables(returns_df, annual_fee_pct, A, fund_names)
    rolling_weight_summary_df = weight_summary_table(rolling_detailed_df, fund_names, "rolling")
    archetypes_df = archetype_checks(calibration_grid_df)
    a_sensitivity_df, a_changes_df = a_sensitivity_tables(calibration_grid_df, fund_names, A)
    borrow_cost_df = borrow_cost_sensitivity(optimal_no_short, optimal_bounded_short, A)
    portfolio_mechanism_df = portfolio_mechanism_summary(
        metadata,
        mean_returns,
        opt_cov_matrix,
        correlation_matrix,
        optimal_no_short,
        optimal_bounded_short,
    )
    bounded_short_sensitivity_df = bounded_short_parameter_sensitivity(
        mean_returns,
        opt_cov_matrix,
        annual_fee_pct,
        returns_df,
        A,
    )
    validation_df = validation_summary(
        metadata,
        price_df,
        frontier_no_short,
        frontier_unconstrained_short,
        frontier_bounded_short,
        optimal_no_short,
        optimal_unconstrained_short,
        optimal_bounded_short,
        resampled_no_short,
        archetypes_df,
        fund_names,
        psd_check_df,
        benchmark_df,
        bounded_short_sensitivity_df,
    )

    metadata.to_csv(tables_dir / "fund_metadata_rationale.csv", index=False)
    fund_selection_evidence_table(metadata).to_csv(tables_dir / "fund_selection_evidence.csv", index=False)
    coverage_df.to_csv(tables_dir / "price_coverage.csv", index=False)
    mean_returns.rename("annualized_mean_return").reset_index().rename(columns={"index": "fund_name"}).to_csv(
        tables_dir / "mean_returns.csv", index=False
    )
    raw_cov_matrix.to_csv(tables_dir / "covariance_matrix.csv")
    opt_cov_matrix.to_csv(tables_dir / "covariance_matrix_repaired.csv")
    psd_check_df.to_csv(tables_dir / "covariance_psd_check.csv", index=False)
    repair_note_df.to_csv(tables_dir / "covariance_repair_note.csv", index=False)
    correlation_matrix.to_csv(tables_dir / "correlation_matrix.csv")
    asset_points_df.to_csv(tables_dir / "individual_fund_points.csv", index=False)
    frontier_no_short.to_csv(tables_dir / "frontier_no_short.csv", index=False)
    frontier_unconstrained_short.to_csv(tables_dir / "frontier_unconstrained_short.csv", index=False)
    frontier_bounded_short.to_csv(tables_dir / "frontier_bounded_short.csv", index=False)
    pd.DataFrame([gmvp_no_short]).to_csv(tables_dir / "gmvp_no_short.csv", index=False)
    pd.DataFrame([gmvp_unconstrained_short]).to_csv(tables_dir / "gmvp_unconstrained_short.csv", index=False)
    pd.DataFrame([gmvp_bounded_short]).to_csv(tables_dir / "gmvp_bounded_short.csv", index=False)
    questionnaire_df.to_csv(tables_dir / "questionnaire_scoring.csv", index=False)
    construct_summary_df.to_csv(tables_dir / "questionnaire_construct_summary.csv", index=False)
    pd.DataFrame([investor_profile]).to_csv(tables_dir / "investor_profile_summary.csv", index=False)
    calibration_grid_df.to_csv(tables_dir / "a_calibration_grid.csv", index=False)
    pd.DataFrame([selected_calibration]).to_csv(tables_dir / "target_vol_calibration.csv", index=False)
    vol_band_sensitivity_df.to_csv(tables_dir / "vol_band_sensitivity.csv", index=False)
    pd.DataFrame([optimal_no_short]).to_csv(tables_dir / "optimal_portfolio_no_short.csv", index=False)
    pd.DataFrame([optimal_unconstrained_short]).to_csv(tables_dir / "optimal_portfolio_unconstrained_short.csv", index=False)
    pd.DataFrame([resampled_no_short]).to_csv(tables_dir / "resampled_no_short_portfolio.csv", index=False)
    pd.DataFrame([optimal_bounded_short]).to_csv(tables_dir / "optimal_portfolio_bounded_short.csv", index=False)
    portfolio_mechanism_df.to_csv(tables_dir / "portfolio_mechanism_summary.csv", index=False)
    benchmark_table(portfolio_rows).to_csv(tables_dir / "benchmark_comparison.csv", index=False)
    robustness_df.to_csv(tables_dir / "robustness_subperiods.csv", index=False)
    scenario_asset_returns.to_csv(tables_dir / "stress_scenario_asset_returns.csv", index=False)
    stress_assumptions_df.to_csv(tables_dir / "stress_assumptions.csv", index=False)
    stress_df.to_csv(tables_dir / "stress_test_scenarios.csv", index=False)
    rolling_detailed_df.to_csv(tables_dir / "rolling_no_short_optima.csv", index=False)
    rolling_summary_df.to_csv(tables_dir / "rolling_no_short_summary.csv", index=False)
    rolling_weight_summary_df.to_csv(tables_dir / "rolling_no_short_weight_summary.csv", index=False)
    bootstrap_detailed_df.to_csv(tables_dir / "bootstrap_no_short_iterations.csv", index=False)
    bootstrap_summary_df.to_csv(tables_dir / "bootstrap_no_short_summary.csv", index=False)
    bootstrap_weight_summary_df.to_csv(tables_dir / "bootstrap_no_short_weight_summary.csv", index=False)
    archetypes_df.to_csv(tables_dir / "questionnaire_archetype_checks.csv", index=False)
    a_sensitivity_df.to_csv(tables_dir / "a_sensitivity_summary.csv", index=False)
    a_changes_df.to_csv(tables_dir / "a_sensitivity_portfolio_changes.csv", index=False)
    borrow_cost_df.to_csv(tables_dir / "borrow_cost_sensitivity.csv", index=False)
    bounded_short_sensitivity_df.to_csv(tables_dir / "bounded_short_parameter_sensitivity.csv", index=False)
    validation_df.to_csv(tables_dir / "validation_summary.csv", index=False)

    plot_efficient_frontier(
        frontier_df=frontier_no_short,
        asset_points_df=asset_points_df,
        gmvp_row=gmvp_no_short,
        optimal_row=optimal_no_short,
        title="Efficient Frontier (No Short Sales)",
        save_path=plots_dir / "efficient_frontier_no_short.png",
    )
    plot_efficient_frontier(
        frontier_df=frontier_unconstrained_short,
        asset_points_df=asset_points_df,
        gmvp_row=gmvp_unconstrained_short,
        optimal_row=optimal_unconstrained_short,
        title="Efficient Frontier (Unconstrained Short Sales)",
        save_path=plots_dir / "efficient_frontier_unconstrained_short.png",
    )
    plot_efficient_frontier(
        frontier_df=frontier_bounded_short,
        asset_points_df=asset_points_df,
        gmvp_row=gmvp_bounded_short,
        optimal_row=optimal_bounded_short,
        title="Efficient Frontier (Bounded Short Sales)",
        save_path=plots_dir / "efficient_frontier_bounded_short.png",
    )
    plot_weights(gmvp_no_short, fund_names, display_labels, "GMVP Weights (No Short Sales)", plots_dir / "gmvp_weights_no_short.png")
    plot_weights(
        gmvp_bounded_short,
        fund_names,
        display_labels,
        "GMVP Weights (Bounded Short Sales)",
        plots_dir / "gmvp_weights_bounded_short.png",
    )
    plot_weights(
        optimal_no_short,
        fund_names,
        display_labels,
        "Investor Optimal Weights (No Short Sales)",
        plots_dir / "optimal_weights_no_short.png",
    )
    plot_weights(
        optimal_bounded_short,
        fund_names,
        display_labels,
        "Investor Optimal Weights (Bounded Short Sales)",
        plots_dir / "optimal_weights_bounded_short.png",
    )
    plot_correlation_heatmap(returns_df, display_labels, plots_dir / "correlation_heatmap.png")
    plot_utility_comparison(benchmark_table(portfolio_rows), plots_dir / "utility_comparison.png")

    write_summary_json(
        tables_dir / "analysis_summary.json",
        metadata,
        price_df,
        investor_profile,
        gmvp_no_short,
        gmvp_unconstrained_short,
        gmvp_bounded_short,
        optimal_no_short,
        optimal_unconstrained_short,
        optimal_bounded_short,
        blockers,
        args.submission_mode,
        psd_check_df,
    )

    print("Analysis completed.")
    print(f"Data range: {price_df.index.min().date()} to {price_df.index.max().date()} ({len(price_df)} monthly observations)")
    print("Investor profile:", f"{investor_profile['investor_type']} with calibrated A={investor_profile['risk_aversion_A']:.4f}")
    if blockers:
        print("Remaining blockers before a true final submission:")
        for blocker in blockers:
            print(f"- {blocker}")
    print("Outputs written to:")
    print(f"- {tables_dir}")
    print(f"- {plots_dir}")


if __name__ == "__main__":
    main()
