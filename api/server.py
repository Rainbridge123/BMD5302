from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import portfolio_engine as pe


HOST = "127.0.0.1"
PORT = 8000
OUTPUTS_DIR = ROOT / "outputs" / "tables"
PRICES_PATH = ROOT / "data" / "fsmone_prices.csv"
FUND_META_PATH = ROOT / "data" / "fsmone_fund_universe.csv"
ANSWERS_PATH = ROOT / "data" / "questionnaire_answers.json"


@dataclass
class RuntimeContext:
    metadata: pd.DataFrame
    fund_names: List[str]
    price_df: pd.DataFrame
    returns_df: pd.DataFrame
    annual_fee_pct: pd.Series
    mean_returns: pd.Series
    covariance_matrix: pd.DataFrame
    calibration_grid: pd.DataFrame
    display_labels: List[str]
    sleeve_lookup: Dict[str, str]


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / name)


def _load_json(name: str) -> Dict[str, Any]:
    return json.loads((OUTPUTS_DIR / name).read_text(encoding="utf-8"))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.to_dict(orient="records")
    return _to_builtin(records)


def _questionnaire_schema() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for question in pe.QUESTIONNAIRE:
        rows.append(
            {
                "questionId": question["question_id"],
                "construct": question["construct"],
                "weight": question["weight"],
                "title": question["title"],
                "options": [
                    {
                        "code": code,
                        "text": option["text"],
                        "score": option["score"],
                    }
                    for code, option in question["options"].items()
                ],
            }
        )
    return rows


def _available_fund_options(context: RuntimeContext) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fund_name in context.fund_names:
        rows.append(
            {
                "fundName": fund_name,
                "shortLabel": pe.SLEEVE_SHORT_LABELS[context.sleeve_lookup[fund_name]],
                "sleeve": context.sleeve_lookup[fund_name],
            }
        )
    return rows


def _portfolio_weights(
    portfolio_row: Dict[str, Any],
    metadata: pd.DataFrame,
    selected_fund_names: List[str],
    sleeve_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fund_name in selected_fund_names:
        rows.append(
            {
                "fundName": fund_name,
                "shortLabel": pe.SLEEVE_SHORT_LABELS[sleeve_lookup[fund_name]],
                "sleeve": sleeve_lookup[fund_name],
                "weight": float(portfolio_row[fund_name]),
            }
        )
    rows.sort(key=lambda row: row["weight"], reverse=True)
    return rows


def _portfolio_performance_series(
    returns_df: pd.DataFrame,
    price_df: pd.DataFrame,
    fund_names: List[str],
    equal_weight_row: Dict[str, Any],
    gmvp_row: Dict[str, Any],
    no_short_row: Dict[str, Any],
    bounded_short_row: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    monthly_returns = returns_df.loc[:, fund_names].dropna()
    if monthly_returns.empty:
        return []

    return_matrix = monthly_returns.to_numpy(dtype=float)

    def portfolio_path(portfolio_row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        weights = np.array([float(portfolio_row[fund_name]) for fund_name in fund_names])
        portfolio_return = return_matrix @ weights
        cumulative_return = np.cumprod(1.0 + portfolio_return) - 1.0
        return portfolio_return, cumulative_return

    equal_weight_return, equal_weight_cumulative = portfolio_path(equal_weight_row)
    gmvp_return, gmvp_cumulative = portfolio_path(gmvp_row)
    no_short_return, no_short_cumulative = portfolio_path(no_short_row)

    bounded_short_return: Optional[np.ndarray] = None
    bounded_short_cumulative: Optional[np.ndarray] = None
    if bounded_short_row is not None:
        bounded_short_return, bounded_short_cumulative = portfolio_path(bounded_short_row)

    rows: List[Dict[str, Any]] = []
    if not price_df.empty:
        first_price_date = price_df.index.min()
        if first_price_date < monthly_returns.index.min():
            rows.append(
                {
                    "date": first_price_date.date().isoformat(),
                    "equalWeightReturn": 0.0,
                    "equalWeightCumulativeReturn": 0.0,
                    "gmvpReturn": 0.0,
                    "gmvpCumulativeReturn": 0.0,
                    "noShortReturn": 0.0,
                    "noShortCumulativeReturn": 0.0,
                    "boundedShortReturn": 0.0 if bounded_short_row is not None else None,
                    "boundedShortCumulativeReturn": 0.0 if bounded_short_row is not None else None,
                }
            )

    for idx, date in enumerate(monthly_returns.index):
        rows.append(
            {
                "date": date.date().isoformat(),
                "equalWeightReturn": float(equal_weight_return[idx]),
                "equalWeightCumulativeReturn": float(equal_weight_cumulative[idx]),
                "gmvpReturn": float(gmvp_return[idx]),
                "gmvpCumulativeReturn": float(gmvp_cumulative[idx]),
                "noShortReturn": float(no_short_return[idx]),
                "noShortCumulativeReturn": float(no_short_cumulative[idx]),
                "boundedShortReturn": float(bounded_short_return[idx]) if bounded_short_return is not None else None,
                "boundedShortCumulativeReturn": (
                    float(bounded_short_cumulative[idx]) if bounded_short_cumulative is not None else None
                ),
            }
        )
    return rows


def _resolve_selected_funds(context: RuntimeContext, requested_funds: Optional[List[str]]) -> List[str]:
    if not requested_funds:
        return context.fund_names

    known_funds = set(context.fund_names)
    normalized_requested = {fund_name for fund_name in requested_funds if fund_name}
    invalid_funds = sorted(normalized_requested - known_funds)
    if invalid_funds:
        raise ValueError(f"Unknown fund selection: {', '.join(invalid_funds)}")

    selected_funds = [fund_name for fund_name in context.fund_names if fund_name in normalized_requested]
    if len(selected_funds) < 3:
        raise ValueError("Please select at least 3 funds to compute the no-short and bounded-short frontiers.")
    return selected_funds


def _frontier_upper_target(
    policy_name: str,
    gmvp_row: Dict[str, Any],
    optimal_row: Dict[str, Any],
    mean_returns: pd.Series,
) -> float:
    start_return = float(gmvp_row["expected_return"])
    optimal_return = float(optimal_row["expected_return"])
    max_asset_return = float(mean_returns.max())

    if policy_name == "bounded_short":
        candidates = [start_return, optimal_return * 1.06, max_asset_return * 1.04]
    else:
        candidates = [start_return, optimal_return * 1.03, max_asset_return * 1.01]

    return max(candidates)


def _generate_frontier_with_fallback(
    policy_name: str,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    gmvp_row: Dict[str, Any],
    optimal_row: Dict[str, Any],
) -> pd.DataFrame:
    try:
        return pe.generate_frontier(policy_name, mean_returns, cov_matrix, annual_fee_pct, returns_df)
    except ValueError:
        upper_target_return = _frontier_upper_target(policy_name, gmvp_row, optimal_row, mean_returns)
        return pe.generate_frontier(
            policy_name,
            mean_returns,
            cov_matrix,
            annual_fee_pct,
            returns_df,
            upper_target_return=upper_target_return,
        )


def _solve_optimal_with_fallback(
    policy_name: str,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    annual_fee_pct: pd.Series,
    returns_df: pd.DataFrame,
    risk_aversion: float,
    fallback_row: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return pe.solve_optimal_portfolio(
            policy_name,
            mean_returns,
            cov_matrix,
            annual_fee_pct,
            returns_df,
            A=risk_aversion,
        )
    except ValueError:
        return fallback_row


@lru_cache(maxsize=1)
def get_runtime_context() -> RuntimeContext:
    metadata = pe.load_fund_metadata(str(FUND_META_PATH))
    fund_names = metadata["fund_name"].tolist()
    sleeve_lookup = metadata.set_index("fund_name")["sleeve"].to_dict()
    price_df, _ = pe.load_price_data(str(PRICES_PATH), fund_names)
    returns_df = pe.compute_returns(price_df)
    annual_fee_pct = metadata.set_index("fund_name")["annual_fee_pct"].reindex(fund_names)
    mean_returns = pe.annualized_mean_returns(returns_df)
    raw_covariance = pe.annualized_covariance(returns_df)
    covariance_matrix, _, _ = pe.covariance_psd_diagnostics(raw_covariance)
    calibration_grid = pe.build_a_calibration_grid(mean_returns, covariance_matrix, annual_fee_pct, returns_df)
    display_labels = [pe.SLEEVE_SHORT_LABELS[sleeve_lookup[fund_name]] for fund_name in fund_names]
    return RuntimeContext(
        metadata=metadata,
        fund_names=fund_names,
        price_df=price_df,
        returns_df=returns_df,
        annual_fee_pct=annual_fee_pct,
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
        calibration_grid=calibration_grid,
        display_labels=display_labels,
        sleeve_lookup=sleeve_lookup,
    )


def build_part1_payload(selected_funds: Optional[List[str]] = None) -> Dict[str, Any]:
    context = get_runtime_context()
    selected_fund_names = _resolve_selected_funds(context, selected_funds)
    summary = _load_json("analysis_summary.json")
    risk_aversion = float(summary["investor_profile"]["risk_aversion_A"])
    metadata = (
        context.metadata.set_index("fund_name").loc[selected_fund_names].reset_index().copy()
    )
    returns_df = context.returns_df.loc[:, selected_fund_names].copy()
    annual_fee_pct = metadata.set_index("fund_name")["annual_fee_pct"].reindex(selected_fund_names)
    mean_returns = pe.annualized_mean_returns(returns_df)
    raw_covariance = pe.annualized_covariance(returns_df)
    covariance_matrix, _, _ = pe.covariance_psd_diagnostics(raw_covariance)
    asset_points = pe.asset_points(mean_returns, covariance_matrix, metadata)
    no_short_gmvp = pe.solve_gmvp("no_short", mean_returns, covariance_matrix, annual_fee_pct, returns_df, A=risk_aversion)
    short_sales_gmvp = pe.solve_gmvp(
        "bounded_short",
        mean_returns,
        covariance_matrix,
        annual_fee_pct,
        returns_df,
        A=risk_aversion,
    )
    no_short_optimal = _solve_optimal_with_fallback(
        "no_short",
        mean_returns,
        covariance_matrix,
        annual_fee_pct,
        returns_df,
        risk_aversion,
        no_short_gmvp,
    )
    short_sales_optimal = _solve_optimal_with_fallback(
        "bounded_short",
        mean_returns,
        covariance_matrix,
        annual_fee_pct,
        returns_df,
        risk_aversion,
        short_sales_gmvp,
    )
    no_short_frontier = _generate_frontier_with_fallback(
        "no_short",
        mean_returns,
        covariance_matrix,
        annual_fee_pct,
        returns_df,
        no_short_gmvp,
        no_short_optimal,
    )
    short_sales_frontier = _generate_frontier_with_fallback(
        "bounded_short",
        mean_returns,
        covariance_matrix,
        annual_fee_pct,
        returns_df,
        short_sales_gmvp,
        short_sales_optimal,
    )
    benchmark = pd.DataFrame(
        [
            pe.equal_weight_portfolio(mean_returns, covariance_matrix, annual_fee_pct, returns_df, A=risk_aversion),
            no_short_gmvp,
            short_sales_gmvp,
            no_short_optimal,
            short_sales_optimal,
        ]
    )
    funds = metadata.loc[
        :,
        [
            "fund_name",
            "sleeve",
            "fund_house",
            "share_class_currency",
            "annual_fee_pct",
            "selection_reason",
        ],
    ].copy()
    label_map = dict(zip(asset_points["fund_name"], asset_points["short_label"]))
    funds["short_label"] = funds["fund_name"].map(label_map)
    return_series_lookup = {}
    for fund_name in selected_fund_names:
        series = context.returns_df[fund_name].dropna()
        return_series_lookup[fund_name] = [
            {
                "date": str(index.date()),
                "return": float(value),
            }
            for index, value in series.items()
        ]
    funds["return_series"] = funds["fund_name"].map(return_series_lookup)

    return {
        "summary": {
            "recommendedPolicy": summary["recommended_policy"],
            "dateStart": summary["date_start"],
            "dateEnd": summary["date_end"],
            "observationCount": summary["observation_count"],
            "investorType": summary["investor_profile"]["investor_type"],
            "riskAversionA": summary["investor_profile"]["risk_aversion_A"],
        },
        "availableFunds": _available_fund_options(context),
        "selectedFundNames": selected_fund_names,
        "funds": _df_records(funds),
        "assetPoints": _df_records(asset_points),
        "noShortFrontier": {
            "title": "Efficient Frontier (No Short Sales)",
            "frontier": _df_records(no_short_frontier.loc[:, ["expected_return", "volatility"]]),
            "gmvp": _to_builtin(no_short_gmvp),
            "gmvpWeights": _to_builtin(
                _portfolio_weights(no_short_gmvp, metadata, selected_fund_names, context.sleeve_lookup)
            ),
            "optimal": _to_builtin(no_short_optimal),
        },
        "shortSalesFrontier": {
            "title": "Efficient Frontier (Bounded Short Sales)",
            "frontier": _df_records(short_sales_frontier.loc[:, ["expected_return", "volatility"]]),
            "gmvp": _to_builtin(short_sales_gmvp),
            "gmvpWeights": _to_builtin(
                _portfolio_weights(short_sales_gmvp, metadata, selected_fund_names, context.sleeve_lookup)
            ),
            "optimal": _to_builtin(short_sales_optimal),
        },
        "benchmark": _df_records(
            benchmark.loc[
                :,
                [
                    "portfolio_name",
                    "expected_return",
                    "volatility",
                    "utility",
                    "max_drawdown",
                ],
            ]
        ),
    }


def build_recommendation_payload(answers: Dict[str, str]) -> Dict[str, Any]:
    context = get_runtime_context()
    questionnaire_df, construct_summary_df, questionnaire_profile = pe.score_questionnaire(answers)
    risk_aversion = float(questionnaire_profile["risk_aversion_A"])
    recommended = pe.solve_optimal_portfolio(
        "no_short",
        context.mean_returns,
        context.covariance_matrix,
        context.annual_fee_pct,
        context.returns_df,
        A=risk_aversion,
    )
    investor_profile = pe.direct_risk_aversion_summary(questionnaire_profile, recommended)

    equal_weight = pe.equal_weight_portfolio(
        context.mean_returns,
        context.covariance_matrix,
        context.annual_fee_pct,
        context.returns_df,
        A=risk_aversion,
    )
    gmvp_no_short = pe.solve_gmvp(
        "no_short",
        context.mean_returns,
        context.covariance_matrix,
        context.annual_fee_pct,
        context.returns_df,
        A=risk_aversion,
    )
    comparison_rows = [equal_weight, gmvp_no_short, recommended]
    bounded_short: Optional[Dict[str, Any]] = None
    try:
        bounded_short = pe.solve_optimal_portfolio(
            "bounded_short",
            context.mean_returns,
            context.covariance_matrix,
            context.annual_fee_pct,
            context.returns_df,
            A=risk_aversion,
        )
        comparison_rows.append(bounded_short)
    except ValueError:
        bounded_short = None

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.loc[
        :,
        [
            "portfolio_name",
            "expected_return",
            "volatility",
            "utility",
            "max_drawdown",
            "gross_exposure",
            "client_ready",
        ],
    ]

    weights = _portfolio_weights(recommended, context.metadata, context.fund_names, context.sleeve_lookup)
    bounded_short_weights = (
        _portfolio_weights(bounded_short, context.metadata, context.fund_names, context.sleeve_lookup)
        if bounded_short is not None
        else None
    )
    performance_series = _portfolio_performance_series(
        context.returns_df,
        context.price_df,
        context.fund_names,
        equal_weight,
        gmvp_no_short,
        recommended,
        bounded_short,
    )

    return {
        "questionRows": _df_records(questionnaire_df),
        "constructSummary": _df_records(construct_summary_df),
        "investorProfile": _to_builtin(investor_profile),
        "recommendation": {
            "portfolio": _to_builtin(recommended),
            "weights": _to_builtin(weights),
            "boundedShortPortfolio": _to_builtin(bounded_short),
            "boundedShortWeights": _to_builtin(bounded_short_weights),
            "comparison": _df_records(comparison_df),
            "performanceSeries": _to_builtin(performance_series),
            "topLongPositions": recommended["top_long_positions"],
            "topShortPositions": recommended["top_short_positions"],
        },
    }


def bootstrap_advisor_payload() -> Dict[str, Any]:
    return {
        "questionnaire": _questionnaire_schema(),
    }


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "BMD5302Platform/1.0"

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._set_headers(status_code=status_code)
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_OPTIONS(self) -> None:
        self._set_headers(status_code=204)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                self._send_json({"status": "ok"})
                return
            if path == "/api/part1":
                query = parse_qs(parsed.query)
                self._send_json(build_part1_payload(query.get("fund")))
                return
            if path == "/api/advisor/bootstrap":
                self._send_json(bootstrap_advisor_payload())
                return
            self._send_json({"error": f"Unknown endpoint: {path}"}, status_code=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status_code=500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/advisor/recommendation":
                payload = self._read_json_body()
                answers = payload.get("answers")
                if not isinstance(answers, dict):
                    self._send_json({"error": "Request body must include an 'answers' object."}, status_code=400)
                    return
                self._send_json(build_recommendation_payload(answers))
                return
            self._send_json({"error": f"Unknown endpoint: {path}"}, status_code=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status_code=500)


def run_server() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ApiHandler)
    print(f"Platform API running at http://{HOST}:{PORT}")
    print("Available endpoints: /api/health, /api/part1, /api/advisor/bootstrap, /api/advisor/recommendation")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
