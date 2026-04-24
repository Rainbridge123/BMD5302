#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


BASE_URL = "https://secure.fundsupermart.com"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FX_USD_TO_SGD = 1.2780
FX_EUR_TO_SGD = 1.4930
DOWNLOAD_DATE = "2026-04-20"
SELECTION_RULE = (
    "Select one real FSMOne SGD or SGD-hedged fund per sleeve with at least 60 monthly observations, "
    "then prefer broader mandates, longer usable history, larger fund size, and lower annual fee."
)
NOTES_TEXT = (
    "inception_date stores the first publicly retrievable monthly chart observation from the FSMOne pre-login chart endpoint. "
    "FSMOne's public pre-login APIs did not expose a reliable legal share-class inception date or ISIN for every selected fund."
)


@dataclass(frozen=True)
class SelectedFund:
    sleeve: str
    code: str
    fund_name: str
    selector_filters: Dict[str, Optional[str]]
    share_class_currency: str
    share_class_type: str
    selection_reason: str
    rejected_alternatives: str
    tie_break_reason: str
    evidence_reference: str


SELECTED_FUNDS: List[SelectedFund] = [
    SelectedFund(
        sleeve="money market/cash",
        code="370209",
        fund_name="LionGlobal SGD Money Market A Acc SGD",
        selector_filters={"main": "DG", "spec": None, "area": None},
        share_class_currency="SGD",
        share_class_type="Accumulation",
        selection_reason="Chosen as the SGD cash sleeve because it is a real FSMOne money-market accumulation class with a full 10-year monthly history and direct cash-equivalent exposure.",
        rejected_alternatives="Fullerton SGD Cash Fund A SGD; Maybank Money Market A Acc SGD; United SGD Money Market A1 SGD",
        tie_break_reason="Preferred the explicit accumulation money-market share class with full 120-month coverage over cash-fund alternatives that were either not explicitly accumulation or had shorter public history.",
        evidence_reference="FSMOne selector filter main=DG, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="short-duration bond",
        code="UOB132",
        fund_name="United SGD Fund B Acc SGD",
        selector_filters={"main": "FI", "spec": "SDMW", "area": "INT"},
        share_class_currency="SGD",
        share_class_type="Accumulation",
        selection_reason="Chosen as the short-duration bond sleeve because it provides a long-standing SGD short-duration bond track record with low annual fee and full 10-year monthly coverage.",
        rejected_alternatives="LionGlobal Short Duration Bond A Acc SGD; United SGD Fund Cl A Acc SGD; HGIF - Ultra Short Duration Bond PM3H SGD",
        tie_break_reason="Selected for the combination of full 120-month history, low fee, and clean SGD accumulation structure; other candidates either had shorter public history or weaker fee efficiency.",
        evidence_reference="FSMOne selector filter main=FI, spec=SDMW, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="global investment-grade bond",
        code="UOB099",
        fund_name="United High Grade Corporate Bond A Acc SGD",
        selector_filters={"main": "FI", "spec": "IG", "area": "INT"},
        share_class_currency="SGD",
        share_class_type="Accumulation",
        selection_reason="Chosen as the investment-grade bond sleeve because it is an accumulation share class with full 10-year monthly history and lower annual fee than the comparable global credit alternatives surfaced by FSMOne.",
        rejected_alternatives="United High Grade Corporate Bond A Acc SGD-H; PIMCO Global Investment Grade Credit Fund Cl E Inc SGD-H; Natixis Loomis Sayles Global Credit RD SGD",
        tie_break_reason="Preferred the lowest-fee accumulation class with 120 monthly observations over distribution classes and shorter-history hedged variants.",
        evidence_reference="FSMOne selector filter main=FI, spec=IG, area=INT, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="higher-yield bond",
        code="ACM140",
        fund_name="AB FCP I Global High Yield A2 SGD-H",
        selector_filters={"main": "FI", "spec": "CB", "area": "INT"},
        share_class_currency="SGD-HEDGED",
        share_class_type="Accumulation",
        selection_reason="Chosen as the higher-yield bond sleeve because it is a hedged accumulation class with a full 10-year monthly history and one of the largest fund sizes in the official FSMOne global high-yield candidate set.",
        rejected_alternatives="JPMorgan Investment Funds - Global High Yield Bond A (mth) SGD; Aviva Investors - Global High Yield Bond Ah SGD; Schroder ISF Global High Yield A Dis SGD-H",
        tie_break_reason="Preferred the long-history SGD-hedged accumulation class over monthly-distribution alternatives that would complicate price-only return measurement in a Markowitz assignment.",
        evidence_reference="FSMOne selector filter main=FI, spec=CB, area=INT, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="singapore equity",
        code="370007",
        fund_name="Amova Singapore Equity SGD (formerly Nikko AM)",
        selector_filters={"main": "EG", "spec": None, "area": "FES"},
        share_class_currency="SGD",
        share_class_type="Non-distributing / price-only share class",
        selection_reason="Chosen as the Singapore equity sleeve because it is a broad Singapore equity fund with full 10-year monthly history and substantially lower annual fee than the larger local-equity peers on FSMOne.",
        rejected_alternatives="Schroder Singapore Trust A Acc SGD; LionGlobal Singapore Trust Acc SGD; United Singapore Growth Fund SGD",
        tie_break_reason="Selected for the best fee-to-history trade-off among broad Singapore-equity candidates with a full 120-month chart history.",
        evidence_reference="FSMOne selector filter main=EG, area=FES, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="u.s. equity",
        code="370283",
        fund_name="Infinity US 500 Stock Index SGD",
        selector_filters={"main": "EG", "spec": None, "area": "NAU"},
        share_class_currency="SGD",
        share_class_type="Index / non-distributing share class",
        selection_reason="Chosen as the U.S. equity sleeve because it offers broad S&P 500-style exposure, a full 10-year monthly history, and far lower annual fee than the actively managed U.S. equity alternatives on FSMOne.",
        rejected_alternatives="Fidelity America A-SGD; FTIF - Franklin US Opportunities A Acc SGD; AB SICAV I American Growth A SGD",
        tie_break_reason="Preferred the broad low-fee index-like U.S. exposure over narrower growth, value, or style-tilted active funds.",
        evidence_reference="FSMOne selector filter main=EG, area=NAU, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="global developed equity",
        code="FI3081",
        fund_name="Fidelity World A-ACC-SGD",
        selector_filters={"main": "EG", "spec": None, "area": "INT"},
        share_class_currency="SGD",
        share_class_type="Accumulation",
        selection_reason="Chosen as the global developed-equity sleeve because it is a broad world-equity accumulation class with a full 10-year monthly history and clearer market-wide exposure than the thematic global alternatives on FSMOne.",
        rejected_alternatives="Ninety One Global Strategy Fund - Global Franchise A Acc SGD-H; GS Global CORE Equity Portfolio Acc Close SGD; Allianz Best Styles Global Equity Cl ET Acc H2-SGD",
        tie_break_reason="Preferred the broad world-equity accumulation mandate with full 120-month coverage over more concentrated quality, low-volatility, or partially thematic global strategies.",
        evidence_reference="FSMOne selector filter main=EG, area=INT, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="asia ex-japan equity",
        code="SCD116",
        fund_name="Schroder ISF Asian Opportunities A Acc SGD",
        selector_filters={"main": "EG", "spec": None, "area": "FE2"},
        share_class_currency="SGD",
        share_class_type="Accumulation",
        selection_reason="Chosen as the Asia ex-Japan sleeve because it is a broad Asian opportunities accumulation class with full 10-year monthly history and large fund size within the official FSMOne Asia ex-Japan candidate set.",
        rejected_alternatives="FSSA Asian Growth A Acc SGD; FTIF - Templeton Asian Growth A Acc SGD; United Asia A Acc SGD",
        tie_break_reason="Selected for its combination of long history, accumulation structure, and larger official fund size than most competing Asia ex-Japan options.",
        evidence_reference="FSMOne selector filter main=EG, area=FE2, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="global REIT",
        code="FST005",
        fund_name="First Sentier Global Property Securities A Acc SGD",
        selector_filters={"main": "EG", "spec": "RE", "area": "INT"},
        share_class_currency="SGD",
        share_class_type="Accumulation",
        selection_reason="Chosen as the global REIT / property sleeve because it is an explicit accumulation class with full 10-year monthly history and direct listed-property exposure on FSMOne.",
        rejected_alternatives="Janus Henderson Horizon Global Property Equities A3 SGD; Amova Global Property Securities SGD; HGIF - Global Real Estate Equity AM3O SGD",
        tie_break_reason="Preferred the explicit accumulation property-securities class with full 120-month coverage over alternatives with less explicit distribution treatment or shorter usable history.",
        evidence_reference="FSMOne selector filter main=EG, spec=RE, area=INT, currency=SGD, cash-eligible; chart period=10y downloaded 2026-04-20.",
    ),
    SelectedFund(
        sleeve="gold/real-asset hedge",
        code="BGF006",
        fund_name="Blackrock World Gold Fund A2 SGD-H",
        selector_filters={"main": "EG", "spec": None, "area": "INT"},
        share_class_currency="SGD-HEDGED",
        share_class_type="Accumulation",
        selection_reason="Chosen as the gold / real-asset hedge sleeve because it provides a long-history SGD-hedged gold-equity exposure and the largest official fund size among the gold-focused candidates surfaced through FSMOne.",
        rejected_alternatives="United Gold and General A Acc SGD; FTIF - Franklin Gold and Precious Metals A (acc) SGD; Schroder ISF Global Gold A Acc SGD-H",
        tie_break_reason="Preferred the largest long-history gold-focused share class with SGD hedging over smaller or less scalable alternatives.",
        evidence_reference="FSMOne selector filter main=EG, area=INT, currency=SGD, cash-eligible; gold-focused candidate isolated from official selector results; chart period=10y downloaded 2026-04-20.",
    ),
]


def start_session() -> Tuple[requests.Session, Dict[str, str]]:
    session = requests.Session()
    session.get(f"{BASE_URL}/fsm/home", timeout=30)
    xsrf_token = session.cookies.get("XSRF-TOKEN")
    if not xsrf_token:
        raise RuntimeError("Failed to obtain FSMOne XSRF token.")
    headers = {
        "X-XSRF-TOKEN": xsrf_token,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/fsm/home",
    }
    return session, headers


def selector_payload(main: Optional[str], spec: Optional[str], area: Optional[str], currency: str) -> Dict:
    return {
        "tableViewSel": "P",
        "managercodeListDisplay": [],
        "fsmRiskRatingListDisplay": [],
        "cpfSrsScListDisplay": ["cash"],
        "shariahCompliantListDisplay": [],
        "sectormaincodeListDisplay": [main] if main else [],
        "sectorspeccodeListDisplay": [spec] if spec else [],
        "sectorareacodeListDisplay": [area] if area else [],
        "fundCurrencyCodeListDisplay": [currency],
        "fundNameListDisplay": [],
        "recommendedOnly": False,
        "etfEnabledOnly": False,
        "rspEnabledOnly": False,
        "dividendOptionExists": False,
        "eipOnly": False,
        "cashEnabled": True,
        "cpfapproved": False,
        "cpfissaApproved": False,
        "srsApproved": False,
        "managercodeList": [],
        "fsmRiskRatingList": [],
        "cpfSrsScList": ["cash"],
        "shariahCompliantList": [],
        "sectormaincodeList": [main] if main else [],
        "sectorspeccodeList": [spec] if spec else [],
        "sectorareacodeList": [area] if area else [],
        "fundCurrencyCodeList": [currency],
        "fundNameList": [],
        "tickerNumberList": [],
        "sedolnumberList": [],
    }


def query_selector(session: requests.Session, headers: Dict[str, str], fund: SelectedFund) -> Dict:
    payload = selector_payload(
        main=fund.selector_filters.get("main"),
        spec=fund.selector_filters.get("spec"),
        area=fund.selector_filters.get("area"),
        currency="SGD",
    )
    response = session.post(
        f"{BASE_URL}/fsm/rest/fund/get-fund-selector-table-info-with-multiple-list",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    response.raise_for_status()
    rows = response.json()["fundSelectorResultTableDisplayList"]
    for row in rows:
        if row["fundInfoSedolnumber"] == fund.code:
            return row
    raise KeyError(f"Selected code {fund.code} not found in selector results for sleeve {fund.sleeve}.")


def query_search(session: requests.Session, headers: Dict[str, str], fund_name: str) -> Dict:
    response = session.post(
        f"{BASE_URL}/fsm/rest/general-search/find-fund-info-by-search-term",
        params={"paramSearchTerm": fund_name},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    rows = response.json()
    for row in rows:
        if row.get("fundName") == fund_name:
            return row
    if rows:
        return rows[0]
    raise KeyError(f"No search results returned for {fund_name}.")


def fetch_chart_series(session: requests.Session, headers: Dict[str, str], code: str) -> pd.Series:
    info = session.post(
        f"{BASE_URL}/fsm/rest/chart/find-active-fund-info-by-sedolnumber",
        params={"paramSedolnumber": code},
        headers=headers,
        timeout=30,
    )
    info.raise_for_status()
    resolved = info.json()
    response = session.post(
        f"{BASE_URL}/fsm/rest/chart/get-bid-price-chart-data",
        params={"paramId": resolved["code"], "paramPeriod": "10y", "paramProduct": "UT"},
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    chart_data = response.json()
    if not chart_data:
        raise ValueError(f"No chart data returned for {code}.")

    series = pd.Series(
        data=[float(point[1]) for point in chart_data],
        index=pd.to_datetime([point[0] for point in chart_data], unit="ms", utc=True).tz_localize(None),
    )
    series.index = series.index.to_period("M").to_timestamp("M")
    if series.index.duplicated().any():
        series = series.groupby(level=0).last()
    return series.sort_index()


def convert_aum_to_sgd(amount: Optional[float], currency: Optional[str], fallback_currency: str) -> float:
    if amount in (None, 0):
        return 0.0
    currency = (currency or fallback_currency or "SGD").upper().replace("_", "-")
    if currency == "SGD":
        return float(amount)
    if currency == "USD":
        return float(amount) * FX_USD_TO_SGD
    if currency == "EUR":
        return float(amount) * FX_EUR_TO_SGD
    return float(amount)


def build_metadata_row(
    fund: SelectedFund,
    selector_row: Dict,
    search_row: Dict,
    series: pd.Series,
) -> Dict[str, object]:
    fund_size = search_row.get("fundsize") or selector_row.get("fundInfoFundsizeInCurrency") or selector_row.get("fundInfoFundsize") or 0
    fund_size_currency = search_row.get("fundsizeCurrency") or selector_row.get("fundInfoFundsizeCurrency") or fund.share_class_currency.replace("-HEDGED", "")
    aum_sgd_m = round(convert_aum_to_sgd(fund_size, fund_size_currency, fund.share_class_currency), 2)
    return {
        "fund_name": fund.fund_name,
        "sleeve": fund.sleeve,
        "fund_house": search_row.get("managername", "").strip(),
        "fsmone_code": fund.code,
        "isin": "",
        "share_class_currency": fund.share_class_currency,
        "share_class_type": fund.share_class_type,
        "inception_date": series.index.min().date().isoformat(),
        "aum_sgd_m": aum_sgd_m,
        "annual_fee_pct": float(selector_row.get("fundInfoAnnualcharge") or 0.0),
        "selection_status": "final",
        "selection_reason": fund.selection_reason,
        "selection_evidence": (
            "Official FSMOne selector result and 10-year chart series downloaded from the public pre-login endpoints on "
            f"{DOWNLOAD_DATE}."
        ),
        "rejected_alternatives": fund.rejected_alternatives,
        "tie_break_reason": fund.tie_break_reason,
        "evidence_reference": fund.evidence_reference,
        "selection_rule": SELECTION_RULE,
        "proxy_source": "",
        "notes": NOTES_TEXT,
    }


def main() -> None:
    session, headers = start_session()
    metadata_rows: List[Dict[str, object]] = []
    price_series_map: Dict[str, pd.Series] = {}

    for fund in SELECTED_FUNDS:
        selector_row = query_selector(session, headers, fund)
        search_row = query_search(session, headers, fund.fund_name)
        series = fetch_chart_series(session, headers, fund.code)
        metadata_rows.append(build_metadata_row(fund, selector_row, search_row, series))
        price_series_map[fund.fund_name] = series

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df = metadata_df[
        [
            "fund_name",
            "sleeve",
            "fund_house",
            "fsmone_code",
            "isin",
            "share_class_currency",
            "share_class_type",
            "inception_date",
            "aum_sgd_m",
            "annual_fee_pct",
            "selection_status",
            "selection_reason",
            "selection_evidence",
            "rejected_alternatives",
            "tie_break_reason",
            "evidence_reference",
            "selection_rule",
            "proxy_source",
            "notes",
        ]
    ]

    price_df = pd.concat(price_series_map, axis=1)
    price_df = price_df.loc[:, metadata_df["fund_name"].tolist()]
    price_df.index.name = "date"
    price_df = price_df.reset_index()
    price_df["date"] = price_df["date"].dt.date.astype(str)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATA_DIR / "fsmone_fund_universe.csv"
    prices_path = DATA_DIR / "fsmone_prices.csv"
    manifest_path = DATA_DIR / "fsmone_download_manifest.json"

    metadata_df.to_csv(metadata_path, index=False, quoting=csv.QUOTE_MINIMAL)
    price_df.to_csv(prices_path, index=False, quoting=csv.QUOTE_MINIMAL)
    manifest = {
        "download_date": DOWNLOAD_DATE,
        "source": "FSMOne / Fundsupermart public pre-login REST endpoints",
        "price_endpoint": "/fsm/rest/chart/get-bid-price-chart-data",
        "selector_endpoint": "/fsm/rest/fund/get-fund-selector-table-info-with-multiple-list",
        "search_endpoint": "/fsm/rest/general-search/find-fund-info-by-search-term",
        "usd_to_sgd_assumption": FX_USD_TO_SGD,
        "eur_to_sgd_assumption": FX_EUR_TO_SGD,
        "notes": NOTES_TEXT,
        "funds": metadata_df[["fund_name", "fsmone_code", "sleeve"]].to_dict(orient="records"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote prices to {prices_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
