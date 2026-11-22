"""
src/financial_calculations.py
==============================
Financial calculations module for the Financial Analytics and Risk Reporting System.

Covers:
- Monthly Profit & Loss (P&L) statement generation
- Balance Sheet calculations and accounting equation validation
- Budget vs. Actual variance analysis with favorable/unfavorable indicators
- Month-over-Month (MoM) and Year-over-Year (YoY) growth calculations
- Pareto analysis for identifying top variance drivers
- Rule-based commentary generation for finance and executive teams
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standard monthly Profit & Loss statement from general ledger/summary data.

    Expected input columns:
    - date (or year/month)
    - revenue
    - cogs
    - opex_salaries, opex_marketing, opex_technology, opex_depreciation, opex_other (or total_opex)
    - interest_expense
    - tax

    Returns:
        DataFrame with computed financial line items:
        revenue, cogs, gross_profit, gross_margin, total_opex, ebitda,
        ebit, interest_expense, pbt, tax, net_profit, net_margin
    """
    df = df.copy()

    # Gross Profit & Margin
    if "gross_profit" not in df.columns:
        df["gross_profit"] = df["revenue"] - df["cogs"]
    
    df["gross_margin"] = np.where(df["revenue"] > 0, df["gross_profit"] / df["revenue"], 0.0)

    # Operating Expenses
    opex_cols = [c for c in ["opex_salaries", "opex_marketing", "opex_technology", "opex_depreciation", "opex_other"] if c in df.columns]
    if "total_opex" not in df.columns:
        if opex_cols:
            df["total_opex"] = df[opex_cols].sum(axis=1)
        elif "operating_expenses" in df.columns:
            df["total_opex"] = df["operating_expenses"]
        else:
            df["total_opex"] = 0.0

    # EBITDA & EBIT
    depr_col = "opex_depreciation" if "opex_depreciation" in df.columns else ("depreciation" if "depreciation" in df.columns else None)
    depreciation = df[depr_col] if depr_col else 0.0

    # If total_opex includes depreciation:
    # Operating Profit / EBIT = Gross Profit - Total OpEx
    # EBITDA = EBIT + Depreciation
    df["ebit"] = df["gross_profit"] - df["total_opex"]
    df["ebitda"] = df["ebit"] + depreciation

    # Profit Before Tax (PBT) / EBT
    interest = df["interest_expense"] if "interest_expense" in df.columns else 0.0
    df["pbt"] = df["ebit"] - interest

    # Tax & Net Profit
    tax = df["tax"] if "tax" in df.columns else (df["tax_expense"] if "tax_expense" in df.columns else 0.0)
    df["net_profit"] = df["pbt"] - tax
    df["net_margin"] = np.where(df["revenue"] > 0, df["net_profit"] / df["revenue"], 0.0)

    return df


def calculate_balance_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and validate monthly Balance Sheet positions.

    Identities:
    Total Current Assets = Cash + Accounts Receivable + Inventory
    Total Assets = Total Current Assets + PP&E
    Total Current Liabilities = Accounts Payable + Short-Term Debt
    Total Liabilities = Total Current Liabilities + Long-Term Debt
    Total Liabilities & Equity = Total Liabilities + Equity
    Balance Check: Total Assets - Total Liabilities & Equity == 0
    """
    df = df.copy()

    df["total_current_assets"] = (
        df.get("cash", 0.0) +
        df.get("accounts_receivable", 0.0) +
        df.get("inventory", 0.0)
    )
    df["total_assets"] = df["total_current_assets"] + df.get("ppe", 0.0)

    df["total_current_liabilities"] = (
        df.get("accounts_payable", 0.0) +
        df.get("short_term_debt", 0.0)
    )
    df["total_liabilities"] = df["total_current_liabilities"] + df.get("long_term_debt", 0.0)
    df["total_liabilities_equity"] = df["total_liabilities"] + df.get("equity", 0.0)

    df["balance_check_variance"] = df["total_assets"] - df["total_liabilities_equity"]
    df["is_balanced"] = df["balance_check_variance"].abs() < 1e-2

    return df


def calculate_budget_variance(
    actual: float | pd.Series | np.ndarray,
    budget: float | pd.Series | np.ndarray,
    is_revenue: bool = True
) -> Tuple[Any, Any, Any]:
    """Calculate variance, variance percentage, and favorable/unfavorable indicator.

    Variance = Actual - Budget
    For revenue/profit: Actual > Budget is Favorable (True)
    For expense/cost: Actual < Budget is Favorable (True)
    """
    variance = actual - budget

    # Avoid zero division while preserving Series/NumPy result shapes.
    if isinstance(budget, pd.Series):
        var_pct = budget.copy().astype(float)
        var_pct[:] = np.where(
            budget.abs().to_numpy() > 1e-6,
            variance.to_numpy() / budget.to_numpy(),
            0.0,
        )
    elif isinstance(budget, np.ndarray):
        budget_arr = np.asarray(budget)
        variance_arr = np.asarray(variance)
        var_pct = np.divide(
            variance_arr,
            budget_arr,
            out=np.zeros_like(variance_arr, dtype=float),
            where=np.abs(budget_arr) > 1e-6,
        )
    else:
        var_pct = variance / budget if abs(budget) > 1e-6 else 0.0

    is_favorable = (variance >= 0) if is_revenue else (variance <= 0)

    return variance, var_pct, is_favorable


def calculate_mom_change(
    series: pd.Series,
    periods: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Month-over-Month change in absolute value and percentage."""
    abs_change = series.diff(periods=periods)
    pct_change = series.pct_change(periods=periods)
    return abs_change, pct_change


def calculate_yoy_change(
    series: pd.Series,
    periods: int = 12
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Year-over-Year change in absolute value and percentage."""
    abs_change = series.diff(periods=periods)
    pct_change = series.pct_change(periods=periods)
    return abs_change, pct_change


def pareto_analysis(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    top_n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """Perform Pareto analysis on variances or expenses.

    Calculates:
    - Absolute variance / spend
    - Share of total variance
    - Cumulative share
    """
    df_sorted = df.groupby(category_col)[value_col].sum().reset_index()
    df_sorted["abs_value"] = df_sorted[value_col].abs()
    df_sorted = df_sorted.sort_values(by="abs_value", ascending=ascending).reset_index(drop=True)

    total_val = df_sorted["abs_value"].sum()
    df_sorted["share_pct"] = np.where(total_val > 0, df_sorted["abs_value"] / total_val, 0.0)
    df_sorted["cumulative_pct"] = df_sorted["share_pct"].cumsum()

    return df_sorted.head(top_n)


def generate_variance_commentary(
    metric_name: str,
    actual: float,
    budget: float,
    variance: float,
    var_pct: float,
    is_favorable: bool,
    dimension_context: Optional[Dict[str, str]] = None
) -> str:
    """Generate professional, audit-ready rule-based commentary for financial variances.

    Examples:
    - Revenue shortfall driven by lower Personal Loans volume in West region.
    - Marketing expenditure overrun due to Q3 brand campaign acceleration.
    """
    context_str = ""
    if dimension_context:
        ctx_parts = [f"{k}: {v}" for k, v in dimension_context.items()]
        context_str = f" [{', '.join(ctx_parts)}]"

    status = "favorable" if is_favorable else "unfavorable"
    pct_str = f"{abs(var_pct) * 100:.1f}%"
    val_str = f"${abs(variance):,.0f}"

    if "revenue" in metric_name.lower():
        if is_favorable:
            driver = "driven by stronger origination volume and favorable product mix"
        else:
            driver = "impacted by lower application volume and tightening credit underwriting criteria"
    elif "cogs" in metric_name.lower() or "cost" in metric_name.lower():
        if is_favorable:
            driver = "benefiting from funding cost optimization and lower default charge-offs"
        else:
            driver = "attributable to elevated short-term borrowing costs and servicing fee pressures"
    elif "marketing" in metric_name.lower():
        if is_favorable:
            driver = "reflecting disciplined ad-spend pacing and improved digital acquisition CAC"
        else:
            driver = "accelerated customer acquisition initiatives and brand partner onboarding"
    elif "technology" in metric_name.lower():
        if is_favorable:
            driver = "due to contract renegotiations and cloud infrastructure optimization"
        else:
            driver = "timing of cloud migration milestones and enterprise data tooling licenses"
    elif "salaries" in metric_name.lower() or "payroll" in metric_name.lower():
        if is_favorable:
            driver = "driven by deferred hiring in non-revenue functions"
        else:
            driver = "reflecting specialized engineering hires and annual performance adjustments"
    else:
        if is_favorable:
            driver = "outperforming budgetary expectations through operational efficiencies"
        else:
            driver = "exceeding budgetary targets due to seasonal and macroeconomic factors"

    commentary = (
        f"{metric_name.title()}{context_str}: Actual ${actual:,.0f} vs Budget ${budget:,.0f} "
        f"({status.upper()} variance of {val_str} or {pct_str}), {driver}."
    )
    return commentary
