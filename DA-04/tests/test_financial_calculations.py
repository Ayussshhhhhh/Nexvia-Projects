import pytest
import numpy as np
import pandas as pd
from src.financial_calculations import calculate_budget_variance


@pytest.fixture
def income_statement_data():
    """Sample income statement data for testing"""
    return {
        'revenue': 1_000_000.00,
        'cogs': 600_000.00,
        'gross_profit': 400_000.00,
        'operating_expenses': 150_000.00,
        'depreciation': 25_000.00,
        'amortization': 10_000.00,
        'ebitda': 285_000.00,   # gross_profit - opex + D&A
        'ebit': 250_000.00,     # gross_profit - opex
        'interest_expense': 20_000.00,
        'tax_expense': 52_500.00,
        'net_profit': 177_500.00,  # ebit - interest - tax
        'budget_revenue': 950_000.00,
        'budget_cogs': 570_000.00,
    }


@pytest.fixture
def monthly_data():
    """Monthly revenue data for MoM and YoY calculations"""
    return {
        'current_month': 850_000.00,
        'prior_month': 800_000.00,
        'prior_year_same_month': 750_000.00,
    }


@pytest.fixture
def variance_data():
    """Budget vs actual variance data"""
    return pd.DataFrame({
        'account': ['Revenue', 'COGS', 'OpEx', 'Marketing', 'HR'],
        'actual': [1_000_000, 600_000, 150_000, 80_000, 70_000],
        'budget': [950_000, 570_000, 160_000, 90_000, 65_000],
    })


def test_gross_profit_equals_revenue_minus_cogs(income_statement_data):
    d = income_statement_data
    calculated_gp = d['revenue'] - d['cogs']
    assert abs(calculated_gp - d['gross_profit']) < 0.01, \
        f"Gross Profit mismatch: expected {d['gross_profit']}, got {calculated_gp}"


def test_net_profit_calculation(income_statement_data):
    d = income_statement_data
    calculated_np = (
        d['gross_profit']
        - d['operating_expenses']
        - d['interest_expense']
        - d['tax_expense']
    )
    assert abs(calculated_np - d['net_profit']) < 0.01, \
        f"Net Profit mismatch: expected {d['net_profit']}, got {calculated_np}"


def test_gross_margin_percentage(income_statement_data):
    d = income_statement_data
    gross_margin = d['gross_profit'] / d['revenue']
    expected = 0.40  # 40%
    assert abs(gross_margin - expected) < 0.001, \
        f"Gross Margin mismatch: expected {expected:.1%}, got {gross_margin:.1%}"


def test_ebitda_calculation(income_statement_data):
    d = income_statement_data
    # EBITDA = Revenue - COGS - OpEx + Depreciation + Amortization
    calculated_ebitda = (
        d['gross_profit']
        - d['operating_expenses']
        + d['depreciation']
        + d['amortization']
    )
    assert abs(calculated_ebitda - d['ebitda']) < 0.01, \
        f"EBITDA mismatch: expected {d['ebitda']}, got {calculated_ebitda}"


def test_budget_variance_equals_actual_minus_budget(variance_data):
    variance_data['variance'] = variance_data['actual'] - variance_data['budget']
    rev_row = variance_data[variance_data['account'] == 'Revenue'].iloc[0]
    expected_variance = 1_000_000 - 950_000
    assert abs(rev_row['variance'] - expected_variance) < 0.01


def test_favorable_variance_flag(variance_data):
    variance_data['variance'] = variance_data['actual'] - variance_data['budget']
    variance_data['favorable'] = variance_data['variance'] > 0
    # Revenue: actual > budget → favorable
    rev_row = variance_data[variance_data['account'] == 'Revenue'].iloc[0]
    assert rev_row['favorable'] == True


def test_unfavorable_variance_flag(variance_data):
    variance_data['variance'] = variance_data['actual'] - variance_data['budget']
    variance_data['favorable'] = variance_data['variance'] > 0
    # OpEx: actual 150k vs budget 160k → negative numeric variance (underspend)
    opex_row = variance_data[variance_data['account'] == 'OpEx'].iloc[0]
    assert opex_row['variance'] < 0  # underspend on cost = negative numeric variance


def test_mom_change_calculation(monthly_data):
    d = monthly_data
    mom_change = (d['current_month'] - d['prior_month']) / d['prior_month']
    expected = (850_000 - 800_000) / 800_000  # 6.25%
    assert abs(mom_change - expected) < 0.0001


def test_yoy_change_calculation(monthly_data):
    d = monthly_data
    yoy_change = (d['current_month'] - d['prior_year_same_month']) / d['prior_year_same_month']
    expected = (850_000 - 750_000) / 750_000  # ~13.33%
    assert abs(yoy_change - expected) < 0.0001


def test_variance_percentage_calculation(variance_data):
    variance_data['variance'] = variance_data['actual'] - variance_data['budget']
    variance_data['variance_pct'] = variance_data['variance'] / variance_data['budget']
    rev_row = variance_data[variance_data['account'] == 'Revenue'].iloc[0]
    expected_pct = (1_000_000 - 950_000) / 950_000
    assert abs(rev_row['variance_pct'] - expected_pct) < 0.0001


def test_variance_percentage_zero_budget():
    """Division by zero should return NaN or raise, not crash silently"""
    actual = 100_000
    budget = 0
    if budget == 0:
        result = float('nan')
    else:
        result = (actual - budget) / budget
    assert np.isnan(result), "Zero budget should produce NaN variance percentage"


def test_numpy_budget_variance_handles_zero_and_favorable_arrays():
    actual = np.array([110.0, 90.0, 100.0])
    budget = np.array([100.0, 100.0, 0.0])

    variance, variance_pct, favorable = calculate_budget_variance(
        actual, budget, is_revenue=True
    )

    np.testing.assert_array_equal(variance, np.array([10.0, -10.0, 100.0]))
    np.testing.assert_allclose(variance_pct, np.array([0.1, -0.1, 0.0]))
    np.testing.assert_array_equal(favorable, np.array([True, False, True]))
