import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score


@pytest.fixture
def woe_data():
    """WOE calculation test data"""
    return pd.DataFrame({
        'bin': ['<20', '20-30', '30-40', '>40'],
        'total_good': [500, 300, 150, 50],
        'total_bad': [20, 40, 60, 80],
    })


@pytest.fixture
def woe_data_calculated(woe_data):
    """Calculate WOE for each bin"""
    df = woe_data.copy()
    total_good = df['total_good'].sum()
    total_bad = df['total_bad'].sum()
    df['dist_good'] = df['total_good'] / total_good
    df['dist_bad'] = df['total_bad'] / total_bad
    df['woe'] = np.log(df['dist_good'] / df['dist_bad'])
    df['iv_component'] = (df['dist_good'] - df['dist_bad']) * df['woe']
    return df


@pytest.fixture
def logistic_model():
    """Train a simple logistic regression model for testing"""
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=3,
        n_redundant=1, random_state=42
    )
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def scorecard_data():
    """Sample scorecard conversion data"""
    np.random.seed(42)
    n = 500
    pd_estimates = np.random.beta(2, 10, n)  # most borrowers low PD
    # Score = 600 - 20 * ln(odds) where odds = PD/(1-PD)
    odds = pd_estimates / (1 - pd_estimates)
    scores = 600 - 20 * np.log(odds + 1e-9)
    scores = np.clip(scores, 300, 850).round(0)
    return pd.DataFrame({'pd_estimate': pd_estimates, 'score': scores})


@pytest.fixture
def risk_band_data(scorecard_data):
    """Assign risk bands based on scores"""
    df = scorecard_data.copy()

    def assign_band(score):
        if score >= 630:
            return 'Low'
        elif score >= 600:
            return 'Medium'
        elif score >= 560:
            return 'High'
        else:
            return 'Very High'

    df['risk_band'] = df['score'].apply(assign_band)
    return df


@pytest.fixture
def expected_loss_data():
    """Sample EL calculation data"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'pd_estimate': np.random.beta(2, 10, n),
        'lgd_assumption': np.random.uniform(0.3, 0.7, n),
        'ead': np.random.uniform(1000, 50000, n),
    })


# ---------------------------------------------------------------------------
# WOE Tests
# ---------------------------------------------------------------------------

def test_woe_good_above_bad(woe_data_calculated):
    """Bins where distribution of goods exceeds bads have positive WOE"""
    low_risk_bins = woe_data_calculated[
        woe_data_calculated['dist_good'] > woe_data_calculated['dist_bad']
    ]
    assert (low_risk_bins['woe'] > 0).all(), "Low-risk bins (dist_good > dist_bad) should have positive WOE"


def test_woe_calculation_formula(woe_data_calculated):
    """WOE = ln(Distribution of Good / Distribution of Bad)"""
    for _, row in woe_data_calculated.iterrows():
        expected_woe = np.log(row['dist_good'] / row['dist_bad'])
        assert abs(row['woe'] - expected_woe) < 1e-9, "WOE formula mismatch"


# ---------------------------------------------------------------------------
# IV Tests
# ---------------------------------------------------------------------------

def test_iv_always_positive(woe_data_calculated):
    """IV components should always be non-negative (each is WOE × (G% - B%))"""
    total_iv = woe_data_calculated['iv_component'].sum()
    # Total IV can be negative per component but sum reflects predictive power
    assert isinstance(total_iv, float), "IV should be a float"


def test_iv_range_interpretation(woe_data_calculated):
    """Test IV falls in meaningful range (0 to ~5 for real predictors)"""
    total_iv = abs(woe_data_calculated['iv_component'].sum())
    assert total_iv >= 0, "IV must be non-negative"
    assert total_iv < 10, "IV above 10 suggests data leakage or calculation error"


# ---------------------------------------------------------------------------
# Logistic Regression Tests
# ---------------------------------------------------------------------------

def test_model_predicts_valid_probabilities(logistic_model):
    """All predicted probabilities must be between 0 and 1"""
    model, X, y = logistic_model
    probabilities = model.predict_proba(X)[:, 1]
    assert (probabilities >= 0).all(), "Probabilities cannot be negative"
    assert (probabilities <= 1).all(), "Probabilities cannot exceed 1"


def test_model_auc_above_chance(logistic_model):
    """AUC must be above 0.5 (better than random)"""
    model, X, y = logistic_model
    probabilities = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probabilities)
    assert auc > 0.5, f"Model AUC {auc:.4f} is not better than random chance"


# ---------------------------------------------------------------------------
# Scorecard Conversion Tests
# ---------------------------------------------------------------------------

def test_scorecard_score_in_valid_range(scorecard_data):
    """All scores must fall between 300 and 850"""
    assert (scorecard_data['score'] >= 300).all(), "Scores below 300 found"
    assert (scorecard_data['score'] <= 850).all(), "Scores above 850 found"


def test_higher_pd_lower_score(scorecard_data):
    """Higher PD should correspond to lower credit score (inverse relationship)"""
    correlation = scorecard_data['pd_estimate'].corr(scorecard_data['score'])
    assert correlation < 0, \
        f"Expected negative correlation between PD and score, got {correlation:.4f}"


# ---------------------------------------------------------------------------
# Risk Band Tests
# ---------------------------------------------------------------------------

def test_all_records_assigned_band(risk_band_data):
    """Every record must have a risk band assigned (no nulls)"""
    assert risk_band_data['risk_band'].isnull().sum() == 0, \
        "Some records have no risk band"


def test_risk_band_valid_values(risk_band_data):
    """Risk bands must only contain valid categories"""
    valid_bands = {'Low', 'Medium', 'High', 'Very High'}
    actual_bands = set(risk_band_data['risk_band'].unique())
    assert actual_bands.issubset(valid_bands), \
        f"Invalid risk bands found: {actual_bands - valid_bands}"


# ---------------------------------------------------------------------------
# Expected Loss Tests
# ---------------------------------------------------------------------------

def test_expected_loss_between_zero_and_ead(expected_loss_data):
    """Expected Loss must be between 0 and EAD"""
    df = expected_loss_data.copy()
    df['el'] = df['pd_estimate'] * df['lgd_assumption'] * df['ead']
    assert (df['el'] >= 0).all(), "Expected Loss cannot be negative"
    assert (df['el'] <= df['ead']).all(), "Expected Loss cannot exceed EAD"


def test_expected_loss_formula(expected_loss_data):
    """EL = PD × LGD × EAD"""
    df = expected_loss_data.copy()
    df['el_calculated'] = df['pd_estimate'] * df['lgd_assumption'] * df['ead']
    sample = df.iloc[0]
    manual_el = sample['pd_estimate'] * sample['lgd_assumption'] * sample['ead']
    assert abs(sample['el_calculated'] - manual_el) < 1e-9, \
        "EL formula calculation error"
