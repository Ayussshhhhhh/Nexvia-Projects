"""
src/credit_scorecard.py
========================
Credit risk scorecard development and validation module.

Implements:
- Logistic Regression model on Weight of Evidence (WOE) features
- Standard credit risk scorecard points scaling:
    Score = Offset + Factor * ln(odds)
    where odds = (1 - PD) / PD (Good / Bad odds)
    Factor = PDO / ln(2)
    Offset = Base_Score - Factor * ln(Base_Odds)
- Risk-band segmentation (Low, Medium, High, Very High)
- Expected Loss (EL) estimation: EL = PD * LGD * EAD
- Discrimination and calibration validation: ROC-AUC, Gini, Kolmogorov-Smirnov (KS)
- Visualizations: ROC curve, score distribution, risk band performance
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import matplotlib

# The analytics pipeline is a batch process and should never initialize a GUI
# backend (which can abort on headless or remote macOS sessions).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class CreditScorecard:
    """Standard Weight of Evidence (WOE) credit risk scorecard."""

    def __init__(
        self,
        base_score: float = 600.0,
        base_odds: float = 2.0,
        pdo: float = 30.0,
        lgd_assumption: float = 0.45,
        random_state: int = 42,
    ):
        """Initialize scorecard scaling parameters.

        Args:
            base_score: Target score at base odds (default 600)
            base_odds: Odds of Good:Bad at base score (default 2:1 for 33% portfolio default rate)
            pdo: Points to Double the Odds (default 30)
            lgd_assumption: Default Loss Given Default (default 45%)
            random_state: Random state for model fitting
        """
        self.base_score = float(base_score)
        self.base_odds = float(base_odds)
        self.pdo = float(pdo)
        self.lgd_assumption = float(lgd_assumption)
        self.random_state = random_state

        # Scorecard scaling constants
        self.factor = self.pdo / np.log(2.0)
        self.offset = self.base_score - (self.factor * np.log(self.base_odds))

        self.model: Optional[LogisticRegression] = None
        self.feature_names: List[str] = []
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

        # Cutoffs for risk bands based on score
        # Low Risk: >= 630, Medium Risk: 600-630, High Risk: 560-600, Very High Risk: < 560
        self.risk_bands_config = [
            {"band": "Very High Risk", "min_score": -np.inf, "max_score": 560.0, "color": "#D9534F"},
            {"band": "High Risk", "min_score": 560.0, "max_score": 600.0, "color": "#F0AD4E"},
            {"band": "Medium Risk", "min_score": 600.0, "max_score": 630.0, "color": "#5BC0DE"},
            {"band": "Low Risk", "min_score": 630.0, "max_score": np.inf, "color": "#5CB85C"},
        ]

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "CreditScorecard":
        """Train logistic regression model on WOE-transformed features.

        Target y must be binary: 1 = Bad/Default, 0 = Good.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_arr = X_train.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
            X_arr = X_train

        y_arr = np.asarray(y_train).ravel()

        self.model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.model.fit(X_arr, y_arr)

        self.coefficients_ = self.model.coef_[0]
        self.intercept_ = float(self.model.intercept_[0])

        logger.info(
            "CreditScorecard fitted successfully on %d samples with %d features.",
            X_arr.shape[0],
            len(self.feature_names),
        )
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict Probability of Default (PD).

        Returns:
            1D array of estimated default probabilities [0.0, 1.0].
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        # Class 1 is default (Bad)
        return self.model.predict_proba(X_arr)[:, 1]

    def calculate_score_from_pd(self, pd_values: np.ndarray) -> np.ndarray:
        """Convert PD estimates to credit scorecard points.

        Score = Offset + Factor * ln( (1 - PD) / PD )
        Clips extreme probabilities to avoid log(0) or division by zero.
        """
        clipped_pd = np.clip(pd_values, 1e-6, 1.0 - 1e-6)
        odds = (1.0 - clipped_pd) / clipped_pd
        scores = self.offset + (self.factor * np.log(odds))
        return np.round(scores, 1)

    def calculate_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict scorecard score directly from feature matrix."""
        pd_vals = self.predict_proba(X)
        return self.calculate_score_from_pd(pd_vals)

    def assign_risk_band(self, scores: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Assign credit risk band based on score thresholds.

        Bands:
        - Low Risk: >= 630
        - Medium Risk: 600 - 629
        - High Risk: 560 - 599
        - Very High Risk: < 560
        """
        scores_arr = np.asarray(scores)
        bands = np.empty(scores_arr.shape, dtype=object)

        bands[scores_arr >= 630.0] = "Low Risk"
        bands[(scores_arr >= 600.0) & (scores_arr < 630.0)] = "Medium Risk"
        bands[(scores_arr >= 560.0) & (scores_arr < 600.0)] = "High Risk"
        bands[scores_arr < 560.0] = "Very High Risk"

        return bands

    def calculate_expected_loss(
        self,
        pd_values: Union[pd.Series, np.ndarray],
        ead: Union[pd.Series, np.ndarray],
        lgd: Optional[Union[float, pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Calculate Expected Loss (EL).

        Formula: Expected Loss = Probability of Default * Loss Given Default * Exposure at Default
        """
        pd_arr = np.asarray(pd_values)
        ead_arr = np.asarray(ead)
        lgd_val = self.lgd_assumption if lgd is None else np.asarray(lgd)
        el = pd_arr * lgd_val * ead_arr
        return np.round(el, 2)

    @staticmethod
    def calculate_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """Compute the Kolmogorov-Smirnov (KS) statistic and optimal threshold.

        KS = max(TPR - FPR) across all classification thresholds.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        ks_distances = tpr - fpr
        ks_idx = np.argmax(ks_distances)
        return float(ks_distances[ks_idx]), float(thresholds[ks_idx])

    @staticmethod
    def calculate_gini(auc_score: float) -> float:
        """Compute Gini coefficient from ROC-AUC: Gini = 2 * AUC - 1."""
        return float(2.0 * auc_score - 1.0)

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Comprehensive scorecard evaluation on holdout test set."""
        y_true = np.asarray(y_test).ravel()
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc_val = float(auc(fpr, tpr))
        gini_val = self.calculate_gini(roc_auc_val)
        ks_val, ks_thresh = self.calculate_ks_statistic(y_true, y_pred_proba)

        cm = confusion_matrix(y_true, y_pred)
        precision_val = float(precision_score(y_true, y_pred, zero_division=0))
        recall_val = float(recall_score(y_true, y_pred, zero_division=0))
        f1_val = float(f1_score(y_true, y_pred, zero_division=0))

        scores = self.calculate_score_from_pd(y_pred_proba)
        bands = self.assign_risk_band(scores)

        results = {
            "roc_auc": round(roc_auc_val, 4),
            "gini": round(gini_val, 4),
            "ks_statistic": round(ks_val, 4),
            "ks_optimal_threshold": round(ks_thresh, 4),
            "precision": round(precision_val, 4),
            "recall": round(recall_val, 4),
            "f1_score": round(f1_val, 4),
            "confusion_matrix": cm.tolist(),
            "mean_score": round(float(np.mean(scores)), 1),
            "median_score": round(float(np.median(scores)), 1),
            "mean_pd": round(float(np.mean(y_pred_proba)), 4),
            "risk_band_counts": pd.Series(bands).value_counts().to_dict(),
        }
        return results

    def plot_roc_curve(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC Curve with AUC and Gini annotations."""
        y_true = np.asarray(y_test).ravel()
        y_proba = self.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        gini_val = self.calculate_gini(roc_auc_val)
        ks_val, _ = self.calculate_ks_statistic(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Scorecard (AUC = {roc_auc_val:.3f}, Gini = {gini_val:.3f}, KS = {ks_val:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        ax.set_title("Credit Scorecard ROC Curve", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

        if save_path:
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_score_distribution(
        self,
        scores: Union[pd.Series, np.ndarray],
        y_true: Optional[Union[pd.Series, np.ndarray]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot credit score distribution segmented by default status."""
        scores_arr = np.asarray(scores)
        fig, ax = plt.subplots(figsize=(9, 5))

        if y_true is not None:
            y_arr = np.asarray(y_true).ravel()
            ax.hist(scores_arr[y_arr == 0], bins=30, alpha=0.65, label="Good Borrowers (Non-default)", color="#2ca02c", density=True)
            ax.hist(scores_arr[y_arr == 1], bins=30, alpha=0.65, label="Bad Borrowers (Default)", color="#d62728", density=True)
        else:
            ax.hist(scores_arr, bins=35, alpha=0.8, color="#1f77b4", edgecolor="black", density=True)

        # Risk band lines (must match assign_risk_band thresholds)
        ax.axvline(560, color="#d9534f", linestyle="--", lw=1.5, label="Very High / High Cutoff (560)")
        ax.axvline(600, color="#f0ad4e", linestyle="--", lw=1.5, label="High / Medium Cutoff (600)")
        ax.axvline(630, color="#5cb85c", linestyle="--", lw=1.5, label="Medium / Low Cutoff (630)")

        ax.set_xlabel("Scorecard Points", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Credit Score Distribution by Performance", fontsize=13, fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

        if save_path:
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def save_model(self, file_path: str) -> None:
        """Serialize trained scorecard to disk."""
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info("Scorecard saved to %s", file_path)

    @classmethod
    def load_model(cls, file_path: str) -> "CreditScorecard":
        """Deserialize trained scorecard from disk."""
        return joblib.load(file_path)
