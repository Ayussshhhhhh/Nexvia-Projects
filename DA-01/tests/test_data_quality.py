"""Pure business-rule tests; no third-party dependencies."""
import unittest
from datetime import date


def margin(revenue, cost):
    return None if revenue in (None, 0) else (revenue - (cost or 0)) / revenue


def yoy(current, prior):
    return None if prior in (None, 0) else (current - prior) / prior


def rolling_average(values, window):
    if window is None or window <= 0:
        return None
    sample = values[-window:]
    return None if len(sample) < window else sum(sample) / window


def days_supply(on_hand, trailing_units, days=90):
    demand = trailing_units / days
    return None if demand == 0 else on_hand / demand


def attrition(exits, average_headcount):
    return None if average_headcount in (None, 0) else exits / average_headcount


class MetricTests(unittest.TestCase):
    def test_margin_and_divide_by_zero(self):
        self.assertAlmostEqual(margin(200, 150), .25)
        self.assertIsNone(margin(0, 0))

    def test_yoy(self):
        self.assertAlmostEqual(yoy(120, 100), .2)
        self.assertIsNone(yoy(10, 0))
        self.assertIsNone(yoy(10, None))

    def test_rolling_monthly_average(self):
        self.assertEqual(rolling_average([1, 2, 3, 4], 3), 3)
        self.assertIsNone(rolling_average([1, 2], 3))
        self.assertIsNone(rolling_average([1, 2, 3], 0))

    def test_inventory_days_supply(self):
        self.assertEqual(days_supply(30, 90), 30)
        self.assertIsNone(days_supply(30, 0))

    def test_attrition(self):
        self.assertAlmostEqual(attrition(2, 40), .05)
        self.assertIsNone(attrition(1, 0))

    def test_date_continuity_principle(self):
        dates = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
        self.assertEqual((dates[-1] - dates[0]).days + 1, len(dates))


if __name__ == "__main__":
    unittest.main()
