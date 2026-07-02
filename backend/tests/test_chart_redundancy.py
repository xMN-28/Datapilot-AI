import unittest

import pandas as pd

from app.services.analysis_service import run_analysis
from app.services.storage_service import dataset_dir


class ChartRedundancyTests(unittest.TestCase):
    def test_formula_like_correlations_are_excluded(self):
        dataset_id = "test_chart_redundancy"
        rows = []
        for idx in range(80):
            salary_min = 6 + idx * 0.15
            salary_max = salary_min + 4
            unit_price = 25 + (idx % 20)
            quantity = 1 + (idx % 5)
            order_amount = unit_price * quantity
            rows.append(
                {
                    "job_id": idx + 1,
                    "salary_min_lpa": salary_min,
                    "salary_max_lpa": salary_max,
                    "salary_midpoint_lpa": (salary_min + salary_max) / 2,
                    "unit_price": unit_price,
                    "quantity": quantity,
                    "order_amount": order_amount,
                    "tax_amount": order_amount * 0.18,
                    "month": (idx % 12) + 1,
                    "quarter": ((idx % 12) // 3) + 1,
                    "company_rating": 3.2 + (idx % 9) * 0.08,
                    "job_family": ["Data", "Backend", "Frontend", "ML"][idx % 4],
                }
            )
        pd.DataFrame(rows).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)

        analysis = run_analysis(dataset_id)
        titles_and_ids = " ".join(f"{chart['chart_id']} {chart['title']}" for chart in analysis["chart_specs"]).lower()
        relationship_chart = next(chart for chart in analysis["chart_specs"] if chart["chart_id"] == "strongest_correlations")
        excluded = relationship_chart["computed_stats"]["excluded_candidates"]
        excluded_reasons = " ".join(str(item.get("exclusion_reason")) for item in excluded)

        self.assertNotIn("salary_midpoint_lpa vs salary_max_lpa", titles_and_ids)
        self.assertNotIn("order_amount vs tax_amount", titles_and_ids)
        self.assertNotIn("month vs quarter", titles_and_ids)
        self.assertIn("salary_midpoint_lpa is derived", excluded_reasons)
        self.assertIn("order_amount relationship is formula-derived", excluded_reasons)
        self.assertIn("date-derived parts", excluded_reasons)

    def test_chart_generation_includes_analytical_chart_vocabulary(self):
        dataset_id = "test_chart_vocabulary"
        rows = []
        for idx in range(120):
            rows.append(
                {
                    "segment": ["A", "B", "C"][idx % 3],
                    "region": ["North", "South", "West", "East"][idx % 4],
                    "score": 50 + (idx % 30) + (8 if idx % 3 == 0 else 0),
                    "stress": 3 + (idx % 7) * 0.4,
                    "sleep_hours": 4 + (idx % 6) * 0.6,
                    "created_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
                }
            )
        pd.DataFrame(rows).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)

        analysis = run_analysis(dataset_id)
        chart_types = {chart["chart_type"] for chart in analysis["chart_specs"]}

        self.assertTrue({"heatmap", "stacked_bar", "density", "line", "area"}.intersection(chart_types))
        for chart in analysis["chart_specs"]:
            self.assertIn("intent", chart)
            self.assertIn("reason_selected", chart)


if __name__ == "__main__":
    unittest.main()
