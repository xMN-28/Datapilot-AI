import unittest

import pandas as pd

from app.services.chart_service import generate_chart_specs
from app.services.cleaning_service import clean_for_analysis
from app.services.eda_service import run_eda
from app.services.schema_service import detect_schema


class BooleanEdaTests(unittest.TestCase):
    def test_boolean_columns_are_not_analyzed_as_numeric(self):
        df = pd.DataFrame(
            {
                "job_id": [101, 102, 103, 104, 105, 106],
                "salary_lpa": [12.5, 18.0, 9.5, 22.0, 15.0, 11.0],
                "experience_years": [2, 5, 1, 7, 4, 3],
                "is_remote": [True, False, True, False, True, None],
                "is_startup": ["yes", "no", "yes", "no", "yes", "no"],
                "city": ["Bengaluru", "Pune", "Delhi", "Bengaluru", "Mumbai", "Pune"],
                "posted_at": ["2026-01-05", "2026-01-08", "2026-02-11", "2026-02-14", "2026-03-01", "2026-03-15"],
            }
        )

        schema = detect_schema(df)
        cleaned, cleaning_report = clean_for_analysis(df, schema)
        eda = run_eda(cleaned, schema, cleaning_report)
        charts = generate_chart_specs(cleaned, eda)

        self.assertIn("is_remote", eda["boolean"])
        self.assertIn("is_startup", eda["boolean"])
        self.assertNotIn("is_remote", eda["numeric"])
        self.assertNotIn("is_startup", eda["numeric"])
        self.assertEqual(eda["boolean"]["is_remote"]["true_count"], 3)
        self.assertEqual(eda["boolean"]["is_remote"]["false_count"], 2)
        self.assertEqual(eda["skipped"], [])
        self.assertTrue(any(chart["chart_id"] == "is_remote_boolean_counts" for chart in charts))
        self.assertFalse(any(chart["chart_id"] == "is_remote_distribution" for chart in charts))


if __name__ == "__main__":
    unittest.main()
