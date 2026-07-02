import unittest
from uuid import uuid4

import pandas as pd

from app.services.prediction_service import train_model
from app.services.storage_service import dataset_dir


class RegressionTrainingTests(unittest.TestCase):
    def test_company_rating_trains_as_regression(self):
        dataset_id = f"test_regression_{uuid4().hex[:8]}"
        rows = []
        for idx in range(80):
            salary = 8.0 + (idx % 15) * 0.8
            experience = 1 + (idx % 12)
            remote = "yes" if idx % 3 == 0 else "no"
            rating = 2.4 + salary * 0.08 + experience * 0.035 + (0.1 if remote == "yes" else 0)
            rows.append(
                {
                    "job_id": f"JOB-{idx:04d}",
                    "company_id": f"CO-{idx:04d}",
                    "salary_midpoint_lpa": salary,
                    "experience_years": experience,
                    "remote_allowed": remote,
                    "job_family": ["Data", "Backend", "Frontend", "ML"][idx % 4],
                    "company_rating": round(rating, 2),
                }
            )
        pd.DataFrame(rows).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)

        summary = train_model(dataset_id, "company_rating", "fast")

        self.assertEqual(summary["task_type"], "regression")
        self.assertIn("mae", summary["metrics"])
        self.assertIn("rmse", summary["metrics"])
        self.assertIn("r2", summary["metrics"])
        self.assertNotIn("accuracy", summary["metrics"])
        self.assertNotIn("f1_weighted", summary["metrics"])
        self.assertEqual(summary["baseline"]["strategy"], "mean_prediction")


if __name__ == "__main__":
    unittest.main()
