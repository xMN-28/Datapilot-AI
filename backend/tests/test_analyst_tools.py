import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.analysis_service import run_analysis
from app.services.analyst_tool_service import answer_question
from app.services.storage_service import dataset_dir


class AnalystToolTests(unittest.TestCase):
    _comparison_analysis = None
    _comparison_dataset_id = "test_student_comparison_tools"

    @classmethod
    def comparison_analysis(cls):
        if cls._comparison_analysis is not None:
            return cls._comparison_analysis
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "Student_ID": idx + 1,
                    "Sleep_Hours_Per_Night": 4.5,
                    "Previous_Semester_GPA": 6.2,
                    "Productivity_Score": 52,
                    "Stress_Level": 8,
                    "Part_Time_Job": "true" if idx < 6 else "false",
                    "Performance_Category": "Low",
                    "Screen_Time_Hours": 9.0,
                }
            )
        for idx in range(18):
            rows.append(
                {
                    "Student_ID": idx + 101,
                    "Sleep_Hours_Per_Night": 8.2,
                    "Previous_Semester_GPA": 7.4,
                    "Productivity_Score": 81,
                    "Stress_Level": 4,
                    "Part_Time_Job": "true" if idx < 4 else "false",
                    "Performance_Category": "High",
                    "Screen_Time_Hours": 5.0,
                }
            )
        for idx in range(10):
            rows.append(
                {
                    "Student_ID": idx + 201,
                    "Sleep_Hours_Per_Night": 6.5,
                    "Previous_Semester_GPA": 6.9,
                    "Productivity_Score": 68,
                    "Stress_Level": 6,
                    "Part_Time_Job": "false",
                    "Performance_Category": "Medium",
                    "Screen_Time_Hours": 7.0,
                }
            )
        pd.DataFrame(rows).to_csv(dataset_dir(cls._comparison_dataset_id) / "raw.csv", index=False)
        cls._comparison_analysis = run_analysis(cls._comparison_dataset_id)
        return cls._comparison_analysis

    def test_filtered_aggregation_for_sleep_and_stress(self):
        dataset_id = "test_student_analyst_tools"
        rows = []
        for idx in range(20):
            sleep = 3.0 + (idx % 6) * 0.7
            stress = 8 if sleep < 4 else 5 if sleep < 6 else 3
            rows.append(
                {
                    "student_id": idx + 1,
                    "sleep_hours_per_night": sleep,
                    "stress_level": stress,
                    "productivity_score": 70 - stress * 3,
                    "part_time_job": "yes" if idx % 2 else "no",
                    "gender": "" if idx % 4 == 0 else "Female",
                }
            )
        pd.DataFrame(rows).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)
        analysis = run_analysis(dataset_id)

        response = answer_question(dataset_id, "What is the average stress level of students who sleep less than 4 hours?", analysis)

        self.assertEqual(response["tool_result"]["tool"], "filtered_aggregation")
        self.assertEqual(response["tool_result"]["column"], "stress_level")
        self.assertEqual(response["tool_result"]["filters_applied"][0]["column"], "sleep_hours_per_night")
        self.assertEqual(response["tool_result"]["filters_applied"][0]["operator"], "<")
        self.assertEqual(response["tool_result"]["result"], 8.0)

    def test_missing_category_values_are_not_chart_categories(self):
        dataset_id = "test_missing_category_cleaning"
        pd.DataFrame(
            {
                "gender": ["Male", "Female", None, np.nan, "nan", "N/A", "Other", "", "null"],
                "score": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        ).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)

        analysis = run_analysis(dataset_id)
        gender_stats = analysis["eda_report"]["categorical"]["gender"]
        labels = {item["value"] for item in gender_stats["top_values"]}
        chart = next(c for c in analysis["chart_specs"] if c["chart_id"] == "gender_top_values")
        chart_labels = {str(item.get("value")) for item in chart["data"]}

        self.assertEqual(labels, {"Male", "Female", "Other"})
        self.assertEqual(chart_labels, {"Male", "Female", "Other"})
        self.assertNotIn("nan", {label.lower() for label in labels})
        self.assertNotIn("N/A", labels)
        self.assertEqual(gender_stats["missing_count"], 6)
        self.assertEqual(chart["excluded_missing_count"], 6)

    def test_compare_groups_maps_sleep_and_gpa_columns(self):
        dataset_id = "test_compare_sleep_gpa"
        rows = []
        for idx in range(60):
            sleep = 4.0 if idx < 20 else 6.5
            gpa = 6.4 if idx < 20 else 7.3
            rows.append({"Student_ID": idx + 1, "Sleep_Hours_Per_Night": sleep, "Semester_GPA": gpa, "Gender": "Female" if idx % 2 else "Male"})
        pd.DataFrame(rows).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)
        analysis = run_analysis(dataset_id)

        response = answer_question(dataset_id, "compare the average semester gpa of students who sleep less than 5 hours a day and students who sleep more than 5 hours a day", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "compare_groups")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["value_column"], "Semester_GPA")
        self.assertEqual(result["groups"][0]["matched_rows"], 20)
        self.assertEqual(result["groups"][1]["matched_rows"], 40)
        self.assertAlmostEqual(result["groups"][0]["result"], 6.4)
        self.assertAlmostEqual(result["groups"][1]["result"], 7.3)
        self.assertAlmostEqual(result["difference"], -0.9)
        self.assertTrue(response["evidence"][0]["title"].startswith("Computed Tool Result"))
        self.assertTrue(any(item["title"] == "Tool trace" for item in response.get("debug_evidence", [])))

    def test_compare_productivity_for_sleep_less_than_vs_at_least(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "compare average productivity for students who sleep less than 5 hours vs students who sleep at least 8 hours", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "compare_groups")
        self.assertEqual(result["value_column"], "Productivity_Score")
        self.assertEqual(result["groups"][0]["filters"][0]["operator"], "<")
        self.assertEqual(result["groups"][1]["filters"][0]["operator"], ">=")
        self.assertEqual(result["groups"][0]["matched_rows"], 12)
        self.assertEqual(result["groups"][1]["matched_rows"], 18)
        self.assertEqual(result["groups"][0]["result"], 52.0)
        self.assertEqual(result["groups"][1]["result"], 81.0)
        self.assertEqual(result["difference"], -29.0)

    def test_filtered_average_gpa_for_sleep_five_or_fewer(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "What is the average GPA of students who sleep 5 or fewer hours per night?", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "filtered_aggregation")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["column"], "Previous_Semester_GPA")
        self.assertEqual(result["filters_applied"][0]["column"], "Sleep_Hours_Per_Night")
        self.assertEqual(result["filters_applied"][0]["operator"], "<=")
        self.assertEqual(result["filters_applied"][0]["value"], 5.0)
        self.assertEqual(result["matched_rows"], 12)
        self.assertLess(result["matched_rows"], 40)
        self.assertEqual(result["result"], 6.2)
        self.assertIn("tool_trace", response)
        self.assertEqual(response["tool_trace"]["user_question"], "What is the average GPA of students who sleep 5 or fewer hours per night?")
        self.assertEqual(response["tool_trace"]["actual_tool_called"], "filtered_aggregation")
        self.assertEqual(response["tool_trace"]["extracted_filters"][0]["operator"], "<=")
        self.assertTrue(any(item["title"] == "Tool trace" for item in response.get("debug_evidence", [])))

    def test_chat_endpoint_returns_tool_trace_and_last_trace(self):
        self.comparison_analysis()
        client = TestClient(app)
        question = "What is the average GPA of students who sleep 5 or fewer hours per night?"

        response = client.post(f"/api/datasets/{self._comparison_dataset_id}/chat", json={"question": question})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("tool_trace", body)
        self.assertEqual(body["tool_trace"]["user_question"], question)
        self.assertEqual(body["tool_trace"]["actual_tool_called"], "filtered_aggregation")

        trace_response = client.get(f"/api/datasets/{self._comparison_dataset_id}/chat/last-trace")
        self.assertEqual(trace_response.status_code, 200)
        self.assertEqual(trace_response.json()["user_question"], question)

    def test_filtered_average_gpa_for_sleep_eight_or_more(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "What is the average GPA of students who sleep 8 or more hours?", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "filtered_aggregation")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["column"], "Previous_Semester_GPA")
        self.assertEqual(result["filters_applied"][0]["operator"], ">=")
        self.assertEqual(result["filters_applied"][0]["value"], 8.0)
        self.assertEqual(result["matched_rows"], 18)
        self.assertLess(result["matched_rows"], 40)
        self.assertEqual(result["result"], 7.4)

    def test_compare_previous_gpa_for_sleep_or_less_vs_or_more(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "Compare previous semester GPA of students who sleep 5 hours or less with students who sleep 8 or more hours.", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "compare_groups")
        self.assertEqual(result["value_column"], "Previous_Semester_GPA")
        self.assertEqual(result["groups"][0]["filters"][0]["operator"], "<=")
        self.assertEqual(result["groups"][1]["filters"][0]["operator"], ">=")
        self.assertEqual(result["groups"][0]["matched_rows"], 12)
        self.assertEqual(result["groups"][1]["matched_rows"], 18)
        self.assertEqual(result["groups"][0]["result"], 6.2)
        self.assertEqual(result["groups"][1]["result"], 7.4)
        self.assertEqual(result["difference"], -1.2)
        self.assertTrue(response["evidence"][0]["title"].startswith("Computed Tool Result"))

    def test_compare_stress_for_part_time_true_vs_false(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "compare average stress level for part-time job true vs false", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "compare_groups")
        self.assertEqual(result["value_column"], "Stress_Level")
        self.assertEqual(result["groups"][0]["matched_rows"], 10)
        self.assertEqual(result["groups"][1]["matched_rows"], 30)
        self.assertAlmostEqual(result["groups"][0]["result"], 6.4)
        self.assertAlmostEqual(result["groups"][1]["result"], 5.4667)
        self.assertAlmostEqual(result["difference"], 0.9333)

    def test_compare_stress_for_with_and_without_part_time_jobs(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "Compare stress levels of students with part time jobs and without part time jobs.", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "compare_groups")
        self.assertEqual(result["value_column"], "Stress_Level")
        self.assertEqual(result["groups"][0]["matched_rows"], 10)
        self.assertEqual(result["groups"][1]["matched_rows"], 30)
        self.assertAlmostEqual(result["groups"][0]["result"], 6.4)
        self.assertAlmostEqual(result["groups"][1]["result"], 5.4667)

    def test_compare_screen_time_for_high_vs_low_performance(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "compare average screen time for high vs low performance category", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "compare_groups")
        self.assertEqual(result["value_column"], "Screen_Time_Hours")
        self.assertEqual(result["groups"][0]["matched_rows"], 18)
        self.assertEqual(result["groups"][1]["matched_rows"], 12)
        self.assertEqual(result["groups"][0]["result"], 5.0)
        self.assertEqual(result["groups"][1]["result"], 9.0)
        self.assertEqual(result["difference"], -4.0)

    def test_generic_ecommerce_count_with_two_filters(self):
        dataset_id = "test_generic_ecommerce_tools"
        rows = [
            {"Order_ID": 1, "Shipping_Method": "Express", "Returned": "Yes", "Delivery_Days": 2, "Order_Amount": 120},
            {"Order_ID": 2, "Shipping_Method": "Express", "Returned": "No", "Delivery_Days": 3, "Order_Amount": 80},
            {"Order_ID": 3, "Shipping_Method": "Standard", "Returned": "Yes", "Delivery_Days": 6, "Order_Amount": 60},
            {"Order_ID": 4, "Shipping_Method": "Express", "Returned": "Yes", "Delivery_Days": 1, "Order_Amount": 200},
        ]
        pd.DataFrame(rows).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)
        analysis = run_analysis(dataset_id)

        response = answer_question(dataset_id, "How many orders used express shipping and were returned?", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "filtered_aggregation")
        self.assertEqual(result["aggregation"], "count")
        self.assertEqual(result["matched_rows"], 2)
        self.assertEqual(result["result"], 2)
        filters = {(f["column"], f["operator"]) for f in result["filters_applied"]}
        self.assertIn(("Shipping_Method", "=="), filters)
        self.assertTrue(any(f["column"] == "Returned" for f in result["filters_applied"]))

    def test_generic_compare_delivery_days_for_shipping_values(self):
        dataset_id = "test_generic_shipping_compare"
        rows = []
        for idx in range(5):
            rows.append({"Shipping_Method": "Standard", "Delivery_Days": 6 + idx % 2, "Returned": "No"})
        for idx in range(4):
            rows.append({"Shipping_Method": "Same-Day", "Delivery_Days": 1, "Returned": "No"})
        pd.DataFrame(rows).to_csv(dataset_dir(dataset_id) / "raw.csv", index=False)
        analysis = run_analysis(dataset_id)

        response = answer_question(dataset_id, "Compare delivery days for standard shipping vs same-day shipping.", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "compare_groups")
        self.assertEqual(result["value_column"], "Delivery_Days")
        self.assertEqual(result["groups"][0]["matched_rows"], 5)
        self.assertEqual(result["groups"][1]["matched_rows"], 4)
        self.assertAlmostEqual(result["groups"][0]["result"], 6.4)
        self.assertEqual(result["groups"][1]["result"], 1.0)

    def test_generic_groupby_highest_average(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "Which performance category has the highest average productivity score?", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "groupby_aggregation")
        self.assertEqual(result["group_by"], "Performance_Category")
        self.assertEqual(result["value_column"], "Productivity_Score")
        self.assertEqual(result["result"][0]["group"], "High")

    def test_generic_correlation_query(self):
        analysis = self.comparison_analysis()
        response = answer_question(self._comparison_dataset_id, "Do students with higher screen time have lower GPA?", analysis)
        result = response["tool_result"]

        self.assertEqual(result["tool"], "correlation_query")
        self.assertIn(result["x_column"], {"Screen_Time_Hours", "Previous_Semester_GPA"})
        self.assertIn(result["y_column"], {"Screen_Time_Hours", "Previous_Semester_GPA"})
        self.assertNotEqual(result["x_column"], result["y_column"])


if __name__ == "__main__":
    unittest.main()
