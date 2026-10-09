import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from data.building_registry import BUILDINGS
from services import recommendation_service as rec

BUILDING_IDS = list(BUILDINGS)

REC_FIELDS = {
    "id",
    "building_id",
    "building_name",
    "category",
    "severity",
    "title",
    "summary",
    "evidence",
    "possible_causes",
    "recommended_actions",
    "timestamp",
    "confidence",
    "status",
}


def _frame(rows):
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _row(ts, energy=10.0, score=0.62, flag=1, z=0.0, cov=1.0, masked=0, missing=0):
    return {
        "timestamp": ts,
        "energy_kwh": energy,
        "anomaly_score": score,
        "anomaly_flag_p98": flag,
        "anomaly_flag_p99": 1 if (flag and score >= 0.6) else 0,
        "robust_z_168": z,
        "coverage": cov,
        "is_masked": masked,
        "is_missing": missing,
    }


def _hourly(base, hours, **kw):
    t0 = pd.Timestamp(base, tz="UTC")
    return [_row((t0 + pd.Timedelta(hours=h)).isoformat(), **kw) for h in hours]


class TestSingleDirectionRules(unittest.TestCase):
    def test_strong_negative_gives_drop(self):
        df = _frame(_hourly("2017-01-01", [0], z=-5.2, score=0.65))
        row = list(df.itertuples())[0]
        r = rec._single_rec("a-block", "A Block", row, "unusual_drop", "A")
        self.assertEqual(r["category"], "unusual_drop")
        self.assertEqual(r["severity"], "medium")
        self.assertEqual(r["status"], "needs_review")
        self.assertLess(r["evidence"]["robust_z_168"], -3)

    def test_strong_positive_gives_spike(self):
        df = _frame(_hourly("2017-01-01", [0], z=9.1, score=0.66))
        row = list(df.itertuples())[0]
        r = rec._single_rec("b-block", "B Block", row, "unusual_spike", "B")
        self.assertEqual(r["category"], "unusual_spike")
        self.assertEqual(r["severity"], "high")
        self.assertGreater(r["evidence"]["robust_z_168"], 3)


class TestClusterRule(unittest.TestCase):
    def test_close_flags_form_one_cluster(self):
        df = _frame(_hourly("2017-01-01", [0, 3, 5, 9], z=-4.0))
        events = rec._find_clusters(df)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["count"], 4)

    def test_scattered_flags_form_no_cluster(self):
        df = _frame(_hourly("2017-01-01", [0, 72, 200], z=-4.0))
        self.assertEqual(rec._find_clusters(df), [])


class TestOvernightRule(unittest.TestCase):

    def test_ist_hour_conversion_with_day_rollover(self):
        df = _frame(_hourly("2017-01-01", [18, 19, 0], z=0.0))
        hours, dates = rec._ist_parts(df)
        self.assertEqual(list(hours), [5, 23, 0])
        self.assertEqual(str(dates.iloc[0]), "2017-01-01")
        self.assertEqual(str(dates.iloc[1]), "2017-01-01")
        self.assertEqual(str(dates.iloc[2]), "2017-01-02")

    def test_single_overnight_flag_does_not_fire(self):
        df = _frame(_hourly("2017-01-01", [20], z=4.0))
        self.assertIsNone(rec._overnight_rec("a-block", "A Block", df, "anomaly_flag_p98"))

    def test_recurring_ist_overnight_fires(self):
        rows = []
        for d in [1, 2, 3, 4]:
            rows += _hourly(f"2017-01-0{d}", [20, 21], z=3.5)
        df = _frame(rows)
        r = rec._overnight_rec("a-block", "A Block", df, "anomaly_flag_p98")
        self.assertIsNotNone(r)
        self.assertEqual(r["category"], "overnight_pattern")
        self.assertIn("IST", r["evidence"]["overnight_window"])
        self.assertGreater(r["evidence"]["night_rate"], r["evidence"]["day_rate"])
        self.assertGreaterEqual(r["evidence"]["distinct_dates"], 3)

    def test_morning_utc_flags_are_not_night(self):
        rows = []
        for d in [1, 2, 3, 4]:
            rows += _hourly(f"2017-01-0{d}", [1, 2], z=3.5)
        df = _frame(rows)
        self.assertIsNone(rec._overnight_rec("a-block", "A Block", df, "anomaly_flag_p98"))

    def test_no_enrichment_no_fire(self):
        rows = []
        for d in [1, 2, 3]:
            rows += _hourly(f"2017-01-0{d}", [20, 21], z=3.5)
            rows += _hourly(f"2017-01-0{d}", [8, 9, 10, 11, 12, 13, 14, 15], z=1.0)
        df = _frame(rows)
        night = df[df["timestamp"].dt.hour.between(20, 21)]
        self.assertEqual((df["anomaly_flag_p98"] == 1).sum(), 6 + 24)
        self.assertIsNone(rec._overnight_rec("a-block", "A Block", df, "anomaly_flag_p98"))

    def test_overnight_severity_thresholds(self):
        self.assertEqual(rec._severity_for_overnight(0.46), "high")
        self.assertEqual(rec._severity_for_overnight(0.25), "high")
        self.assertEqual(rec._severity_for_overnight(0.15), "medium")
        self.assertEqual(rec._severity_for_overnight(0.05), "low")

    def test_ist_night_differs_by_building(self):
        fired = {b: any(r["category"] == "overnight_pattern"
                        for r in rec.get_insights(b)["recommendations"]) for b in BUILDING_IDS}
        self.assertTrue(fired["mess"])
        self.assertFalse(any(fired[b] for b in ("a-block", "b-block", "library")))
        self.assertFalse(any(
            r["category"] == "overnight_pattern" and r["severity"] == "high"
            for b in BUILDING_IDS for r in rec.get_insights(b)["recommendations"]
            if b != "mess"))


class TestPolarityAndGaps(unittest.TestCase):
    def test_b_block_not_auto_drop(self):
        d = rec.get_insights("b-block", severity="p98")
        cats = {r["category"] for r in d["recommendations"]}
        self.assertIn("unusual_spike", cats)
        self.assertIn("unusual_drop", cats)
        for r in d["recommendations"]:
            if r["category"] == "unusual_spike":
                self.assertGreaterEqual(r["evidence"]["robust_z_168"], 3)
            if r["category"] == "unusual_drop":
                self.assertLessEqual(r["evidence"]["robust_z_168"], -3)

    def test_masked_rows_not_treated_as_normal(self):
        rows = _hourly("2017-01-01", [0, 1, 2, 3], z=-4.0)
        rows += [
            {
                "timestamp": "2017-01-02T00:00:00+00:00",
                "energy_kwh": float("nan"),
                "anomaly_score": float("nan"),
                "anomaly_flag_p98": 0,
                "anomaly_flag_p99": 0,
                "robust_z_168": float("nan"),
                "coverage": 0.0,
                "is_masked": 1,
                "is_missing": 1,
            }
        ]
        df = _frame(rows)
        scorable = df[df["anomaly_score"].notna()]
        self.assertEqual(len(scorable), 4)
        events = rec._find_clusters(scorable)
        self.assertEqual(events[0]["count"], 4)

    def test_missing_optional_fields_do_not_crash(self):
        df = _frame(_hourly("2017-01-01", [0, 1, 2, 30, 31, 32], z=-4.0))
        df = df.drop(columns=["coverage", "is_masked", "is_missing"])
        dq = rec._data_quality_rec("a-block", "A Block", df)
        self.assertTrue(dq is None or dq["category"] == "data_quality")
        df2 = df.drop(columns=["robust_z_168"])
        over = rec._overnight_rec("a-block", "A Block", df2, "anomaly_flag_p98")
        self.assertTrue(over is None or over["category"] == "overnight_pattern")


class TestClusterDensityAndDirection(unittest.TestCase):
    def test_density_severity_mapping(self):
        self.assertEqual(rec._severity_for_cluster(42, 9.3), "high")
        self.assertEqual(rec._severity_for_cluster(17, 8.0), "high")
        self.assertEqual(rec._severity_for_cluster(3, 36.0), "medium")
        self.assertEqual(rec._severity_for_cluster(19, 4.75), "medium")
        self.assertEqual(rec._severity_for_cluster(27, 3.58), "low")
        self.assertEqual(rec._severity_for_cluster(22, 3.12), "low")

    def test_direction_labels(self):
        self.assertEqual(rec._cluster_direction(3.9), "high-use")
        self.assertEqual(rec._cluster_direction(-3.7), "low-use")
        self.assertEqual(rec._cluster_direction(0.3), "mixed")
        self.assertEqual(rec._cluster_direction(float("nan")), "mixed")
        self.assertEqual(rec._cluster_direction(None), "mixed")

    def test_real_clusters_have_direction_and_density(self):
        for bid, want in (("a-block", "low-use"), ("b-block", "high-use")):
            with self.subTest(building=bid):
                recs = [r for r in rec.get_insights(bid)["recommendations"]
                        if r["category"] == "repeated_anomalies"]
                self.assertTrue(recs)
                by_n = sorted(recs, key=lambda r: r["evidence"]["affected_hours"], reverse=True)
                self.assertEqual(by_n[0]["evidence"]["direction"], want)
                self.assertGreater(by_n[0]["evidence"]["density_per_24h"], 0)

    def test_cluster_severity_not_all_high(self):
        sevs = [r["severity"] for b in BUILDING_IDS
                for r in rec.get_insights(b)["recommendations"]
                if r["category"] == "repeated_anomalies"]
        self.assertIn("medium", sevs)
        self.assertIn("low", sevs)


class TestMissingDataWording(unittest.TestCase):
    def test_unscorable_share_headline(self):
        d = rec.get_insights("a-block")
        dq = [r for r in d["recommendations"] if r["category"] == "data_quality"][0]
        ev = dq["evidence"]
        self.assertAlmostEqual(ev["unscorable_share"], ev["unscorable_hours"] / ev["total_hours"], places=4)
        self.assertIn("could not be scored", dq["summary"])
        self.assertIn("Of the unscorable hours", dq["summary"])
        self.assertNotIn("masked (", dq["summary"])
        self.assertIn("masked_ratio", ev)
        self.assertIn("masked_hours", ev)


class TestInsightsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app

        flask_app.app.testing = True
        cls.client = flask_app.app.test_client()

    def test_unknown_building_404(self):
        r = self.client.get("/api/buildings/nope/insights")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.get_json())

    def test_bad_severity_400(self):
        r = self.client.get("/api/buildings/a-block/insights?severity=p999")
        self.assertEqual(r.status_code, 400)

    def test_insights_schema_all_buildings(self):
        for bid in BUILDING_IDS:
            with self.subTest(building=bid):
                r = self.client.get(f"/api/buildings/{bid}/insights")
                self.assertEqual(r.status_code, 200)
                body = r.get_json()
                for key in ("building", "available", "message", "building_id",
                            "severity", "rules_version", "total", "recommendations"):
                    self.assertIn(key, body)
                self.assertTrue(body["available"])
                self.assertIsInstance(body["recommendations"], list)
                for item in body["recommendations"]:
                    self.assertTrue(REC_FIELDS.issubset(item.keys()))
                    self.assertEqual(item["status"], "needs_review")
                    self.assertIn(item["severity"], ("low", "medium", "high"))
                    self.assertIn(item["confidence"], ("limited", "moderate", "strong"))

    def test_severity_and_limit_params(self):
        dflt = self.client.get("/api/buildings/a-block/insights").get_json()
        self.assertEqual(dflt["severity"], "p98")
        p99 = self.client.get("/api/buildings/a-block/insights?severity=p99").get_json()
        self.assertEqual(p99["severity"], "p99")
        lim = self.client.get("/api/buildings/a-block/insights?limit=2").get_json()
        self.assertLessEqual(len(lim["recommendations"]), 2)
        self.assertGreaterEqual(lim["total"], len(lim["recommendations"]))

    def test_deterministic(self):
        a = rec.get_insights("mess", severity="p98")
        b = rec.get_insights("mess", severity="p98")
        self.assertEqual(a, b)

    def test_existing_endpoints_and_pages(self):
        c = self.client
        self.assertEqual(c.get("/api/buildings").status_code, 200)
        self.assertEqual(c.get("/api/buildings/a-block").status_code, 200)
        self.assertEqual(c.get("/api/buildings/a-block/trend?range=7d").status_code, 200)
        self.assertEqual(c.get("/api/buildings/a-block/anomalies?severity=p99").status_code, 200)
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(c.get("/about").status_code, 200)
        for bid in BUILDING_IDS:
            with self.subTest(page=bid):
                self.assertEqual(c.get(f"/building/{bid}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
