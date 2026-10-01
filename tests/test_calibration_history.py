import unittest
from datetime import datetime, timedelta, timezone

from coldledger.telemetry.calibration_history import (
    GRACE,
    SUPERSEDED,
    VALID,
    CalibrationCertificate,
    CalibrationRegistry,
    DeviceReplacement,
    GracePeriod,
    MeasuredDeviation,
)
from coldledger.telemetry.calibration_schedule import Bench

UTC = timezone.utc


def dt(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=UTC)


class CalibrationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.registry = CalibrationRegistry(
            certificates=[
                CalibrationCertificate(
                    certificate_id="C-1",
                    device_id="S-1",
                    effective_at=dt(1),
                    expires_at=dt(10),
                    calibrated_at=dt(1),
                    tolerance=0.2,
                ),
                CalibrationCertificate(
                    certificate_id="C-2",
                    device_id="S-1",
                    effective_at=dt(5),
                    expires_at=dt(20),
                    calibrated_at=dt(5),
                    tolerance=0.4,
                    version=2,
                    replaces="C-1",
                    recorded_at=dt(6),
                ),
            ],
            deviations=[
                MeasuredDeviation(
                    reference="D-1",
                    device_id="S-1",
                    observed_at=dt(4, 12),
                    error=0.3,
                    recorded_at=dt(4, 13),
                ),
                MeasuredDeviation(
                    reference="D-2",
                    device_id="S-1",
                    observed_at=dt(5, 12),
                    error=0.3,
                    recorded_at=dt(5, 13),
                ),
            ],
        )

    def test_certificate_version_changes_at_effective_time(self):
        before = self.registry.effective_certificate("S-1", dt(4, 23))
        after = self.registry.effective_certificate("S-1", dt(5))
        self.assertEqual(before.certificate_id, "C-1")
        self.assertEqual(after.certificate_id, "C-2")
        self.assertEqual(
            self.registry.effective_certificate("S-1", dt(4), as_of=dt(4)).certificate_id,
            "C-1",
        )

    def test_old_batch_replays_rules_known_then_and_rules_at_observation_time(self):
        old_assessment = self.registry.assess_reading(
            "S-1",
            dt(4, 12),
            error=0.3,
            as_of=dt(5),
        )
        new_assessment = self.registry.assess_reading(
            "S-1",
            dt(5, 12),
            error=0.3,
            as_of=dt(6),
        )

        self.assertEqual(old_assessment.certificate.certificate_id, "C-1")
        self.assertFalse(old_assessment.within_tolerance)
        self.assertFalse(old_assessment.accepted)
        self.assertEqual(new_assessment.certificate.certificate_id, "C-2")
        self.assertTrue(new_assessment.within_tolerance)
        self.assertTrue(new_assessment.accepted)

    def test_replay_filters_late_recorded_information(self):
        as_of_dt4 = self.registry.replay(as_of=dt(4, 20))
        current = self.registry.replay(as_of=dt(6))
        self.assertEqual(len(as_of_dt4), 1)
        self.assertEqual(as_of_dt4[0].certificate.certificate_id, "C-1")
        self.assertEqual([a.certificate.certificate_id for a in current], ["C-1", "C-2"])

    def test_superseded_certificate_is_not_currently_effective(self):
        current = self.registry.assess_reading("S-1", dt(8), error=0.1, as_of=dt(7))
        superseded = self.registry.assess_reading(
            "S-1",
            dt(8),
            error=0.1,
            as_of=dt(7),
            certificate_id="C-1",
        )
        self.assertEqual(current.status, VALID)
        self.assertEqual(current.certificate.certificate_id, "C-2")
        self.assertEqual(superseded.status, SUPERSEDED)
        self.assertEqual(superseded.certificate.certificate_id, "C-1")

    def test_grace_uses_expired_certificate_until_grace_ends(self):
        registry = CalibrationRegistry(
            certificates=[
                CalibrationCertificate("C-OLD", "S-OLD", dt(1), dt(10), tolerance=0.2),
                CalibrationCertificate("C-NEW", "S-NEW", dt(12), dt(20), tolerance=0.2),
            ],
            graces=[GracePeriod("G-1", "S-OLD", dt(10), dt(12), certificate_id="C-OLD")],
            replacements=[DeviceReplacement("R-1", "S-OLD", "S-NEW", dt(12))],
        )
        in_grace = registry.assess_reading("S-OLD", dt(11), error=0.1)
        after_replacement = registry.assess_reading("S-OLD", dt(12), error=0.1)
        self.assertEqual(in_grace.status, GRACE)
        self.assertEqual(in_grace.certificate.certificate_id, "C-OLD")
        self.assertTrue(in_grace.accepted)
        self.assertEqual(after_replacement.status, VALID)
        self.assertEqual(after_replacement.certificate.certificate_id, "C-NEW")

    def test_replacement_lineage_resolves_historical_and_current_device(self):
        registry = CalibrationRegistry(
            certificates=[CalibrationCertificate("C-A", "A", dt(1), dt(20))],
            replacements=[DeviceReplacement("R-A", "A", "B", dt(8))],
        )
        self.assertEqual(registry.device_at("B", dt(5)), "A")
        self.assertEqual(registry.device_at("A", dt(9)), "B")
        self.assertEqual(registry.device_lineage("A"), ("A", "B"))
        self.assertEqual([r.reference for r in registry.replacement_history("B")], ["R-A"])

    def test_upcoming_schedule_links_certificate_expiry_and_device_replacement(self):
        registry = CalibrationRegistry(
            certificates=[
                CalibrationCertificate("C-OLD", "S-OLD", dt(1), dt(10), tolerance=0.2),
                CalibrationCertificate("C-NEW", "S-NEW", dt(12), dt(25), tolerance=0.2),
            ],
            graces=[GracePeriod("G-1", "S-OLD", dt(10), dt(12), certificate_id="C-OLD")],
            replacements=[DeviceReplacement("R-1", "S-OLD", "S-NEW", dt(12))],
        )
        dues = registry.upcoming_calibrations(
            now=dt(13),
            lead_time=timedelta(days=1),
            default_grace=timedelta(days=2),
            as_of=dt(13),
        )
        self.assertEqual([d.certificate_id for d in dues], ["C-NEW"])
        self.assertEqual(dues[0].due_at, dt(24))
        jobs = registry.calibration_jobs(
            now=dt(13),
            lead_time=timedelta(days=1),
            duration=timedelta(hours=1),
            zone="Z",
            as_of=dt(13),
        )
        assignments = registry.plan_calibrations(
            [Bench("bench", "Z", dt(13), dt(30))],
            now=dt(13),
            lead_time=timedelta(days=1),
            duration=timedelta(hours=1),
            zone="Z",
            as_of=dt(13),
        )
        self.assertEqual([j.key for j in jobs], ["C-NEW"])
        self.assertEqual(assignments[0].item_key, "C-NEW")

    def test_overdue_certificate_without_actual_replacement_is_still_due(self):
        registry = CalibrationRegistry(
            certificates=[CalibrationCertificate("C-OLD", "S-OLD", dt(1), dt(10), tolerance=0.2)],
            replacements=[DeviceReplacement("R-1", "S-OLD", "S-NEW", dt(14))],
        )
        dues = registry.upcoming_calibrations(
            now=dt(11),
            lead_time=timedelta(days=1),
            as_of=dt(11),
        )
        self.assertEqual([d.certificate_id for d in dues], ["C-OLD"])

    def test_timeline_contains_connected_records(self):
        registry = CalibrationRegistry(
            certificates=[CalibrationCertificate("C-A", "A", dt(1), dt(10))],
            deviations=[MeasuredDeviation("D-A", "A", dt(3), 0.1)],
            replacements=[DeviceReplacement("R-A", "A", "B", dt(8))],
        )
        timeline = registry.history("B")
        self.assertEqual([(e.kind, e.reference) for e in timeline], [
            ("certificate", "C-A"),
            ("deviation", "D-A"),
            ("replacement", "R-A"),
        ])


if __name__ == "__main__":
    unittest.main()
