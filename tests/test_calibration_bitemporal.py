import unittest
from datetime import datetime, timedelta, timezone

from coldledger.core.bitemporal import Bitemporal
from coldledger.telemetry.certificates import (
    Certificate,
    DeviationPoint,
    active_certificate,
    certificate_versions,
    correct,
)
from coldledger.telemetry.grace_rules import GraceRule, applicable_grace
from coldledger.telemetry.replacement_chain import active_device, chain_for, replacements_touching
from coldledger.telemetry.calibration_ledger import (
    CalibrationLedger,
    Reading,
    VALID,
    GRACE,
    EXPIRED,
    UNCOVERED,
)
from coldledger.telemetry.calibration_schedule import job_from_certificate

UTC = timezone.utc


def dt(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=UTC)


def span(vf, vt, rf, rt=None):
    return Bitemporal(vf, rf, vt, rt)


class BitemporalTests(unittest.TestCase):
    def test_new_assertion_invisible_before_recording(self):
        c = Certificate("C1", "S1", 1, span(dt(2026, 1, 1), dt(2027, 1, 1), dt(2026, 3, 1)), 0.5)
        # validity covers Feb, but the cert was not recorded until March
        self.assertIsNone(active_certificate([c], "S1", dt(2026, 2, 15)))
        self.assertIsNotNone(active_certificate([c], "S1", dt(2026, 4, 1)))

    def test_retracted_assertion_drops_out_of_later_views(self):
        b = span(dt(2026, 1, 1), dt(2027, 1, 1), dt(2026, 1, 5))
        c = Certificate("C1", "S1", 1, b, 0.5)
        retracted = type(c)(c.cert_id, c.device_id, c.version,
                            type(b)(b.valid_from, b.recorded_from, b.valid_to, dt(2026, 6, 1)),
                            c.tolerance_c)
        self.assertIsNotNone(active_certificate([retracted], "S1", dt(2026, 5, 1)))
        self.assertIsNone(active_certificate([retracted], "S1", dt(2026, 7, 1)))


class CertificateVersionTests(unittest.TestCase):
    def _base(self):
        return Certificate("C9", "S9", 1,
                           span(dt(2026, 1, 1), dt(2026, 7, 1), dt(2026, 1, 10)),
                           0.5, deviations=(DeviationPoint(dt(2026, 1, 1), 0.1),))

    def test_correction_version_only_affects_later_decisions(self):
        base = self._base()
        # on 2026-05-01 an out-of-tolerance as-found deviation is discovered and
        # a corrected version 2 is recorded on 2026-05-02
        v2 = correct(base, recorded_at=dt(2026, 5, 2),
                     deviations=(DeviationPoint(dt(2026, 1, 1), 0.1),
                                 DeviationPoint(dt(2026, 4, 30), 1.2)))
        # replay of an April decision: still sees version 1, in tolerance
        as_april = active_certificate([base, v2], "S9", dt(2026, 4, 30), at=dt(2026, 4, 15))
        self.assertEqual(as_april.version, 1)
        self.assertTrue(as_april.in_tolerance())
        # decision taken on/after May 2 sees version 2
        as_may = active_certificate([base, v2], "S9", dt(2026, 5, 3), at=dt(2026, 4, 15))
        self.assertEqual(as_may.version, 2)
        self.assertFalse(as_may.in_tolerance())

    def test_version_history_ordered(self):
        base = self._base()
        v2 = correct(base, recorded_at=dt(2026, 5, 2),
                     deviations=(DeviationPoint(dt(2026, 1, 1), 0.9),))
        self.assertEqual([c.version for c in certificate_versions([v2, base], "C9")], [1, 2])


class GraceTests(unittest.TestCase):
    def test_grace_change_does_not_rewrite_old_shipment(self):
        old = GraceRule("g-default", "default", timedelta(days=7),
                        span(dt(2026, 1, 1), None, dt(2026, 1, 1), dt(2026, 6, 1)))
        new = GraceRule("g-default2", "default", timedelta(days=0),
                        span(dt(2026, 6, 1), None, dt(2026, 6, 1)))
        # event just past expiry, judged while the 7-day rule was in force
        self.assertEqual(applicable_grace([old, new], "S1", dt(2026, 7, 20),
                                          at=dt(2026, 7, 20)), timedelta(0))
        self.assertEqual(applicable_grace([old, new], "S1", dt(2026, 5, 15),
                                          at=dt(2026, 7, 20)), timedelta(days=7))

    def test_device_specific_grace_outranks_default(self):
        rules = [
            GraceRule("d", "device:S7", timedelta(days=3), span(dt(2026, 1, 1), None, dt(2026, 1, 1))),
            GraceRule("g", "default", timedelta(days=14), span(dt(2026, 1, 1), None, dt(2026, 1, 1))),
        ]
        self.assertEqual(applicable_grace(rules, "S7", dt(2026, 2, 1)), timedelta(days=3))
        self.assertEqual(applicable_grace(rules, "S8", dt(2026, 2, 1)), timedelta(days=14))


class ReplacementTests(unittest.TestCase):
    def test_chain_resolution_and_late_known_swap(self):
        from coldledger.telemetry.replacement_chain import Replacement
        r = Replacement(
            "S1", "S1B", dt(2026, 3, 1), "failure",
            span(dt(2026, 3, 1), None, dt(2026, 4, 1)),
        )
        self.assertEqual(active_device([r], "S1", dt(2026, 3, 15)), "S1")  # not yet known
        self.assertEqual(active_device([r], "S1", dt(2026, 4, 2)), "S1B")
        self.assertEqual(chain_for([r], "S1"), ["S1", "S1B"])
        self.assertEqual(len(replacements_touching([r], "S1B")), 1)


class LedgerReplayTests(unittest.TestCase):
    def setUp(self):
        # S1: cert valid Jan-Jun 2026, recorded Jan 5, measured bias within tol
        self.cert = Certificate(
            "CERT-S1", "S1", 1,
            span(dt(2026, 1, 1), dt(2026, 6, 1), dt(2026, 1, 5)),
            tolerance_c=0.5,
            deviations=(DeviationPoint(dt(2026, 1, 1), 0.2),),
        )
        self.grace = GraceRule("g", "default", timedelta(days=7),
                               span(dt(2026, 1, 1), None, dt(2026, 1, 1)))

    def test_valid_reading_is_corrected_by_bias(self):
        led = CalibrationLedger([self.cert], [self.grace])
        v = led.status("S1", dt(2026, 3, 1))
        self.assertEqual(v.status, VALID)
        self.assertTrue(v.acceptable)
        self.assertAlmostEqual(led.correct_reading(Reading("S1", dt(2026, 3, 1), 5.0)), 4.8)

    def test_grace_then_expired(self):
        led = CalibrationLedger([self.cert], [self.grace])
        self.assertEqual(led.status("S1", dt(2026, 6, 5)).status, GRACE)
        self.assertEqual(led.status("S1", dt(2026, 6, 10)).status, EXPIRED)

    def test_uncovered_device(self):
        led = CalibrationLedger([self.cert], [])
        self.assertEqual(led.status("S2", dt(2026, 3, 1)).status, UNCOVERED)

    def test_replay_old_shipment_ignores_later_certificate(self):
        # A brand-new cert recorded in August covering the whole year.
        late = Certificate(
            "CERT-S2", "S1", 1,
            span(dt(2026, 1, 1), dt(2027, 1, 1), dt(2026, 8, 1)),
            tolerance_c=0.5,
        )
        led = CalibrationLedger([self.cert, late], [self.grace])
        readings = [Reading("S1", dt(2026, 3, 1), 5.0), Reading("S1", dt(2026, 3, 2), 5.0)]
        old = led.replay("SHIP-OLD", readings, dt(2026, 3, 3))
        now = led.replay("SHIP-OLD", readings, dt(2026, 8, 2))
        self.assertTrue(old.acceptable)
        self.assertEqual({v.certificate.cert_id for v in old.verdicts}, {"CERT-S1"})
        # same physical readings re-judged today pick the newer certificate
        self.assertEqual({v.certificate.cert_id for v in now.verdicts}, {"CERT-S2"})

    def test_replay_with_replacement_known_only_afterwards(self):
        from coldledger.telemetry.replacement_chain import Replacement
        cert_b = Certificate(
            "CERT-S1B", "S1B", 1,
            span(dt(2026, 3, 1), dt(2027, 3, 1), dt(2026, 4, 1)),
            tolerance_c=0.5,
        )
        swap = Replacement("S1", "S1B", dt(2026, 3, 1), "failure",
                           span(dt(2026, 3, 1), None, dt(2026, 4, 1)))
        led = CalibrationLedger([self.cert, cert_b], [self.grace], [swap])
        # on March 15 the swap was unknown: still judged against S1's cert
        v_old = led.status("S1", dt(2026, 3, 15), asof=dt(2026, 3, 15))
        self.assertEqual(v_old.physical_device, "S1")
        v_new = led.status("S1", dt(2026, 3, 15), asof=dt(2026, 4, 2))
        self.assertEqual(v_new.physical_device, "S1B")

    def test_next_due_includes_grace_and_tracks_knowledge(self):
        led = CalibrationLedger([self.cert], [self.grace])
        self.assertEqual(led.next_due("S1", dt(2026, 2, 1)), dt(2026, 6, 8))
        no_grace = CalibrationLedger([self.cert], [])
        self.assertEqual(no_grace.next_due("S1", dt(2026, 2, 1)), dt(2026, 6, 1))

    def test_scheduling_job_derived_from_certificate(self):
        job = job_from_certificate(self.cert, lead=timedelta(days=30),
                                   duration=timedelta(hours=2), zone="lab-1",
                                   grace=timedelta(days=7))
        self.assertEqual(job.latest, dt(2026, 6, 8))
        self.assertEqual(job.setup_group, "S1")


if __name__ == "__main__":
    unittest.main()
