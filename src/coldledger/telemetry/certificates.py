from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from ..core.bitemporal import Bitemporal, as_of, utc


@dataclass(frozen=True)
class DeviationPoint:
    """A measured bias sample for a certificate (as-found/as-left or later check)."""

    at: datetime
    offset_c: float
    source: str = "calibration"

    def norm(self) -> "DeviationPoint":
        return DeviationPoint(utc(self.at), self.offset_c, self.source)


@dataclass(frozen=True)
class Certificate:
    """One immutable issued version of a calibration certificate.

    cert_id  : identity shared across corrected re-issues of the same physical cert.
    version  : 1 for the original, 2+ for corrections that supersede earlier versions.
    interval : validity (valid axis) and assertion/record axis.
    deviations are part of the certificate record; extra checks are appended as
    separate certificates only when they change the validity claim.
    """

    cert_id: str
    device_id: str
    version: int
    interval: Bitemporal
    tolerance_c: float
    deviations: tuple[DeviationPoint, ...] = ()
    revoked: bool = False
    calibrates_job: str | None = None
    note: str = ""

    @property
    def valid_from(self) -> datetime:
        return self.interval.valid_from

    @property
    def valid_to(self) -> datetime:
        if self.interval.valid_to is None:
            raise ValueError("open-ended certificate has no valid_to")
        return self.interval.valid_to

    def validate(self) -> None:
        if self.version < 1:
            raise ValueError("version must start at 1")
        if self.interval.norm().valid_to is None:
            raise ValueError("calibration certificate needs a declared expiry")
        if self.tolerance_c <= 0:
            raise ValueError("tolerance must be positive")
        for d in self.deviations:
            dn = d.norm()
            if not self.valid_from <= dn.at <= self.valid_to:
                raise ValueError("deviation outside certificate validity window")

    def max_abs_deviation(self) -> float:
        return max((abs(d.offset_c) for d in self.deviations), default=0.0)

    def in_tolerance(self) -> bool:
        return self.max_abs_deviation() <= self.tolerance_c + 1e-9

    def deviation_at(self, at: datetime) -> float:
        """Bias applicable to a reading: the most recent sample not after ``at``."""
        t = utc(at)
        prior = sorted(
            (d for d in (d.norm() for d in self.deviations) if d.at <= t),
            key=lambda d: d.at,
        )
        if not prior:
            return 0.0
        return prior[-1].offset_c


def _known(records: Iterable[Certificate], asof: datetime) -> list[Certificate]:
    return as_of(records, asof, known="interval")


def certificate_versions(certificates: Iterable[Certificate], cert_id: str, asof: datetime | None = None) -> list[Certificate]:
    rows = [c for c in certificates if c.cert_id == cert_id]
    if asof is not None:
        rows = _known(rows, asof)
    return sorted(rows, key=lambda c: (c.version, c.interval.recorded_from))


def known_certificates(certificates: Iterable[Certificate], device_id: str, asof: datetime, at: datetime | None = None) -> list[Certificate]:
    """Certificates on the books at ``asof`` whose validity covers ``at``.

    The two instants differ during replay: at=event time selects the covering
    certificate, asof=decision time hides anything recorded afterwards.
    """
    know = utc(asof)
    t = utc(at) if at is not None else know
    rows = [
        c
        for c in _known(certificates, know)
        if c.device_id == device_id and not c.revoked and c.interval.covers(t)
    ]
    return sorted(rows, key=lambda c: (c.valid_from, c.version, c.interval.recorded_from))


def active_certificate(certificates: Iterable[Certificate], device_id: str, asof: datetime, at: datetime | None = None) -> Certificate | None:
    """Certificate governing the event at ``at`` under knowledge at ``asof``.

    Latest valid_from wins; ties resolved by highest version (a correction known
    by then) and earliest recording. A certificate recorded after ``asof`` is
    invisible here even if its validity window covers the shipment.
    """
    rows = known_certificates(certificates, device_id, asof, at)
    if not rows:
        return None
    return max(rows, key=lambda c: (c.valid_from, c.interval.recorded_from, c.version))


def certificate_timeline(certificates: Iterable[Certificate], device_id: str) -> list[Certificate]:
    rows = [c for c in certificates if c.device_id == device_id]
    return sorted(rows, key=lambda c: (c.valid_from, c.interval.recorded_from, c.version))


def correct(old: Certificate, *, recorded_at: datetime, deviations: tuple[DeviationPoint, ...], tolerance_c: float | None = None, note: str = "") -> Certificate:
    """Re-issue a corrected version. Only judgments after ``recorded_at`` see it."""
    if not deviations:
        raise ValueError("correction must carry measured deviations")
    b = old.interval.norm()
    if utc(recorded_at) < b.recorded_from:
        raise ValueError("cannot backdate a correction before the original was known")
    return Certificate(
        cert_id=old.cert_id,
        device_id=old.device_id,
        version=old.version + 1,
        interval=Bitemporal(b.valid_from, utc(recorded_at), b.valid_to, None),
        tolerance_c=old.tolerance_c if tolerance_c is None else tolerance_c,
        deviations=tuple(d.norm() for d in deviations),
        revoked=old.revoked,
        calibrates_job=old.calibrates_job,
        note=note,
    )


def valid_through(cert: Certificate, grace: timedelta = timedelta()) -> datetime:
    return cert.interval.norm().valid_to + grace
