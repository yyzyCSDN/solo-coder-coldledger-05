from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from ..core.bitemporal import utc
from .certificates import (
    Certificate,
    active_certificate,
    certificate_timeline,
)
from .grace_rules import GraceRule, applicable_grace
from .replacement_chain import chain_for

VALID = "valid"
GRACE = "grace"
EXPIRED = "expired"
UNCOVERED = "uncovered"


@dataclass(frozen=True)
class Reading:
    device_id: str
    at: datetime
    value_c: float


@dataclass(frozen=True)
class Verdict:
    device_id: str
    physical_device: str
    at: datetime
    asof: datetime
    status: str
    compliant: bool
    grace: timedelta
    certificate: Certificate | None
    bias_c: float

    @property
    def acceptable(self) -> bool:
        return self.status in {VALID, GRACE} and self.compliant


@dataclass(frozen=True)
class ReplayedShipment:
    shipment_id: str
    asof: datetime
    verdicts: tuple[Verdict, ...]

    @property
    def acceptable(self) -> bool:
        return bool(self.verdicts) and all(v.acceptable for v in self.verdicts)

    def reasons(self) -> list[str]:
        out = []
        for v in self.verdicts:
            if not v.acceptable:
                which = v.certificate.cert_id if v.certificate else "-"
                out.append(f"{v.device_id}@{v.at.isoformat()} {v.status} cert={which}")
        return out


class CalibrationLedger:
    """Bitemporal view over certificates, grace rules and device replacements.

    Every query takes an ``asof`` (defaulting to the event time): the answer is
    computed strictly from facts recorded up to that instant, so a new
    certificate, grace change or replacement never alters older judgments.
    """

    def __init__(
        self,
        certificates: Iterable[Certificate] = (),
        grace_rules: Iterable[GraceRule] = (),
        replacements: Iterable = (),
        groups: dict[str, frozenset[str]] | None = None,
    ):
        self.certificates = list(certificates)
        self.grace_rules = list(grace_rules)
        self.replacements = list(replacements)
        self.groups = groups or {}
        for c in self.certificates:
            c.validate()
        for r in self.grace_rules:
            r.validate()
        for r in self.replacements:
            r.validate()

    # --- resolution ------------------------------------------------------

    def resolve_device(self, device_id: str, at: datetime, asof: datetime | None = None) -> str:
        t, know = utc(at), utc(asof) if asof else utc(at)
        return _device_at(self.replacements, device_id, t, know)

    def governing(self, device_id: str, at: datetime, asof: datetime | None = None) -> tuple[str, Certificate | None]:
        t, know = utc(at), utc(asof) if asof else utc(at)
        if know < t:
            raise ValueError("asof precedes the event time")
        physical = _device_at(self.replacements, device_id, t, know)
        return physical, active_certificate(self.certificates, physical, know, t)

    # --- single judgment -------------------------------------------------

    def status(self, device_id: str, at: datetime, asof: datetime | None = None) -> Verdict:
        t, know = utc(at), utc(asof) if asof else utc(at)
        physical, cert = self.governing(device_id, t, know)
        grace = applicable_grace(self.grace_rules, physical, know, self.groups.get(physical, frozenset()), at=t)
        if cert is not None:
            return Verdict(
                device_id, physical, t, know, VALID, cert.in_tolerance(), grace, cert,
                cert.deviation_at(t),
            )
        # no certificate covers t: look for the most recent expired one and test grace
        prior = self._last_expired(physical, t, know)
        if prior is not None and t < prior.interval.valid_to + grace:
            return Verdict(
                device_id, physical, t, know, GRACE, prior.in_tolerance(), grace, prior,
                prior.deviation_at(t),
            )
        state = EXPIRED if prior is not None else UNCOVERED
        bias = prior.deviation_at(t) if prior is not None else 0.0
        return Verdict(device_id, physical, t, know, state, prior.in_tolerance() if prior else False, grace, prior, bias)

    def _last_expired(self, physical: str, at: datetime, know: datetime) -> Certificate | None:
        rows = [
            c
            for c in self.certificates
            if c.device_id == physical
            and not c.revoked
            and c.interval.known_at(know)
            and c.interval.valid_to <= at
        ]
        return max(rows, key=lambda c: (c.interval.valid_to, c.version), default=None)

    # --- telemetry correction -------------------------------------------

    def correct_reading(self, reading: Reading, asof: datetime | None = None) -> float:
        v = self.status(reading.device_id, reading.at, asof)
        return reading.value_c - v.bias_c

    # --- batch replay ----------------------------------------------------

    def replay(self, shipment_id: str, readings: Iterable[Reading], asof: datetime) -> ReplayedShipment:
        know = utc(asof)
        verdicts = tuple(self.status(r.device_id, r.at, know) for r in readings)
        return ReplayedShipment(shipment_id, know, verdicts)

    # --- scheduling ------------------------------------------------------

    def next_due(self, device_id: str, asof: datetime) -> datetime | None:
        """Next calibration deadline under knowledge at ``asof``.

        Expiry of the latest certificate known then, extended by the grace rule
        known then. Feeds telemetry.calibration_schedule as the job's ``latest``.
        """
        know = utc(asof)
        physical = _device_at(self.replacements, device_id, know, know)
        certs = [
            c
            for c in self.certificates
            if c.device_id == physical and not c.revoked and c.interval.known_at(know)
        ]
        if not certs:
            return None
        latest = max(certs, key=lambda c: (c.interval.valid_to, c.version))
        grace = applicable_grace(self.grace_rules, physical, know, self.groups.get(physical, frozenset()))
        return latest.interval.valid_to + grace

    # --- history ---------------------------------------------------------

    def certificate_history(self, device_id: str) -> list[Certificate]:
        return certificate_timeline(self.certificates, device_id)

    def lineage(self, device_id: str, asof: datetime | None = None) -> list[str]:
        return chain_for(self.replacements, device_id, asof)


def _device_at(records, logical: str, at: datetime, know: datetime) -> str:
    """Follow replacements recorded by ``know`` whose swap instant is <= ``at``."""
    t = utc(at)
    current = logical
    seen = set()
    while current not in seen:
        seen.add(current)
        step = [
            r for r in records
            if r.old_device == current
            and utc(r.at) <= t
            and r.interval.known_at(know)
            and r.shipment_scope is None
        ]
        if not step:
            return current
        current = min(step, key=lambda r: (utc(r.at), r.interval.recorded_from)).new_device
    return current
