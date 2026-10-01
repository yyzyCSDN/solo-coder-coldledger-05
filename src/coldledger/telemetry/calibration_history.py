"""Bitemporal calibration evidence and device replacement history."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from threading import RLock
from typing import Iterable

from .calibration_schedule import CalibrationJob, build as build_schedule


VALID = "valid"
GRACE = "grace"
EXPIRED = "expired"
REVOKED = "revoked"
SUPERSEDED = "superseded"
UNCALIBRATED = "uncalibrated"
NOT_YET_EFFECTIVE = "not_yet_effective"
WRONG_DEVICE = "wrong_device"


def _id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} required")
    return value


def _time(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _non_negative(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


@dataclass(frozen=True)
class CalibrationCertificate:
    certificate_id: str
    device_id: str
    effective_at: datetime
    expires_at: datetime
    calibrated_at: datetime | None = None
    tolerance: float | None = None
    series_id: str | None = None
    version: int = 1
    replaces: str | None = None
    recorded_at: datetime | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        _id(self.certificate_id, "certificate_id")
        _id(self.device_id, "device_id")
        if not isinstance(self.version, int) or self.version <= 0:
            raise ValueError("version must be positive")
        effective = _time(self.effective_at, "effective_at")
        expires = _time(self.expires_at, "expires_at")
        calibrated = _time(self.calibrated_at, "calibrated_at") if self.calibrated_at else effective
        recorded = _time(self.recorded_at, "recorded_at") if self.recorded_at else effective
        if calibrated > effective:
            raise ValueError("calibrated_at cannot be after effective_at")
        if expires <= effective:
            raise ValueError("expires_at must be after effective_at")
        object.__setattr__(self, "device_id", self.device_id.strip())
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "calibrated_at", calibrated)
        object.__setattr__(self, "tolerance", _non_negative(self.tolerance, "tolerance"))
        object.__setattr__(self, "series_id", (self.series_id or self.certificate_id).strip())
        object.__setattr__(self, "recorded_at", recorded)
        if self.replaces is not None:
            object.__setattr__(self, "replaces", _id(self.replaces, "replaces"))


@dataclass(frozen=True)
class CertificateRevocation:
    certificate_id: str
    revoked_at: datetime
    recorded_at: datetime | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        _id(self.certificate_id, "certificate_id")
        revoked = _time(self.revoked_at, "revoked_at")
        recorded = _time(self.recorded_at, "recorded_at") if self.recorded_at else revoked
        object.__setattr__(self, "certificate_id", self.certificate_id.strip())
        object.__setattr__(self, "revoked_at", revoked)
        object.__setattr__(self, "recorded_at", recorded)


@dataclass(frozen=True)
class MeasuredDeviation:
    reference: str
    device_id: str
    observed_at: datetime
    error: float = 0.0
    limit: float | None = None
    certificate_id: str | None = None
    recorded_at: datetime | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _id(self.reference, "reference")
        _id(self.device_id, "device_id")
        observed = _time(self.observed_at, "observed_at")
        recorded = _time(self.recorded_at, "recorded_at") if self.recorded_at else observed
        error = float(self.error)
        if not isfinite(error):
            raise ValueError("error must be finite")
        if recorded < observed:
            raise ValueError("recorded_at cannot be before observed_at")
        object.__setattr__(self, "device_id", self.device_id.strip())
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "recorded_at", recorded)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "limit", _non_negative(self.limit, "limit"))
        if self.certificate_id is not None:
            object.__setattr__(self, "certificate_id", _id(self.certificate_id, "certificate_id"))


@dataclass(frozen=True)
class GracePeriod:
    grace_id: str
    device_id: str
    start_at: datetime
    end_at: datetime
    reason: str = ""
    certificate_id: str | None = None
    recorded_at: datetime | None = None

    def __post_init__(self) -> None:
        _id(self.grace_id, "grace_id")
        _id(self.device_id, "device_id")
        start = _time(self.start_at, "start_at")
        end = _time(self.end_at, "end_at")
        recorded = _time(self.recorded_at, "recorded_at") if self.recorded_at else start
        if end <= start:
            raise ValueError("grace end must be after start")
        object.__setattr__(self, "device_id", self.device_id.strip())
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "end_at", end)
        object.__setattr__(self, "recorded_at", recorded)
        if self.certificate_id is not None:
            object.__setattr__(self, "certificate_id", _id(self.certificate_id, "certificate_id"))


@dataclass(frozen=True)
class DeviceReplacement:
    reference: str
    old_device_id: str
    new_device_id: str
    replaced_at: datetime
    recorded_at: datetime | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        _id(self.reference, "reference")
        _id(self.old_device_id, "old_device_id")
        _id(self.new_device_id, "new_device_id")
        if self.old_device_id == self.new_device_id:
            raise ValueError("replacement requires different devices")
        replaced = _time(self.replaced_at, "replaced_at")
        recorded = _time(self.recorded_at, "recorded_at") if self.recorded_at else replaced
        object.__setattr__(self, "old_device_id", self.old_device_id.strip())
        object.__setattr__(self, "new_device_id", self.new_device_id.strip())
        object.__setattr__(self, "replaced_at", replaced)
        object.__setattr__(self, "recorded_at", recorded)


@dataclass(frozen=True)
class CalibrationAssessment:
    device_id: str
    physical_device_id: str
    at: datetime
    certificate: CalibrationCertificate | None
    status: str
    grace: GracePeriod | None
    error: float | None
    within_tolerance: bool | None
    accepted: bool
    reasons: tuple[str, ...]
    as_of: datetime | None = None


@dataclass(frozen=True)
class TimelineEntry:
    at: datetime
    kind: str
    device_id: str
    reference: str
    payload: object


@dataclass(frozen=True)
class CalibrationDue:
    certificate_id: str
    device_id: str
    version: int
    effective_at: datetime
    expires_at: datetime
    due_at: datetime
    grace_end: datetime | None
    revoked: bool


class CalibrationRegistry:
    """Append-only, bitemporal view of certificates, deviations, grace and replacements."""

    def __init__(
        self,
        certificates: Iterable[CalibrationCertificate] = (),
        deviations: Iterable[MeasuredDeviation] = (),
        graces: Iterable[GracePeriod] = (),
        replacements: Iterable[DeviceReplacement] = (),
        revocations: Iterable[CertificateRevocation] = (),
    ):
        self._lock = RLock()
        self._certificates: list[CalibrationCertificate] = []
        self._deviations: list[MeasuredDeviation] = []
        self._graces: list[GracePeriod] = []
        self._replacements: list[DeviceReplacement] = []
        self._revocations: list[CertificateRevocation] = []
        for row in certificates:
            self._store_certificate(row)
        for row in revocations:
            self._store_revocation(row, check_certificate=False)
        for row in deviations:
            self._store_deviation(row, check_references=False)
        for row in graces:
            self._store_grace(row, check_references=False)
        for row in replacements:
            self._check_replacement(row, self._certificates, self._replacements)
            self._replacements.append(row)
        self._validate_integrity()

    def issue_certificate(self, row: CalibrationCertificate) -> CalibrationCertificate:
        with self._lock:
            self._check_certificate_relation(row, {c.certificate_id: c for c in self._certificates})
            self._store_certificate(row)
            return row

    def revoke_certificate(self, row: CertificateRevocation) -> CertificateRevocation:
        with self._lock:
            self._store_revocation(row, check_certificate=True)
            return row

    def record_deviation(self, row: MeasuredDeviation) -> MeasuredDeviation:
        with self._lock:
            self._store_deviation(row, check_references=True)
            return row

    def grant_grace(self, row: GracePeriod) -> GracePeriod:
        with self._lock:
            self._store_grace(row, check_references=True)
            return row

    def replace_device(self, row: DeviceReplacement) -> DeviceReplacement:
        with self._lock:
            self._check_replacement(row, self._certificates, self._replacements)
            self._store_replacement(row)
            self._validate_replacement_chains(self._replacements)
            return row

    def certificates(self, device_id: str | None = None, as_of: datetime | None = None) -> tuple[CalibrationCertificate, ...]:
        knowledge = self._as_of(as_of)
        with self._lock:
            rows = list(self._certificates)
        rows = [c for c in rows if self._known(c.recorded_at, knowledge)]
        if device_id is not None:
            lineage = set(self.device_lineage(device_id, knowledge))
            rows = [c for c in rows if c.device_id in lineage]
        return tuple(sorted(rows, key=lambda c: (c.effective_at, c.version, c.certificate_id)))

    def deviations(self, device_id: str | None = None, as_of: datetime | None = None) -> tuple[MeasuredDeviation, ...]:
        knowledge = self._as_of(as_of)
        with self._lock:
            rows = list(self._deviations)
        if device_id is not None:
            lineage = set(self.device_lineage(device_id, knowledge))
            rows = [d for d in rows if d.device_id in lineage]
        rows = [d for d in rows if self._known(d.recorded_at, knowledge)]
        return tuple(sorted(rows, key=lambda d: (d.observed_at, d.reference)))

    def graces(self, device_id: str | None = None, as_of: datetime | None = None) -> tuple[GracePeriod, ...]:
        knowledge = self._as_of(as_of)
        with self._lock:
            rows = list(self._graces)
        if device_id is not None:
            lineage = set(self.device_lineage(device_id, knowledge))
            rows = [g for g in rows if g.device_id in lineage]
        return tuple(sorted((g for g in rows if self._known(g.recorded_at, knowledge)), key=lambda g: (g.start_at, g.grace_id)))

    def replacements(self, device_id: str | None = None, as_of: datetime | None = None) -> tuple[DeviceReplacement, ...]:
        knowledge = self._as_of(as_of)
        with self._lock:
            rows = list(self._replacements)
        rows = [r for r in rows if self._known(r.recorded_at, knowledge)]
        if device_id is not None:
            lineage = set(self.device_lineage(device_id, knowledge))
            rows = [r for r in rows if r.old_device_id in lineage and r.new_device_id in lineage]
        return tuple(sorted(rows, key=lambda r: (r.replaced_at, r.reference)))

    def revocations(self, as_of: datetime | None = None) -> tuple[CertificateRevocation, ...]:
        knowledge = self._as_of(as_of)
        with self._lock:
            rows = list(self._revocations)
        return tuple(sorted((r for r in rows if self._known(r.recorded_at, knowledge)), key=lambda r: (r.revoked_at, r.certificate_id)))

    def device_lineage(self, device_id: str, as_of: datetime | None = None) -> tuple[str, ...]:
        knowledge = self._as_of(as_of)
        outgoing, incoming = self._replacement_maps(knowledge)
        current = device_id
        while current in incoming:
            current = incoming[current].old_device_id
        out = []
        while current is not None:
            out.append(current)
            edge = outgoing.get(current)
            current = edge.new_device_id if edge else None
        return tuple(out)

    def replacement_history(self, device_id: str, as_of: datetime | None = None) -> tuple[DeviceReplacement, ...]:
        knowledge = self._as_of(as_of)
        lineage = self.device_lineage(device_id, knowledge)
        edges = {edge.old_device_id: edge for edge in self.replacements(device_id, knowledge)}
        return tuple(edges[old] for old in lineage if old in edges)

    def device_at(self, device_id: str, at: datetime, as_of: datetime | None = None) -> str:
        point = _time(at, "at")
        knowledge = self._as_of(as_of)
        outgoing, incoming = self._replacement_maps(knowledge)
        current = device_id
        while True:
            edge = outgoing.get(current)
            if edge is not None and edge.replaced_at <= point:
                current = edge.new_device_id
                continue
            edge = incoming.get(current)
            if edge is not None and point < edge.replaced_at:
                current = edge.old_device_id
                continue
            return current

    def effective_certificate(
        self,
        device_id: str,
        at: datetime,
        as_of: datetime | None = None,
    ) -> CalibrationCertificate | None:
        point = _time(at, "at")
        knowledge = self._as_of(as_of)
        physical = self.device_at(device_id, point, knowledge)
        certificates, cutoffs, revocations = self._certificate_view(knowledge)
        candidates = []
        for cert in certificates.values():
            if cert.device_id != physical:
                continue
            if cert.effective_at > point:
                continue
            end = self._certificate_end(cert, cutoffs, revocations)
            if end is not None and cert.effective_at <= point < end:
                candidates.append(cert)
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.effective_at, c.version, c.recorded_at, c.certificate_id))

    def assess_reading(
        self,
        device_id: str,
        at: datetime,
        error: float | None = None,
        limit: float | None = None,
        as_of: datetime | None = None,
        certificate_id: str | None = None,
    ) -> CalibrationAssessment:
        point = _time(at, "at")
        knowledge = self._as_of(as_of)
        physical = self.device_at(device_id, point, knowledge)
        certificates, cutoffs, revocations = self._certificate_view(knowledge)
        cert = None
        if certificate_id is not None:
            cert = certificates.get(certificate_id)
            if cert is None:
                raise KeyError(certificate_id)
        if cert is None:
            cert = self.effective_certificate(device_id, point, knowledge)
        if cert is None:
            latest = self._latest_certificate(physical, point, [c for c in certificates.values()])
            if latest is not None and latest.effective_at <= point:
                cert = latest
        return self._assess_certificate(cert, physical, point, error, limit, knowledge, certificates, cutoffs, revocations)

    def assess_deviation(self, row: MeasuredDeviation, as_of: datetime | None = None) -> CalibrationAssessment:
        knowledge = self._as_of(as_of)
        if knowledge is not None and row.recorded_at > knowledge:
            raise ValueError("deviation was not known at as_of")
        known = {cert.certificate_id for cert in self.certificates(as_of=knowledge)}
        if row.certificate_id is not None and row.certificate_id not in known:
            raise KeyError(row.certificate_id)
        return self.assess_reading(
            row.device_id,
            row.observed_at,
            error=row.error,
            limit=row.limit,
            as_of=knowledge,
            certificate_id=row.certificate_id,
        )

    def replay(
        self,
        deviations: Iterable[MeasuredDeviation] | None = None,
        as_of: datetime | None = None,
    ) -> tuple[CalibrationAssessment, ...]:
        knowledge = self._as_of(as_of)
        rows = list(deviations) if deviations is not None else list(self._deviations)
        if knowledge is not None:
            rows = [row for row in rows if row.recorded_at <= knowledge]
        rows.sort(key=lambda d: (d.observed_at, d.reference))
        return tuple(self.assess_deviation(row, knowledge) for row in rows)

    def history(self, device_id: str, as_of: datetime | None = None) -> tuple[TimelineEntry, ...]:
        knowledge = self._as_of(as_of)
        lineage = set(self.device_lineage(device_id, knowledge))
        entries: list[TimelineEntry] = []
        with self._lock:
            certificates = list(self._certificates)
            deviations = list(self._deviations)
            graces = list(self._graces)
            replacements = list(self._replacements)
            revocations = list(self._revocations)
        for cert in certificates:
            if cert.device_id in lineage and self._known(cert.recorded_at, knowledge):
                entries.append(TimelineEntry(cert.effective_at, "certificate", cert.device_id, cert.certificate_id, cert))
        for row in revocations:
            cert = next((c for c in certificates if c.certificate_id == row.certificate_id), None)
            if cert is not None and cert.device_id in lineage and self._known(row.recorded_at, knowledge):
                entries.append(TimelineEntry(row.revoked_at, "revocation", cert.device_id, row.certificate_id, row))
        for row in deviations:
            if row.device_id in lineage and self._known(row.recorded_at, knowledge):
                entries.append(TimelineEntry(row.observed_at, "deviation", row.device_id, row.reference, row))
        for row in graces:
            if row.device_id in lineage and self._known(row.recorded_at, knowledge):
                entries.append(TimelineEntry(row.start_at, "grace", row.device_id, row.grace_id, row))
        for row in replacements:
            if row.old_device_id in lineage and row.new_device_id in lineage and self._known(row.recorded_at, knowledge):
                entries.append(TimelineEntry(row.replaced_at, "replacement", row.new_device_id, row.reference, row))
        return tuple(sorted(entries, key=lambda e: (e.at, e.kind, e.reference)))

    def upcoming_calibrations(
        self,
        *,
        now: datetime,
        lead_time: timedelta,
        horizon: datetime | None = None,
        default_grace: timedelta | None = None,
        as_of: datetime | None = None,
    ) -> tuple[CalibrationDue, ...]:
        current = _time(now, "now")
        knowledge = self._as_of(as_of) if as_of is not None else current
        if not isinstance(lead_time, timedelta) or lead_time < timedelta():
            raise ValueError("lead_time must be non-negative")
        if default_grace is not None and (not isinstance(default_grace, timedelta) or default_grace < timedelta()):
            raise ValueError("default_grace must be non-negative")
        end = _time(horizon, "horizon") if horizon is not None else None
        if end is not None and end < current:
            raise ValueError("horizon cannot be before now")

        certificates, cutoffs, revocations = self._certificate_view(knowledge)
        outgoing, _ = self._replacement_maps(knowledge)
        with self._lock:
            graces = [g for g in self._graces if self._known(g.recorded_at, knowledge)]
        dues: list[CalibrationDue] = []
        for cert in certificates.values():
            revoked_at = revocations.get(cert.certificate_id)
            revoked = revoked_at is not None and revoked_at <= cert.expires_at
            grace_end = None
            if not revoked:
                grace_end = cert.expires_at + default_grace if default_grace is not None else None
                for grace in graces:
                    if grace.device_id != cert.device_id:
                        continue
                    if grace.certificate_id is not None and grace.certificate_id != cert.certificate_id:
                        continue
                    if grace.start_at <= cert.expires_at < grace.end_at:
                        grace_end = max(grace_end or cert.expires_at, grace.end_at)
            coverage_end = max(cert.expires_at, grace_end or cert.expires_at)
            replacement = outgoing.get(cert.device_id)
            if replacement is not None and replacement.replaced_at <= coverage_end:
                continue
            successor_at = cutoffs.get(cert.certificate_id)
            if successor_at is not None and successor_at <= coverage_end:
                continue
            deadline = revoked_at if revoked else cert.expires_at
            if deadline is None:
                continue
            due_at = deadline - lead_time
            if end is not None and due_at > end:
                continue
            dues.append(
                CalibrationDue(
                    cert.certificate_id,
                    cert.device_id,
                    cert.version,
                    cert.effective_at,
                    cert.expires_at,
                    due_at,
                    grace_end,
                    bool(revoked),
                )
            )
        return tuple(sorted(dues, key=lambda d: (d.due_at, d.certificate_id)))

    def calibration_jobs(
        self,
        *,
        now: datetime,
        lead_time: timedelta,
        duration: timedelta,
        zone: str,
        horizon: datetime | None = None,
        default_grace: timedelta | None = None,
        as_of: datetime | None = None,
        setup_group: str = "calibration",
    ) -> tuple[CalibrationJob, ...]:
        current = _time(now, "now")
        if duration <= timedelta():
            raise ValueError("duration must be positive")
        dues = self.upcoming_calibrations(
            now=current,
            lead_time=lead_time,
            horizon=horizon,
            default_grace=default_grace,
            as_of=as_of,
        )
        jobs = []
        for due in dues:
            latest = max(current + duration, due.due_at)
            priority = 100 if due.due_at < current else 10
            jobs.append(
                CalibrationJob(
                    key=due.certificate_id,
                    earliest=current,
                    latest=latest,
                    duration=duration,
                    zone=zone,
                    priority=priority,
                    setup_group=setup_group,
                )
            )
        return tuple(jobs)

    def plan_calibrations(self, resources: Iterable, **kwargs):
        return build_schedule(self.calibration_jobs(**kwargs), resources)

    def _store_certificate(self, row: CalibrationCertificate) -> None:
        if any(c.certificate_id == row.certificate_id for c in self._certificates):
            raise ValueError("duplicate certificate_id")
        self._certificates.append(row)

    def _store_revocation(self, row: CertificateRevocation, check_certificate: bool) -> None:
        if check_certificate and not any(c.certificate_id == row.certificate_id for c in self._certificates):
            raise KeyError(row.certificate_id)
        if any(r.certificate_id == row.certificate_id for r in self._revocations):
            raise ValueError("certificate already revoked")
        self._revocations.append(row)

    def _store_deviation(self, row: MeasuredDeviation, check_references: bool) -> None:
        if any(d.reference == row.reference for d in self._deviations):
            raise ValueError("duplicate deviation reference")
        if check_references:
            devices = {c.device_id for c in self._certificates} | self._replacement_devices()
            if row.device_id not in devices:
                raise ValueError("unknown device_id")
            if row.certificate_id is not None:
                cert = next((c for c in self._certificates if c.certificate_id == row.certificate_id), None)
                if cert is None:
                    raise KeyError(row.certificate_id)
                if cert.device_id != row.device_id:
                    raise ValueError("deviation certificate belongs to another device")
        self._deviations.append(row)

    def _store_grace(self, row: GracePeriod, check_references: bool) -> None:
        if any(g.grace_id == row.grace_id for g in self._graces):
            raise ValueError("duplicate grace_id")
        if check_references:
            devices = {c.device_id for c in self._certificates} | self._replacement_devices()
            if row.device_id not in devices:
                raise ValueError("unknown device_id")
            if row.certificate_id is not None:
                cert = next((c for c in self._certificates if c.certificate_id == row.certificate_id), None)
                if cert is None:
                    raise KeyError(row.certificate_id)
                if cert.device_id != row.device_id:
                    raise ValueError("grace certificate belongs to another device")
        self._graces.append(row)

    def _store_replacement(self, row: DeviceReplacement) -> None:
        if any(r.reference == row.reference for r in self._replacements):
            raise ValueError("duplicate replacement reference")
        if any(r.old_device_id == row.old_device_id for r in self._replacements):
            raise ValueError("device already replaced once")
        if any(r.new_device_id == row.new_device_id for r in self._replacements):
            raise ValueError("replacement device already has a predecessor")
        self._replacements.append(row)

    def _replacement_devices(self) -> set[str]:
        out = set()
        for row in self._replacements:
            out.add(row.old_device_id)
            out.add(row.new_device_id)
        return out

    @staticmethod
    def _check_replacement(
        row: DeviceReplacement,
        certificates: list[CalibrationCertificate],
        replacements: list[DeviceReplacement],
    ) -> None:
        if any(r.reference == row.reference for r in replacements):
            raise ValueError("duplicate replacement reference")
        if any(r.old_device_id == row.old_device_id for r in replacements):
            raise ValueError("device already replaced once")
        if any(r.new_device_id == row.new_device_id for r in replacements):
            raise ValueError("replacement device already has a predecessor")
        if not any(c.device_id == row.old_device_id and c.effective_at <= row.replaced_at for c in certificates):
            raise ValueError("device replaced before its first calibration certificate")

    def _validate_integrity(self) -> None:
        by_id = {c.certificate_id: c for c in self._certificates}
        for cert in self._certificates:
            self._check_certificate_relation(cert, by_id)
        for row in self._revocations:
            cert = by_id.get(row.certificate_id)
            if cert is None:
                raise KeyError(row.certificate_id)
            if row.revoked_at < cert.effective_at or row.revoked_at > cert.expires_at:
                raise ValueError("revocation outside certificate validity")
        devices = {c.device_id for c in self._certificates} | self._replacement_devices()
        certificates_by_device: dict[str, list[CalibrationCertificate]] = {}
        for cert in self._certificates:
            certificates_by_device.setdefault(cert.device_id, []).append(cert)
        for row in self._deviations:
            if row.device_id not in devices:
                raise ValueError("unknown deviation device")
            if row.certificate_id is not None:
                cert = by_id.get(row.certificate_id)
                if cert is None or cert.device_id != row.device_id:
                    raise ValueError("deviation references invalid certificate")
        for row in self._graces:
            if row.device_id not in devices:
                raise ValueError("unknown grace device")
            if row.certificate_id is not None:
                cert = by_id.get(row.certificate_id)
                if cert is None or cert.device_id != row.device_id:
                    raise ValueError("grace references invalid certificate")
        for row in self._replacements:
            old_certificates = certificates_by_device.get(row.old_device_id, ())
            if not any(c.effective_at <= row.replaced_at for c in old_certificates):
                raise ValueError("device replaced before its first calibration certificate")
        self._validate_replacement_chains(self._replacements)

    @staticmethod
    def _check_certificate_relation(row: CalibrationCertificate, by_id: dict[str, CalibrationCertificate]) -> None:
        if row.replaces is None:
            return
        old = by_id.get(row.replaces)
        if old is None:
            raise KeyError(row.replaces)
        if old.device_id != row.device_id:
            raise ValueError("replacement certificate belongs to another device")
        if row.version <= old.version:
            raise ValueError("replacement certificate version must increase")
        seen = set()
        cursor = old.certificate_id
        while cursor:
            if cursor == row.certificate_id or cursor in seen:
                raise ValueError("certificate replacement cycle")
            seen.add(cursor)
            current = by_id.get(cursor)
            cursor = current.replaces if current is not None else None

    @staticmethod
    def _validate_replacement_chains(rows: list[DeviceReplacement]) -> None:
        outgoing = {row.old_device_id: row for row in rows}
        for start in outgoing:
            seen = set()
            current = start
            while current in outgoing:
                if current in seen:
                    raise ValueError("device replacement cycle")
                seen.add(current)
                current = outgoing[current].new_device_id

    def _replacement_maps(self, as_of: datetime | None) -> tuple[dict[str, DeviceReplacement], dict[str, DeviceReplacement]]:
        with self._lock:
            rows = list(self._replacements)
        rows = [row for row in rows if self._known(row.recorded_at, as_of)]
        return {r.old_device_id: r for r in rows}, {r.new_device_id: r for r in rows}

    def _certificate_view(
        self, as_of: datetime | None
    ) -> tuple[dict[str, CalibrationCertificate], dict[str, datetime], dict[str, datetime]]:
        with self._lock:
            certificate_rows = list(self._certificates)
            revocation_rows = list(self._revocations)
        certificates = {
            c.certificate_id: c
            for c in certificate_rows
            if self._known(c.recorded_at, as_of)
        }
        cutoffs: dict[str, datetime] = {}
        for cert in certificates.values():
            old_id = cert.replaces
            seen = set()
            while old_id and old_id in certificates and old_id not in seen:
                seen.add(old_id)
                previous = cutoffs.get(old_id)
                cutoffs[old_id] = cert.effective_at if previous is None else min(previous, cert.effective_at)
                old_id = certificates[old_id].replaces
        revocations: dict[str, datetime] = {}
        for row in revocation_rows:
            if not self._known(row.recorded_at, as_of):
                continue
            if row.certificate_id in certificates:
                previous = revocations.get(row.certificate_id)
                revocations[row.certificate_id] = row.revoked_at if previous is None else min(previous, row.revoked_at)
        return certificates, cutoffs, revocations

    @staticmethod
    def _certificate_end(
        cert: CalibrationCertificate,
        cutoffs: dict[str, datetime],
        revocations: dict[str, datetime],
    ) -> datetime | None:
        end = cert.expires_at
        cutoff = cutoffs.get(cert.certificate_id)
        if cutoff is not None:
            end = min(end, cutoff)
        revoked = revocations.get(cert.certificate_id)
        if revoked is not None:
            end = min(end, revoked)
        return end if end > cert.effective_at else None

    def _latest_certificate(
        self,
        physical_device: str,
        at: datetime,
        rows: list[CalibrationCertificate],
    ) -> CalibrationCertificate | None:
        candidates = [c for c in rows if c.device_id == physical_device and c.effective_at <= at]
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.effective_at, c.version, c.recorded_at, c.certificate_id))

    def _grace_for(
        self,
        physical_device: str,
        cert: CalibrationCertificate,
        at: datetime,
        as_of: datetime | None,
    ) -> GracePeriod | None:
        with self._lock:
            rows = list(self._graces)
        candidates = [
            row
            for row in rows
            if self._known(row.recorded_at, as_of)
            and row.device_id == physical_device
            and row.start_at <= at < row.end_at
            and (row.certificate_id is None or row.certificate_id == cert.certificate_id)
        ]
        return max(candidates, key=lambda g: (g.end_at, g.start_at, g.grace_id), default=None)

    def _assess_certificate(
        self,
        cert: CalibrationCertificate | None,
        physical_device: str,
        at: datetime,
        error: float | None,
        limit: float | None,
        as_of: datetime | None,
        certificates: dict[str, CalibrationCertificate],
        cutoffs: dict[str, datetime],
        revocations: dict[str, datetime],
    ) -> CalibrationAssessment:
        status = UNCALIBRATED
        grace = None
        reasons: list[str] = []
        if cert is None:
            reasons.append("no effective calibration certificate")
        elif cert.device_id != physical_device:
            status = WRONG_DEVICE
            reasons.append("certificate belongs to another device in replacement chain")
        elif at < cert.effective_at:
            status = NOT_YET_EFFECTIVE
            reasons.append("certificate not yet effective")
        else:
            end = self._certificate_end(cert, cutoffs, revocations)
            revoked_at = revocations.get(cert.certificate_id)
            successor_at = cutoffs.get(cert.certificate_id)
            if end is not None and at < end:
                status = VALID
            elif successor_at is not None and at >= successor_at:
                status = SUPERSEDED
                reasons.append("certificate superseded")
            elif revoked_at is not None and at >= revoked_at:
                status = REVOKED
                reasons.append("certificate revoked")
            elif at >= cert.expires_at:
                status = EXPIRED
                reasons.append("certificate expired")
                grace = self._grace_for(physical_device, cert, at, as_of)
                if grace is not None:
                    status = GRACE
                    reasons = ["within calibration grace period"]
            else:
                status = EXPIRED
                reasons.append("certificate not effective")

        within = None
        if error is not None:
            number = float(error)
            if not isfinite(number):
                raise ValueError("error must be finite")
            tolerance = limit if limit is not None else (cert.tolerance if cert is not None else None)
            tolerance = _non_negative(tolerance, "tolerance")
            if tolerance is None:
                within = None
            else:
                within = abs(number) <= tolerance
                if not within:
                    reasons.append(f"measured deviation {number:g} exceeds tolerance {tolerance:g}")
        accepted = status in {VALID, GRACE} and within is not False
        return CalibrationAssessment(
            cert.device_id if cert is not None else physical_device,
            physical_device,
            at,
            cert,
            status,
            grace,
            None if error is None else float(error),
            within,
            accepted,
            tuple(reasons),
            as_of,
        )

    @staticmethod
    def _as_of(value: datetime | None) -> datetime | None:
        return None if value is None else _time(value, "as_of")

    @staticmethod
    def _known(recorded_at: datetime, as_of: datetime | None) -> bool:
        return as_of is None or recorded_at <= as_of


CalibrationLedger = CalibrationRegistry
