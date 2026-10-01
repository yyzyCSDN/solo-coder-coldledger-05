from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable, TypeVar

T = TypeVar("T")


def utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return v.astimezone(timezone.utc)


@dataclass(frozen=True)
class Bitemporal:
    """Two independent time axes.

    valid_from/valid_to : the physical interval the fact claims to describe.
    recorded_from      : when the fact became known to the system.
    recorded_to        : when the assertion was retracted/corrected; None == current.
    """

    valid_from: datetime
    recorded_from: datetime
    valid_to: datetime | None = None
    recorded_to: datetime | None = None

    def norm(self) -> "Bitemporal":
        vf, rf = utc(self.valid_from), utc(self.recorded_from)
        vt = utc(self.valid_to) if self.valid_to is not None else None
        rt = utc(self.recorded_to) if self.recorded_to is not None else None
        if vt is not None and vt <= vf:
            raise ValueError("valid_to must be after valid_from")
        if rt is not None and rt < rf:
            raise ValueError("recorded_to must not precede recorded_from")
        return Bitemporal(vf, rf, vt, rt)

    def known_at(self, as_of: datetime) -> bool:
        """Assertion was on the books at time as_of (record axis only)."""
        t = utc(as_of)
        b = self.norm()
        return b.recorded_from <= t and (b.recorded_to is None or t < b.recorded_to)

    def covers(self, at: datetime) -> bool:
        """The fact's claimed validity interval contains at (valid axis only)."""
        t = utc(at)
        b = self.norm()
        return b.valid_from <= t and (b.valid_to is None or t < b.valid_to)


def as_of(records: Iterable[T], at: datetime, *, known: str = "known_at", covers: str | None = None) -> list[T]:
    """Filter records that were asserted at ``at`` and optionally cover it.

    A later-recorded fact never leaks into an earlier as_of view: this is what
    lets an old shipment be replayed under the rules known at the time.
    """
    t = utc(at)
    out = []
    for r in records:
        b = getattr(r, known) if isinstance(getattr(r, known, None), Bitemporal) else r
        if not b.known_at(t):
            continue
        if covers is not None and not b.covers(t):
            continue
        out.append(r)
    return out


def current_assertions(records: Iterable[T], *, known: str = "known_at") -> list[T]:
    return [r for r in records if getattr(r, known).norm().recorded_to is None]


def retract(record: T, at: datetime, *, known: str = "known_at") -> T:
    b = getattr(record, known).norm()
    t = utc(at)
    if t < b.recorded_from:
        raise ValueError("cannot retract before the assertion existed")
    return replace(record, **{known: replace(b, recorded_to=t)})
