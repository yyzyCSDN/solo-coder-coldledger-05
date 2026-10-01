from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from ..core.bitemporal import Bitemporal, as_of, utc


@dataclass(frozen=True)
class Replacement:
    """old_device removed at ``at`` and replaced by new_device from that instant."""

    old_device: str
    new_device: str
    at: datetime
    reason: str
    interval: Bitemporal
    shipment_scope: str | None = None

    def validate(self) -> None:
        if self.old_device == self.new_device:
            raise ValueError("replacement needs distinct devices")
        b = self.interval.norm()
        if utc(self.at) != b.valid_from:
            raise ValueError("replacement instant must anchor the valid axis")
        if self.reason not in {"recalibration", "failure", "upgrade", "recall", "other"}:
            raise ValueError("unknown replacement reason")


def _known(records: Iterable[Replacement], asof: datetime) -> list[Replacement]:
    return as_of(records, asof, known="interval")


def active_device(records: Iterable[Replacement], device_id: str, asof: datetime) -> str:
    """Physical device serving the logical id at ``asof``, following the chain.

    Only replacements known by ``asof`` are followed: a swap discovered during a
    later investigation does not rewrite what was believed during the shipment.
    """
    t = utc(asof)
    current = device_id
    while True:
        step = [
            r
            for r in _known(records, asof)
            if r.old_device == current
            and utc(r.at) <= t
            and (r.shipment_scope is None)
        ]
        if not step:
            return current
        current = min(step, key=lambda r: (utc(r.at), r.interval.recorded_from)).new_device


def chain_for(records: Iterable[Replacement], device_id: str, asof: datetime | None = None) -> list[str]:
    rows = list(records) if asof is None else _known(records, asof)
    order = sorted(rows, key=lambda r: utc(r.at))
    chain = [device_id]
    current = device_id
    for r in order:
        if r.old_device == current:
            chain.append(r.new_device)
            current = r.new_device
    return chain


def replacements_touching(records: Iterable[Replacement], device_id: str) -> list[Replacement]:
    return sorted(
        (r for r in records if device_id in {r.old_device, r.new_device}),
        key=lambda r: (utc(r.at), r.interval.recorded_from),
    )


def gap_windows(records: Iterable[Replacement], asof: datetime) -> list[tuple[str, datetime, datetime]]:
    """Intervals a device sat past expiry before its replacement was recorded."""
    out = []
    for r in _known(records, asof):
        b = r.interval.norm()
        if b.recorded_from > utc(r.at):
            out.append((r.old_device, utc(r.at), b.recorded_from))
    return sorted(out, key=lambda x: x[1])
