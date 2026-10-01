from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

from ..core.bitemporal import Bitemporal, as_of, utc


@dataclass(frozen=True)
class GraceRule:
    """A versioned post-expiry grace allowance.

    scope is 'default', or ('device', id) / ('group', name). Rules themselves are
    bitemporal: a grace change never rewrites how an older shipment was judged.
    """

    rule_id: str
    scope: str
    grace: timedelta
    interval: Bitemporal

    def validate(self) -> None:
        if self.grace < timedelta():
            raise ValueError("grace cannot be negative")
        kind, _, value = self.scope.partition(":")
        if kind not in {"default", "device", "group"}:
            raise ValueError("unknown grace scope")
        if kind != "default" and not value:
            raise ValueError("scoped grace needs a value")
        self.interval.norm()

    def matches(self, device_id: str, groups: frozenset[str] = frozenset()) -> bool:
        kind, _, value = self.scope.partition(":")
        if kind == "device":
            return value == device_id
        if kind == "group":
            return value in groups
        return kind == "default"


def _rank(scope: str) -> int:
    return {"device": 0, "group": 1, "default": 2}.get(scope.partition(":")[0], 3)


def applicable_grace(
    rules: Iterable[GraceRule],
    device_id: str,
    asof,
    groups: frozenset[str] = frozenset(),
    at=None,
) -> timedelta:
    """Grace for an event at ``at`` under the rules known at ``asof``.

    The rule must be valid at the event time but already recorded by the
    decision time, so a grace change made after a shipment cannot rewrite how
    that shipment was judged. ``at`` defaults to ``asof`` (live decisions).
    """
    know = utc(asof)
    t = utc(at) if at is not None else know
    known = as_of(rules, know, known="interval")
    covering = [r for r in known if r.interval.covers(t) and r.matches(device_id, groups)]
    if not covering:
        return timedelta()
    best = min(covering, key=lambda r: (_rank(r.scope), -r.grace.total_seconds(), r.rule_id))
    return best.grace


def grace_history(rules: Iterable[GraceRule], scope: str) -> list[GraceRule]:
    return sorted((r for r in rules if r.scope == scope), key=lambda r: r.interval.recorded_from)
