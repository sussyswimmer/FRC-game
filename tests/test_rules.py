"""Rules engine checks, including the cases observed on the 2026 Istanbul Regional broadcast
(docs/01-video-analysis.md section 5.6)."""

import pytest

from rebuilt_sim import constants as C
from rebuilt_sim.constants import Alliance
from rebuilt_sim.rules import (
    AllianceScore,
    FoulLedger,
    HubSchedule,
    Period,
    display_clock,
    fuel_counts_for_auto,
    period_at,
    ranking_points,
)


def test_match_timeline():
    assert period_at(0) == Period.AUTO
    assert period_at(19.99) == Period.AUTO
    assert period_at(20) == Period.PAUSE
    assert period_at(23) == Period.TRANSITION
    assert period_at(33) == Period.SHIFT1
    assert period_at(58) == Period.SHIFT2
    assert period_at(83) == Period.SHIFT3
    assert period_at(108) == Period.SHIFT4
    assert period_at(133) == Period.ENDGAME
    assert period_at(163) == Period.POST
    assert period_at(166) == Period.DONE
    assert C.MATCH_END - C.TELEOP_START == 140  # TELEOP is 2:20


def test_display_clock_matches_broadcast():
    assert display_clock(0) == 20
    assert display_clock(21.5) == 140  # clock holds at 2:20 between AUTO and TELEOP
    # broadcast P3: "2:01" with "2/6 :16" -> 16 s left in SHIFT 1
    t = C.MATCH_END - 121
    assert period_at(t) == Period.SHIFT1
    assert C.SHIFT1_START + C.SHIFT_LEN - t == pytest.approx(16)


def test_auto_winner_hub_is_inactive_first():
    # Practice 12: red scored 3 in AUTO, blue 75 -> red HUB active in SHIFT 1
    hs = HubSchedule()
    assert hs.resolve_auto(blue_auto_fuel=75, red_auto_fuel=3, coin=0.9) == Alliance.BLUE
    s1, s2, s3, s4 = (C.SHIFT1_START + k * C.SHIFT_LEN + 1 for k in range(4))
    assert hs.is_active(Alliance.RED, s1) and not hs.is_active(Alliance.BLUE, s1)
    assert hs.is_active(Alliance.BLUE, s2) and not hs.is_active(Alliance.RED, s2)
    assert hs.is_active(Alliance.RED, s3) and not hs.is_active(Alliance.BLUE, s3)
    assert hs.is_active(Alliance.BLUE, s4) and not hs.is_active(Alliance.RED, s4)
    for t in (5, 25, 140):  # AUTO, TRANSITION, END GAME: both on
        assert hs.is_active(Alliance.BLUE, t) and hs.is_active(Alliance.RED, t)


def test_auto_tie_is_random():
    assert HubSchedule().resolve_auto(10, 10, coin=0.2) == Alliance.BLUE
    assert HubSchedule().resolve_auto(10, 10, coin=0.8) == Alliance.RED


def test_every_alliance_gets_90_seconds_of_active_teleop():
    hs = HubSchedule()
    hs.resolve_auto(5, 1, coin=0.0)
    for a in Alliance:
        dt = 0.01
        active = sum(dt for k in range(int(140 / dt)) if hs.is_active(a, C.TELEOP_START + k * dt + dt / 2))
        assert active == pytest.approx(90, abs=0.05)


def test_grace_window_after_deactivation():
    hs = HubSchedule()
    hs.resolve_auto(blue_auto_fuel=50, red_auto_fuel=10, coin=0.5)  # blue off in SHIFT 1
    off = C.SHIFT1_START
    assert hs.counts(Alliance.BLUE, off + 2.9)
    assert not hs.counts(Alliance.BLUE, off + 3.1)
    assert hs.counts(Alliance.RED, off + 10)  # red is active in SHIFT 1


def test_grace_after_match_end_and_auto_credit():
    hs = HubSchedule()
    hs.resolve_auto(1, 2, coin=0.5)
    assert hs.counts(Alliance.BLUE, C.MATCH_END + 2.9)
    assert not hs.counts(Alliance.BLUE, C.MATCH_END + 3.1)
    assert fuel_counts_for_auto(C.AUTO_LEN + 2.9)
    assert not fuel_counts_for_auto(C.AUTO_LEN + 3.1)


def test_next_toggle():
    hs = HubSchedule()
    assert hs.next_toggle(Alliance.BLUE, 5.0) is None  # order not known during AUTO
    hs.resolve_auto(blue_auto_fuel=30, red_auto_fuel=0, coin=0.5)
    assert hs.next_toggle(Alliance.BLUE, 25.0) == C.SHIFT1_START
    assert hs.next_toggle(Alliance.RED, 25.0) == C.SHIFT1_START + C.SHIFT_LEN
    assert hs.next_toggle(Alliance.BLUE, 140.0) is None  # END GAME: stays on


def test_ranking_point_thresholds():
    # Practice 18: red 99 FUEL -> no ENERGIZED
    red = AllianceScore(auto_fuel=6, teleop_fuel=93, penalty_points=15)
    blue = AllianceScore(auto_fuel=10, teleop_fuel=29, teleop_tower=10)
    rp = ranking_points(red, blue)
    assert red.total == 114 and blue.total == 49
    assert rp.win == 3 and rp.energized == 0 and rp.supercharged == 0 and rp.traversal == 0
    assert ranking_points(AllianceScore(teleop_fuel=100), AllianceScore()).energized == 1
    full = ranking_points(AllianceScore(auto_fuel=78, teleop_fuel=301, teleop_tower=50), AllianceScore())
    assert (full.energized, full.supercharged, full.traversal, full.total) == (1, 1, 1, 6)
    tie = ranking_points(AllianceScore(teleop_fuel=5), AllianceScore(teleop_fuel=5))
    assert tie.win == 1
    # District Championship thresholds
    assert ranking_points(AllianceScore(teleop_fuel=200), AllianceScore(), level="dcmp").energized == 0


def test_fouls_credit_the_opponent():
    scores = [AllianceScore(), AllianceScore()]
    ledger = FoulLedger()
    ledger.call(scores, 50.0, Alliance.BLUE, 0, "G418", major=False)
    ledger.call(scores, 60.0, Alliance.BLUE, 0, "G418", major=True)
    assert scores[Alliance.RED].penalty_points == 20
    assert scores[Alliance.BLUE].minor_fouls == 1 and scores[Alliance.BLUE].major_fouls == 1
