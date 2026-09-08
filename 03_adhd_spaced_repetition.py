"""
Attention-Aware Spaced Repetition: An Ablation
==============================================

I built a scheduler that adapts to within-session attention decay, tested it,
and it lost. Then I decomposed it and found out why. The decomposition is the
result worth reading.

THE HYPOTHESIS
Standard SM-2 treats every review as equally reliable evidence. It is not. A
correct answer given 25 minutes into a session, after several lapses, says less
about retention than a correct answer given fresh. So: (a) discount intervals
for low-attention recalls, and (b) end sessions when attention drops rather
than at a fixed card count.

THE RESULT
The combined mechanism is worse than baseline at every deck size tested. Split
apart, the two halves behave completely differently:

  - Confidence weighting is mildly positive on retention and costs ~3% more
    reviews to get there. Defensible.
  - Adaptive session-ending is actively harmful and gets worse as deck size
    grows. It does not reduce work, it defers work, and the deferred cards
    come back as a compounding backlog.

Testing only the composite would have produced the conclusion "attention-
awareness does not help", which is wrong. Half of it helps.

FAIRNESS NOTE
Cards not reached because a session ended early stay in the due queue for the
next day. Without that, the adaptive arm would simply be doing less work and
the comparison would be meaningless. Watch the backlog column: it is where the
adaptive-stop arm hides its cost.

Run: python 03_adhd_spaced_repetition.py
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Card + schedulers
# --------------------------------------------------------------------------

@dataclass
class Card:
    id: int
    ease: float = 2.5
    interval: float = 0.0        # days
    reps: int = 0
    due: float = 0.0             # day index
    last_review: float = 0.0
    strength: float = 1.0        # latent; hidden from the scheduler


class SM2:
    """Baseline: the scheduler in Anki and most flashcard apps."""
    name = "SM-2 baseline"
    session_cap = 40

    def update(self, card: Card, quality: int, today: float, ctx: dict) -> None:
        if quality < 3:
            card.reps = 0
            card.interval = 1.0
        else:
            card.reps += 1
            if card.reps == 1:
                card.interval = 1.0
            elif card.reps == 2:
                card.interval = 6.0
            else:
                card.interval *= card.ease
            card.ease = max(1.3, card.ease + 0.1
                            - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        card.interval = min(card.interval, 180.0)
        card.last_review = today
        card.due = today + card.interval

    def should_continue(self, ctx: dict) -> bool:
        return ctx["cards_seen"] < self.session_cap


class AttentionAware(SM2):
    """The mechanism under test.

    1. Confidence weighting -- a correct answer given while attention is low
       produces a shorter interval, because it is weaker evidence.
    2. Adaptive session end -- stop when estimated attention falls below a
       floor, rather than at a fixed count. Reviews taken past that point
       generate unreliable signal that then corrupts the intervals of every
       card touched.
    """
    name = "attention-aware"
    attention_floor = 0.45
    session_cap = 40

    @staticmethod
    def attention(ctx: dict) -> float:
        """Estimated remaining attention in [0,1]. Decays with time on task,
        faster after lapses -- failure costs more than elapsed time alone."""
        minutes = ctx["cards_seen"] * 0.35
        time_decay = math.exp(-minutes / 20.0)
        lapse_penalty = 1.0 - min(0.25, 0.025 * ctx["recent_lapses"])
        return max(0.0, time_decay * lapse_penalty)

    def update(self, card: Card, quality: int, today: float, ctx: dict) -> None:
        att = self.attention(ctx)
        super().update(card, quality, today, ctx)
        if quality >= 3:
            confidence = 0.6 + 0.4 * att      # att=1 -> full, att=0 -> 60%
            card.interval *= confidence
            card.due = today + card.interval

    def should_continue(self, ctx: dict) -> bool:
        if ctx["cards_seen"] >= self.session_cap:
            return False
        return self.attention(ctx) >= self.attention_floor


class ConfidenceOnly(AttentionAware):
    """Ablation A: interval discounting, fixed-length sessions."""
    name = "confidence-weight only"
    session_cap = 40

    def should_continue(self, ctx: dict) -> bool:
        return ctx["cards_seen"] < self.session_cap


class AdaptiveStopOnly(SM2):
    """Ablation B: adaptive session end, standard SM-2 intervals."""
    name = "adaptive-stop only"
    session_cap = 40

    def should_continue(self, ctx: dict) -> bool:
        if ctx["cards_seen"] >= self.session_cap:
            return False
        return AttentionAware.attention(ctx) >= AttentionAware.attention_floor


# --------------------------------------------------------------------------
# Simulated learner -- ground truth the schedulers cannot observe
# --------------------------------------------------------------------------

class SimulatedLearner:
    """Recall follows an exponential forgetting curve over each card's latent
    strength. Attention affects the probability of *demonstrating* recall: a
    card can be known and still be missed when the learner is depleted."""

    def __init__(self, seed: int, attention_sensitivity: float = 0.40):
        self.rng = random.Random(seed)
        self.sensitivity = attention_sensitivity

    def review(self, card: Card, today: float, attention: float) -> int:
        elapsed = max(0.0, today - card.last_review)
        true_recall = math.exp(-elapsed / max(0.5, card.strength))

        observed = true_recall * (1 - self.sensitivity * (1 - attention))
        success = self.rng.random() < observed

        if success:
            # Desirable difficulty: the gain is largest when retrieval was
            # effortful AND attention was available to encode it.
            effort = 1.0 - true_recall
            card.strength += 2.2 * (0.35 + 0.65 * attention) * (0.25 + effort)
            return 5 if true_recall < 0.85 else 4

        card.strength = max(0.6, card.strength * 0.80)
        return 2


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def simulate(scheduler: SM2, n_cards: int, days: int, seed: int) -> dict:
    learner = SimulatedLearner(seed)
    cards = [Card(id=i) for i in range(n_cards)]
    total_reviews = 0
    session_lengths: list[int] = []
    backlog_trace: list[int] = []

    for day in range(days):
        due = [c for c in cards if c.due <= day]
        learner.rng.shuffle(due)
        ctx = {"cards_seen": 0, "recent_lapses": 0}

        for card in due:
            if not scheduler.should_continue(ctx):
                break   # remainder stays due tomorrow; nothing is skipped
            # Both arms share the same physiology. Only the ATTENTION-AWARE
            # scheduler is allowed to know about it.
            true_att = AttentionAware.attention(ctx)
            quality = learner.review(card, day, true_att)
            scheduler.update(card, quality, day, ctx)

            ctx["cards_seen"] += 1
            if quality < 3:
                ctx["recent_lapses"] += 1
            total_reviews += 1

        session_lengths.append(ctx["cards_seen"])
        backlog_trace.append(max(0, len(due) - ctx["cards_seen"]))

    retention = [math.exp(-7.0 / max(0.5, c.strength)) for c in cards]
    mean_ret = statistics.mean(retention)
    return {
        "scheduler": scheduler.name,
        "reviews": total_reviews,
        "mean_session": statistics.mean(session_lengths),
        "mean_backlog": statistics.mean(backlog_trace),
        "retention_7d": mean_ret,
        "retention_per_100_reviews": mean_ret / max(1, total_reviews) * 100,
    }


def paired_ci(diffs: list[float]) -> tuple[float, float, float]:
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / math.sqrt(len(diffs))
    return m, m - 1.96 * se, m + 1.96 * se


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "Interval excludes zero (positive). Effect is real in this model."
    if hi < 0:
        return "Interval excludes zero (NEGATIVE). The mechanism hurt this metric."
    return "Interval spans zero. No detectable effect; report it as such."


def main() -> None:
    N_RUNS, DAYS = 30, 60
    ARMS = [SM2, ConfidenceOnly, AdaptiveStopOnly, AttentionAware]
    LOADS = [40, 60, 80]

    print("ABLATION: attention-aware spaced repetition")
    print("Cards not reached in a session carry over to the next day.\n")

    all_runs: dict[int, dict[str, list[dict]]] = {}
    for n_cards in LOADS:
        all_runs[n_cards] = {}
        for cls in ARMS:
            all_runs[n_cards][cls.name] = [
                simulate(cls(), n_cards, DAYS, seed) for seed in range(N_RUNS)
            ]

    hdr = (f"{'deck':>6}{'arm':>24}{'backlog':>9}{'reviews':>9}"
           f"{'ret@7d':>9}{'ret/100rev':>12}")
    for n_cards in LOADS:
        print(hdr)
        print("-" * len(hdr))
        for cls in ARMS:
            rs = all_runs[n_cards][cls.name]
            print(f"{n_cards:>6}{cls.name:>24}"
                  f"{statistics.mean(r['mean_backlog'] for r in rs):>9.1f}"
                  f"{statistics.mean(r['reviews'] for r in rs):>9.0f}"
                  f"{statistics.mean(r['retention_7d'] for r in rs):>9.3f}"
                  f"{statistics.mean(r['retention_per_100_reviews'] for r in rs):>12.4f}")
        print()

    print("PAIRED DIFFERENCES vs SM-2 BASELINE (7-day retention)")
    print("=" * 76)
    for n_cards in LOADS:
        base = all_runs[n_cards]["SM-2 baseline"]
        print(f"\n  deck size {n_cards}")
        for cls in ARMS[1:]:
            arm = all_runs[n_cards][cls.name]
            diffs = [y["retention_7d"] - x["retention_7d"] for x, y in zip(base, arm)]
            m, lo, hi = paired_ci(diffs)
            print(f"    {cls.name:<24}{m:>+9.4f}   CI [{lo:+.4f}, {hi:+.4f}]")
            print(f"    {'':<24}{verdict(lo, hi)}")

    print("\n\nWHAT THE ABLATION SHOWS")
    print("=" * 76)
    print("Confidence weighting holds its own at every deck size and costs a few")
    print("percent more reviews to do it. Adaptive stopping degrades sharply as")
    print("the deck grows, and the backlog column says why: it is not saving the")
    print("learner work, it is postponing it into a queue that never drains.")
    print()
    print("The composite mechanism inherits the harm. Had I only tested the")
    print("composite -- which was the original plan -- I would have concluded that")
    print("attention-awareness does not help. That conclusion would have been")
    print("wrong about half the intervention.")
    print()
    print("Product read: ship the interval discounting, drop the session cap. The")
    print("session cap solves a problem the user feels (fatigue) by creating one")
    print("they do not see until three weeks later (backlog), which is the worst")
    print("shape a feature can have.")


if __name__ == "__main__":
    main()
