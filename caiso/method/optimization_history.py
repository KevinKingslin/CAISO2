"""Validation evidence for scheduling blocks and avoiding rejected state transitions."""

from typing import Any

from pydantic import Field

from caiso.core.serialization import Record, plain


class OptimizationAttempt(Record):
    iteration: int
    members: list[str]
    context: str
    candidate: str | None = None
    configurations: dict[str, Any] = Field(default_factory=dict)
    previous_configurations: dict[str, Any] = Field(default_factory=dict)
    outcome: str
    current_score: float | None = None
    candidate_score: float | None = None
    improvement: float | None = None
    accepted: bool = False


class OptimizationHistory(Record):
    attempts: list[OptimizationAttempt] = Field(default_factory=list)

    def record_validation(self, iteration, state, candidate, members, gate):
        self.attempts.append(
            OptimizationAttempt(
                iteration=iteration,
                members=sorted(members),
                context=state.fingerprint(),
                candidate=candidate.fingerprint(),
                configurations=plain(
                    {cid: candidate.configurations[cid] for cid in sorted(members)}
                ),
                previous_configurations=plain(
                    {cid: state.configurations[cid] for cid in sorted(members)}
                ),
                outcome="validated",
                **gate.model_dump(),
            )
        )

    def record_no_candidate(self, iteration, state, members):
        self.attempts.append(
            OptimizationAttempt(
                iteration=iteration,
                members=sorted(members),
                context=state.fingerprint(),
                outcome="no_new_candidate",
            )
        )

    def rejected_candidates(self, state):
        context = state.fingerprint()
        return {
            attempt.candidate
            for attempt in self.attempts
            if attempt.context == context
            and attempt.outcome == "validated"
            and not attempt.accepted
        }

    def filter_candidates(self, state, candidates):
        rejected = self.rejected_candidates(state)
        return [
            candidate
            for candidate in candidates
            if state.with_configs(candidate.configurations).fingerprint() not in rejected
        ]

    def selection_summary(self, state, blocks, recent_limit=5):
        """Map evidence by membership, even when the partition's labels change.

        Old contexts remain advisory; only identical full system states permit
        automatic exclusion. Prompt text is bounded, while checkpoints retain it all.
        """
        context = state.fingerprint()
        result = {}
        for bid, members in blocks.items():
            relevant = [a for a in self.attempts if set(a.members) & set(members)]
            current = [a for a in relevant if a.context == context]
            other = [a for a in relevant if a.context != context]
            recent = []
            evidence = current[-recent_limit:]
            # Include a little older evidence too, especially previously accepted gains.
            remaining = recent_limit - len(evidence)
            if remaining > 0:
                evidence.extend(other[-remaining:])
            for attempt in evidence:
                item = attempt.model_dump(exclude={"context", "candidate"})
                item["same_system_context"] = attempt.context == context
                # Keep full discrete settings, but bound potentially long prompts.
                configs = [
                    *item["configurations"].values(),
                    *item["previous_configurations"].values(),
                ]
                for config in configs:
                    value = config.get("value")
                    if isinstance(value, str) and len(value) > 400:
                        config["value"] = value[:400] + " [truncated]"
                recent.append(item)
            result[bid] = {
                "members": sorted(members),
                "current_context_attempts": len(current),
                "validated_rejections": sum(
                    a.outcome == "validated" and not a.accepted for a in current
                ),
                "no_new_candidate_attempts": sum(a.outcome == "no_new_candidate" for a in current),
                "other_context_attempts": len(relevant) - len(current),
                "recent_attempts": recent,
            }
        return result
