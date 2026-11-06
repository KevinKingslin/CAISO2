"""Constitution-guided inference uses DICAI paper Figure 16, blind to labels."""

from caiso.constitution.models import PreferencePair

from .reconstruction import apply_constitution


class ConstitutionJudge:
    def __init__(self, providers, role="constitution_judge", temperature=0.0):
        self.providers, self.role = providers, role
        self.temperature = temperature

    def choose(self, constitution, a, b, seed):
        pair = PreferencePair(
            pair_id="inference", context={}, candidate_a=a, candidate_b=b, preferred="A"
        )  # Placeholder is never sent to the judge.
        return apply_constitution(
            self.providers, self.role, pair, constitution.principles, seed, self.temperature
        )[0]
