from pydantic import Field

from caiso.core.serialization import Record

# Exact attribute labels in DICAI Appendix H, Figures 20-24. Values in CAISO's
# task profiles are original, fictional adaptations for QA/recommendation.
PROFILE_FIELDS = (
    "Age",
    "Sex",
    "Race",
    "Ancestry",
    "Household Language",
    "Education",
    "Employment Status",
    "Class Of Worker",
    "Industry Category",
    "Occupation Category",
    "Detailed Job Description",
    "Income",
    "Marital Status",
    "Household Type",
    "Family Presence And Age",
    "Place Of Birth",
    "Citizenship",
    "Veteran Status",
    "Disability",
    "Health Insurance",
    "Fertility",
    "Hearing Difficulty",
    "Vision Difficulty",
    "Cognitive Difficulty",
    "Ability To Speak English",
    "Big Five Scores",
    "Defining Quirks",
    "Mannerisms",
    "Personal Time",
    "Lifestyle",
    "Ideology",
    "Political Views",
    "Religion",
)
BIG_FIVE_FIELDS = {"Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"}


class Persona(Record):
    identity: str
    strategy: str
    attributes: dict = Field(default_factory=dict)

    def prompt(self):
        lines = [f"{self.identity} Expert Persona"]
        for label in PROFILE_FIELDS:
            if label in self.attributes:
                value = self.attributes[label]
                if isinstance(value, dict):
                    value = ", ".join(f"{k}: {v}" for k, v in value.items())
                lines.append(f"{label}: {value}")
        return "\n".join(lines)


def assign_personas(identities, seed):
    if len(identities) != 3:
        raise ValueError("DICAI appendix procedure requires exactly three expert personas")
    strategies = ["reasoning_cot", "reasoning_self_consistency", "reasoning_self_refine"]
    # Official run_pipeline.load_personas assigns CSV columns in this fixed order.
    personas = []
    for identity, strategy in zip(identities, strategies, strict=True):
        if isinstance(identity, str):  # Support old resolved runs and offline fixtures.
            personas.append(Persona(identity=identity, strategy=strategy))
            continue
        attributes = identity["attributes"]
        missing = set(PROFILE_FIELDS) - attributes.keys()
        extra = attributes.keys() - set(PROFILE_FIELDS)
        if missing or extra:
            raise ValueError(
                f"Persona profile fields: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if set(attributes["Big Five Scores"]) != BIG_FIVE_FIELDS:
            raise ValueError("Big Five Scores must specify all five personality traits")
        personas.append(
            Persona(identity=identity["identity"], attributes=attributes, strategy=strategy)
        )
    return personas
