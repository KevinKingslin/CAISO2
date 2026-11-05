"""Explicit schema-valid baselines; never infer replacements from model outputs."""

from caiso.core.workflow import Baseline


def text_baseline(neutral=""):
    return Baseline(value=neutral)


def retrieval_baseline(as_list=False):
    return Baseline(value=[] if as_list else "")


def passthrough_baseline(input_field="question"):
    return Baseline(policy="passthrough", input_field=input_field)


def structured_baseline(default):
    return Baseline(value=default)
