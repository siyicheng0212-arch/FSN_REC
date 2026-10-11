from copy import deepcopy
import json

import pytest

from tools.check_public_release import ROOT, check_privacy_text, validate_results


def payload():
    return (json.loads((ROOT/"reports/results.json").read_text()),
            json.loads((ROOT/"reports/epoch_curves.json").read_text()))


def test_all_actual_seven_class_counts_and_complete_epoch_budgets():
    result, curves = payload()
    assert validate_results(result, curves) == "asformer_interval_self_reference"


@pytest.mark.parametrize("mutation", ["count", "f1", "accuracy", "step", "epoch", "independence", "extra"])
def test_inconsistent_or_unapproved_results_rejected(mutation):
    result, curves = deepcopy(payload())
    chosen = result["variants"]["asformer_interval_self_reference"]
    if mutation == "count":
        chosen["metrics"]["confusion_matrix"][0][0] += 1
    elif mutation == "f1":
        chosen["metrics"]["macro_f1"] += .001
    elif mutation == "accuracy":
        chosen["metrics"]["accuracy"] += .001
    elif mutation == "step":
        curves["variants"]["asformer_interval_self_reference"]["curve"][0][4] += 1
    elif mutation == "epoch":
        chosen["resources"]["epochs_completed"] += 1
    elif mutation == "independence":
        result["independent_test"] = True
    else:
        result["unexpected_raw_payload"] = []
    with pytest.raises(ValueError):
        validate_results(result, curves)


def test_deployment_path_and_binary_rejected():
    for value in ("/"+"root/"+"private", "/"+"Users/"+"private", "\x00"):
        with pytest.raises(ValueError):
            check_privacy_text(value)
    check_privacy_text("aggregate confusion_matrix and generic clip_id schema")
