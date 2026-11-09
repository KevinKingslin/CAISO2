"""The supplied code is the source of truth for complete trajectory requests."""

import hashlib
import json
from pathlib import Path

import pytest

from caiso.constitution.dicai.abstraction import abstract_clusters
from caiso.constitution.dicai.debate import debate
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.dicai.reasoning_assembly import assemble
from caiso.constitution.dicai.reconstruction import apply_constitution, parse_choice
from caiso.constitution.models import PreferencePair
from caiso.core.recovery import ModelResponseError
from caiso.prompts import catalog_entries, render_messages
from tests.dicai_source import extract_requests, template, trajectory_adaptation
from tests.prompt_requests import PRINCIPLES, Recorder

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'third_party/democratic_icai/PROMPTS.json').read_text())


def pair(preferred='A'):
    return PreferencePair(
        pair_id='source-test', preferred=preferred, context={'question': 'Q', 'gold': 'EVALUATION_ONLY'},
        candidate_a={'executions': [{'component_id': 'answer_generator', 'component_input': {'hints': 'Detailed hints'}, 'component_output': {'answer': 'A'}}], 'final_output': {'answer': 'A'}, 'reward': 1},
        candidate_b={'executions': [{'component_id': 'hint_generator', 'component_input': {'question': 'Q'}, 'component_output': {'hints': 'Detailed hints'}}], 'final_output': {'answer': 'B'}, 'reward': 0},
    )


def test_original_source_files_are_preserved_byte_for_byte():
    for name, expected in MANIFEST['files'].items():
        assert hashlib.sha256((ROOT / 'third_party/democratic_icai/original' / name).read_bytes()).hexdigest() == expected


def test_every_complete_original_request_is_extracted_from_the_source_ast():
    requests = extract_requests()
    assert set(requests) == set(MANIFEST['requests']) == set(catalog_entries('dicai'))
    for key, messages in requests.items():
        original = (ROOT / MANIFEST['requests'][key]['original']).read_text()
        assert original == template(messages), key
        active = (ROOT / MANIFEST['requests'][key]['active']).read_text()
        assert active == trajectory_adaptation(key, original), key
        assert hashlib.sha256(active.encode()).hexdigest() == MANIFEST['requests'][key]['active_sha256']
        assert 'Select the response' not in active
        assert '<component_id>' not in active
        if 'Select the trajectory' in original:
            pytest.fail('The original reference was overwritten with trajectory wording')


@pytest.mark.parametrize('samples,steps,counts', [(5, 2, [1, 6, 5]), (2, 1, [1, 3, 3])])
def test_original_reasoning_stages_return_their_own_questions(samples, steps, counts):
    recorder = Recorder()
    personas = assign_personas(['First', 'Second', 'Third'], 42)
    catalog = assemble(pair(), personas, recorder, 'audit', 42, {'reasoning_workers': 1, 'self_consistency_samples': samples, 'self_refine_steps': steps})
    assert [row['calls_before_retries'] for row in catalog] == counts
    assert len(recorder.calls) == sum(counts)
    assert all(row['principles'] == PRINCIPLES for row in catalog)
    assert all(row['dimensions'] for row in catalog)
    assert not any('seed_questions' in str(call['messages']) for call in recorder.calls)
    for call in recorder.calls:
        assert 'EVALUATION_ONLY' not in str(call['messages'])
        assert '"reward"' not in str(call['messages'])
    assert catalog[2]['trace'][1]['stage'] == 'self_refine_critique'
    assert catalog[2]['trace'][2]['stage'] == 'self_refine_revision'


def test_sampled_consensus_uses_only_recurring_dimensions():
    from caiso.constitution.dicai.reasoning_assembly import Reasoning

    class Provider(Recorder):
        sample = 0
        def complete(self, *args, **kwargs):
            result = super().complete(*args, **kwargs)
            stage = kwargs['response_context']['stage']
            if stage == 'reasoning.self_consistency':
                result.dimensions = ['support'] if self.sample < 3 else ['novelty' if self.sample == 3 else 'specificity']
                self.sample += 1
            elif stage == 'reasoning.self_consistency_consensus':
                assert "['support']" in args[1][1]['content']
                result = Reasoning(reasoning='Consensus', dimensions=['unsupported drift'], questions=PRINCIPLES)
            return result
    catalog = assemble(pair(), assign_personas(['First', 'Second', 'Third'], 42), Provider(), 'audit', 42, {'reasoning_workers': 1, 'self_refine_steps': 0})
    assert catalog[1]['dimensions'] == ['support']


def test_consensus_ties_keep_first_seen_order_and_count_an_axis_once_per_chain():
    class Provider(Recorder):
        def complete(self, *args, **kwargs):
            result = super().complete(*args, **kwargs)
            stage = kwargs['response_context']['stage']
            if stage == 'reasoning.self_consistency':
                result.dimensions = ['z_quality', 'a_quality', 'z_quality']
            elif stage == 'reasoning.self_consistency_consensus':
                assert "['z_quality', 'a_quality']" in args[1][1]['content']
            return result
    catalog = assemble(pair(), assign_personas(['First', 'Second', 'Third'], 42), Provider(), 'audit', 42, {'reasoning_workers': 1, 'self_refine_steps': 0})
    assert catalog[1]['dimensions'] == ['z_quality', 'a_quality']


@pytest.mark.parametrize('seed,expected', [(0, 'A'), (1, 'B')])
def test_original_json_judge_is_blind_and_maps_shuffled_choices(seed, expected):
    recorder = Recorder()
    choice, detail = apply_constitution(recorder, 'audit', pair(), PRINCIPLES, seed=seed)
    assert choice == expected
    assert json.loads(detail['response'])['choice'] == '1'
    messages = recorder.calls[0]['messages']
    assert 'EVALUATION_ONLY' not in str(messages)
    assert '"reward"' not in str(messages)
    assert 'PRINCIPLES:' in messages[1]['content']
    assert 'TRAJECTORY 1:' in messages[1]['content']
    assert '"question": "Q"' in messages[1]['content']


@pytest.mark.parametrize('value', ['Output(a)', '{"choice":"3"}', '{"choice":true}', '{}', '{"choice":"1"} {"choice":"2"}'])
def test_invalid_source_judge_choices_fail(value):
    with pytest.raises(ModelResponseError):
        parse_choice(value)


@pytest.mark.parametrize('atomicity_enabled', [True, False])
def test_successful_reconstruction_allows_early_finalization(atomicity_enabled):
    recorder = Recorder()
    catalog = assemble(pair(), assign_personas(['First', 'Second', 'Third'], 42), recorder, 'audit', 42, {'reasoning_workers': 1})
    recorder.calls.clear()
    principles, history = debate(pair(), catalog, recorder, 'audit', 42, rounds=3, config={'atomicity': {'enabled': atomicity_enabled}})
    assert principles == PRINCIPLES
    assert history[-1] == {'speaker': 'debate_outcome', 'cycles': 1, 'finalized': True, 'reconstructed': True}
    assert len([r for r in history if r['speaker'] == 'reconstruction']) == 2
    assert len(recorder.calls) == 6 + atomicity_enabled
    assert 'reasoning trace' in recorder.calls[1]['messages'][0]['content']


def test_failed_reconstruction_is_fed_back_and_cannot_finalize():
    recorder = Recorder()
    catalog = assemble(pair('B'), assign_personas(['First', 'Second', 'Third'], 42), recorder, 'audit', 42, {'reasoning_workers': 1})
    recorder.calls.clear()
    principles, history = debate(pair('B'), catalog, recorder, 'audit', 42, rounds=2)
    assert principles == PRINCIPLES  # Original source returns a best-effort list at the cap.
    assert history[-1] == {'speaker': 'debate_outcome', 'cycles': 2, 'finalized': False, 'reconstructed': False}
    assert any('DEBATE_OVER REJECTED' in str(call['messages']) for call in recorder.calls)
    assert len([h for h in history if h['speaker'] == 'reconstruction']) == 4


def test_minimum_question_gate_uses_the_source_feedback_template():
    class Provider(Recorder):
        def complete(self, *args, **kwargs):
            if 'You are the Moderator/Judge.' in args[1][0]['content']:
                self.calls.append({'messages': args[1]})
                result = '- Q1: ' + PRINCIPLES[0] + '\nDEBATE_OVER'
                return kwargs['validate'](result)
            return super().complete(*args, **kwargs)
    recorder = Provider()
    catalog = assemble(pair(), assign_personas(['First', 'Second', 'Third'], 42), recorder, 'audit', 42, {'reasoning_workers': 1})
    _, history = debate(pair(), catalog, recorder, 'audit', 42, rounds=2)
    assert not history[-1]['finalized']
    assert any('fewer than the required 3' in str(call['messages']) for call in recorder.calls)


def test_singleton_clusters_retain_their_principle_without_an_abstraction_call():
    recorder = Recorder()
    assert abstract_clusters([PRINCIPLES[0]], [0], recorder, 'audit', 42) == [PRINCIPLES[0]]
    assert recorder.calls == []


def test_original_audit_prompt_is_available_without_enabling_an_audit_stage():
    messages = render_messages('dicai', 'audit', principle=PRINCIPLES[0])
    assert [m['role'] for m in messages] == ['user']
    assert 'Axis 1 - Demographic / Stereotype Bias.' in messages[0]['content']
    assert 'Axis 2 - Spurious / Superficial Criteria.' in messages[0]['content']
    assert 'Axis 3 - Harmfulness / Safety.' in messages[0]['content']
