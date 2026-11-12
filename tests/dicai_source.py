"""Extract complete requests from the preserved original AST, without importing it."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/democratic_icai/original"


def token(name):
    return "<<" + name + ">>"


def extract_requests():
    trees = {p.stem: ast.parse(p.read_text()) for p in SOURCE.glob("*.py")}
    constants = {}
    for tree in trees.values():
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            else:
                continue
            try:
                constants[target] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    expressions = {
        "pair.prompt": "context:json", "pair.response_a": "response_a:json", "pair.response_b": "response_b:json",
        "pair.preferred.upper()": "preferred", "pair.preferred_text()": "preferred_text:json",
        "self.samples": "samples", "joined": "analyses", "stable": "stable_dimensions",
        "current.reasoning": "reasoning", "current.dimensions": "dimensions", "critique": "critique",
        "render_list(questions)": "working_list", "status": "reconstruction_status",
        "cycle": "cycle", "max_cycles": "total_cycles", "min_principles": "min_principles",
        "len(questions)": "question_count", "_history(transcript)": "debate_history",
        "_history(transcript,n=len(debaters))": "cycle_arguments",
        "principles": "constitution", "first": "response_1:json", "second": "response_2:json",
    }

    def named(tree, name, kind=ast.FunctionDef):
        return next(n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name)

    def assigned(tree, name):
        return next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

    def evaluate(node, context=None):
        context = context or {}
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return context[node.id] if node.id in context else constants[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left, context) + evaluate(node.right, context)
        if isinstance(node, ast.JoinedStr):
            parts = []
            for part in node.values:
                if isinstance(part, ast.Constant):
                    parts.append(part.value)
                else:
                    expression = ast.unparse(part.value).replace(" ", "")
                    parts.append(context[expression] if expression in context else token(expressions[expression]))
            return "".join(parts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_pair_block":
            return pair
        raise AssertionError(ast.dump(node))

    pair = evaluate(next(n.value for n in ast.walk(named(trees['reasoning'], '_pair_block')) if isinstance(n, ast.Return)))
    story = evaluate(next(n.value for n in ast.walk(named(trees['debate'], '_story_block')) if isinstance(n, ast.Return)))
    base = constants['_BASE_TASK'].format(pref=token('preferred'))
    requests = {}
    for cls_name, key in [('CoTReasoner', 'reasoning.cot'), ('CoTSCReasoner', 'reasoning.self_consistency')]:
        method = named(named(trees['reasoning'], cls_name, ast.ClassDef), 'reason')
        requests[key] = [('system', base), ('user', evaluate(assigned(method, 'user')))]
        if cls_name == 'CoTSCReasoner':
            requests['reasoning.self_consistency_consensus'] = [('system', base), ('user', evaluate(assigned(method, 'synth_user')))]
    method = named(named(trees['reasoning'], 'SelfRefineReasoner', ast.ClassDef), 'reason')
    calls = sorted((n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'chat'), key=lambda n: n.lineno)
    for key, call in zip(('reasoning.self_refine', 'reasoning.self_refine_critique', 'reasoning.self_refine_revision'), calls, strict=True):
        messages = []
        for row in call.args[0].elts:
            fields = {k.value: v for k, v in zip(row.keys, row.values, strict=True)}
            messages.append((evaluate(fields['role']), evaluate(fields['content'], {'sys': base, 'block': pair})))
        requests[key] = messages
    method = named(trees['debate'], '_debate')
    notes = sorted((n.value for n in ast.walk(method) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'note' for t in n.targets)), key=lambda n: n.lineno)
    for suffix, rejection in zip(('', '_reconstruction_failed', '_too_few'), ('', evaluate(notes[0]), evaluate(notes[1])), strict=True):
        context = {'stories': story, 'rejection': rejection}
        requests['debate.debater' + suffix] = [
            ('system', constants['_DEBATER_SYS'].format(name=token('agent_name'), identity=token('identity'), trace=token('trace'), seeds=token('seeds'))),
            ('user', evaluate(assigned(method, 'prompt'), context)),
        ]
        requests['debate.chairperson' + suffix] = [
            ('system', constants['_JUDGE_SYS'].format(min_principles=token('min_principles'))),
            ('user', evaluate(assigned(method, 'j_prompt'), context)),
        ]
    requests['reconstruction'] = [('system', constants['_RECON_SYS']), ('user', evaluate(assigned(named(trees['common'], 'apply_constitution'), 'user')))]
    requests['abstraction'] = [('system', constants['_ABSTRACT_SYS']), ('user', constants['_ABSTRACT_USER'].format(principles=token('principles')))]
    requests['audit'] = [('user', constants['AUDIT_PROMPT'].format(principle=token('principle')))]
    return requests


def template(messages):
    def encode(text):
        text = text.replace('{', '{{').replace('}', '}}')
        return re.sub(r'<<([^<>]+)>>', lambda m: '{' + m[1] + '}', text)
    return '\n'.join('[' + role.upper() + ']\n' + encode(text) for role, text in messages)


def trajectory_adaptation(key, original):
    general = (
        'Express the reusable quality that explains the preference. '
        'Exclude sample-specific names of people, places, organizations and works, '
        'and example-specific dates and identifiers from principles. '
        'Describe their semantic roles instead, while preserving actual component IDs. '
        'The principle should remain applicable when those entities change.'
    )
    text = original.replace('A human annotator preferred response', 'The provided preference label favors trajectory')
    for old, new in [('human preference', 'provided preference'), ('human-preferred', 'preferred'), ('a human preferred', 'the provided preference label favors'), ('(human choice)', '(provided preference label)'), ('the human preferred', 'the provided preference label favors')]:
        text = text.replace(old, new)
    for old, new in [('responses', 'trajectories'), ('response', 'trajectory'), ('RESPONSE', 'TRAJECTORY'), ('stories', 'trajectories'), ('story', 'trajectory'), ('Story', 'Trajectory'), ('STORY', 'TRAJECTORY')]:
        text = re.sub(r'(?<![A-Za-z_])' + old + r'(?![A-Za-z_])', new, text)
    if key.startswith('reasoning.') and key != 'reasoning.self_refine_critique':
        text = text.replace(
            'State the reason as something reusable, not a description of these exact texts.',
            'State the reason as something reusable, not a description of these exact texts. '
            'Examine each component in the trajectories. Where an observable difference helps explain the preference, propose a separate principle naming that component\'s exact component_id. '
            'Every question must evaluate one observable behavior of named components, using exact component_id values from the trajectories; final-output questions must name the producing component. '
            + general.replace('from principles.', 'from proposed questions and dimensions.'),
        )
        text = text.replace(
            '"select_prompt": "Select the trajectory that ..."',
            '"select_prompt": "Select the trajectory where ..."',
        )
    if key == 'reasoning.self_refine_critique':
        text = text.replace(
            'Be brief.',
            'Also flag overlooked component differences that help explain the preference, and sample-specific names or facts used as criteria or dimensions. '
            + general.replace('from principles.', 'from proposed criteria and dimensions.') + ' Be brief.',
        )
    if key.startswith('debate.debater'):
        text = text.replace('DEBATE RULES (must follow):', 'Proposed questions may concern individual component behavior, interactions between components, or the final output. Every proposed question must evaluate one observable behavior using exact component_id values from the trajectories, including the producing component for final-output questions. Examine each component in the trajectories. Where an observable difference helps explain the preference, propose a separate principle naming that component\'s exact component_id.\n\n' + general + '\n\nDEBATE RULES (must follow):')
        text = text.replace('The final list must reconstruct the provided preference.', 'The final list must reconstruct the provided preference through reusable criteria.')
    if key.startswith('debate.chairperson'):
        text = text.replace(
            '- Apply edits if they improve clarity.\n',
            '- Apply edits if they improve clarity.\n'
            '- Every question must name the exact component_id from the trajectories for the component or components whose behavior it evaluates, including the producing component for final-output questions. Preserve these IDs when editing or merging.\n'
            '- Retain supported principles for different components. Do not remove a component\'s only supported principle merely to shorten the list.\n'
            '- ' + general + '\n',
        )
        text = text.replace('Select the trajectory ...\n', 'Select the trajectory where ...\n')
        text = text.replace('RECONSTRUCTION CHECK (mandatory, overrides everything else):', 'RECONSTRUCTION CHECK (mandatory):\nReconstruct the preference using reusable component-behavior criteria.')
    if key == 'abstraction':
        text = text.replace('Each has the form "Select the trajectory that <condition>".', 'Each begins with "Select the trajectory" and names the component whose behavior it evaluates.')
        text = text.replace('Use the exact format: "Select the trajectory that <condition>."', 'Begin with "Select the trajectory where", followed by the actual component ID and the reusable condition.')
        text = text.replace('\nCLUSTER:', '\nPreserve the single quality shared by the principles. ' + general + ' Name the component or components whose behavior the shared rule evaluates, using exact component IDs from the member principles. Do not replace exact component IDs with generic references such as "final answer", "rewritten query", or "retrieved content".\n\nCLUSTER:')
    if key.startswith('debate.'):
        component_rule = (
            'Write every principle beginning exactly with "Select the trajectory where", '
            'immediately followed by the actual component_id and one observable, reusable criterion. '
            'Include at least one separate principle for every component shown in the trajectories '
            '(for a block comparison, every block member identified in the task). '
            'Inspect each component and propose a missing principle grounded in its observed behavior. '
            'Keep principles for different components separate even when they evaluate the same quality. '
            'An interaction principle does not replace the participating components\' own principles. '
            'Do not remove a component\'s only supported principle for brevity. '
            'If the evidence does not support a component principle, identify that gap in the debate rather than inventing behavior.'
        )
        text = text.replace('"Select the trajectory"', '"Select the trajectory where"')
        text = text.replace('\n[USER]\n', '\n\nCOMPONENT PRINCIPLES:\n' + component_rule + '\n[USER]\n', 1)
    if key.startswith('debate.debater'):
        text = text.replace(
            '- A question true of BOTH trajectories discriminates nothing and must be attacked.',
            '- A question true of BOTH trajectories does not explain this pair\'s preference. '
            'Retain it only when it provides the sole evidence-grounded principle for a component; '
            'the list must also retain decisive criteria that reconstruct the preference.',
        )
    if key.startswith('debate.chairperson'):
        text = text.replace(
            'select the non-preferred trajectory, remove questions true of both trajectories, and',
            'select the non-preferred trajectory, remove redundant questions true of both trajectories '
            'while preserving each component\'s sole evidence-grounded principle, and',
        )
    if key == 'abstraction':
        component_rule = (
            'Preserve component coverage during abstraction. The constitution should retain a principle '
            'for every component represented in the source principles. '
            'Begin exactly with "Select the trajectory where", followed immediately by the actual component_id '
            'whose behavior the shared criterion evaluates. For a cluster about one component, preserve that '
            'component as the subject. For a cluster spanning different components, explicitly retain every '
            'evaluated component ID and its role in the single shared condition; do not silently narrow the '
            'rule to one component or replace IDs with generic aliases. '
            'Keep one shared quality and return exactly one principle for this cluster.'
        )
        text = text.replace('\nCLUSTER:', '\n' + component_rule + '\n\nCLUSTER:')
    if key.startswith('debate.') or key == 'abstraction':
        source = 'the trajectory execution records' if key.startswith('debate.') else 'the source principles in this cluster'
        format_rule = (
            'REQUIRED PRINCIPLE FORMAT:\n'
            'Select the trajectory where <component_id> <criterion>.\n\n'
            'Replace <component_id> with the exact component_id from ' + source + '. '
            'Replace <criterion> with one observable, reusable behavior or quality of that component. '
            'The actual component ID must appear immediately after "where", not only later in the sentence. '
            'Do not use generic subjects such as "the final answer", "the system", or "the component". '
            'Do not return the literal placeholders <component_id> or <criterion>. '
            'Before publishing, rewrite any principle that omits the actual component ID in that position.'
        )
        text = text.replace('Select the trajectory where ...', 'Select the trajectory where <component_id> <criterion>.')
        text = text.replace('Select the trajectory...', 'Select the trajectory where <component_id> <criterion>.')
        text = text.replace('Select the trajectory that is', 'Select the trajectory where <component_id> is')
        text = text.replace('\n[USER]\n', '\n\n' + format_rule + '\n[USER]\n', 1)
    if key.startswith(('reasoning.', 'debate.')):
        score_rule = (
            'TRAINING SCORE FEEDBACK:\n'
            'When training task feedback is supplied, use both absolute scores and their difference '
            'to explain observable trajectory differences associated with the measured advantage. '
            'Scores measure end-to-end benchmark performance: a higher score does not establish '
            'that an answer is entirely correct, the score gap is not a confidence estimate, and '
            'it does not establish which component caused the outcome. '
            'Produce reusable component-behavior principles that can be applied without reference '
            'answers, scores or preference labels. Never use "prefer the higher score" as a principle. '
            'If the relation is tie, there is no measured task-metric advantage; do not claim that '
            'the scores support the supplied A/B label. The supplied label and existing reconstruction '
            'requirement are retained. If feedback is null, use the supplied preference and trajectory '
            'evidence as before.'
        )
        text = text.replace('\n[USER]\n', '\n\n' + score_rule + '\n[USER]\n', 1)
        text += '\n\nTRAINING TASK FEEDBACK (A/B order, training only):\n{score_feedback:json}'
    return text
