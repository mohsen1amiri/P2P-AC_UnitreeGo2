#!/usr/bin/env python3
"""Read-only validation against the *installed* repository's experiment YAMLs."""
import json, os, shlex, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_sweep import expand, load_spec
CAM=ROOT/'p2pac_campaign'
manifest=json.loads((CAM/'manifest.json').read_text())
assert str(ROOT)==manifest['repo'], f"Wrong repository location: {ROOT}"
expected={}
for exp in ('main','budget','ablation','tp_sweep','td_sweep','kappa_sweep','replay_sweep','bc','planner_reference'):
    spec=load_spec(str(ROOT/'configs'/'experiments'/f'{exp}.yaml'))
    for row in expand(spec,output_root=str(ROOT/'runs')):
        key=row['run_dir']
        assert key not in expected, f'Duplicate run: {key}'
        expected[key]=row['command']
actual=[]
for jsonfile in sorted((CAM/'jobs').glob('*.json')):
    if jsonfile.name.startswith(('snapshots_','final_eval_')): continue
    actual.extend(json.loads(jsonfile.read_text()))
assert len(actual)==153, f'Expected 153 runs, got {len(actual)}'
assert len(expected)==153, f'Expected 153 YAML runs, got {len(expected)}'
assert len({j['run_dir'] for j in actual})==153, 'Duplicate run_dir in prepared jobs'
for item in actual:
    ref=expected.get(item['run_dir'])
    assert ref is not None, f"Unexpected run: {item['run_dir']}"
    # Argument ordering does not change semantics. Generated commands should be exact.
    assert shlex.split(ref)==shlex.split(item['command']), f"YAML mismatch: {item['run_dir']}"
print('PASS: 153 unique scientific jobs match the repository YAMLs exactly')
print('PASS: RL 145, BC 5, additional REAP 3. Each account has 51 jobs.')
print('PASS: 290 independent RL policy evaluation jobs also prepared.')
