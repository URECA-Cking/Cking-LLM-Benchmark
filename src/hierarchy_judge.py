"""Resume relevance judging for frozen hierarchical candidates."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from src import flow_eval as flow
from src.flow_experiment import score_cases
from src.clients.openai_judge import OpenAIJudge,BINARY_SCHEMA
from src.judge import pair_text_hash


def execute(args):
    """Report full-input scores only when every needed pair is present."""
    output=args.output_dir
    plan=json.loads((output/'hierarchy_plan.json').read_text())
    candidates=json.loads((output/'candidates.json').read_text())
    pairs=json.loads((output/'evaluation_pairs.json').read_text())
    baseline=json.loads((args.baseline_dir/'evaluation_plan.json').read_text())
    if flow.digest(baseline)!=plan['baseline_hash'] or flow.digest(candidates)!=plan['candidate_hash']:
        raise ValueError('Frozen baseline or candidate content changed')
    hashes={key:pair_text_hash(row['input'],row['candidate']) for key,row in pairs.items()}
    if hashes!=plan['pair_hashes']:raise ValueError('Frozen judging evidence changed')
    frozen=json.loads((args.flow_dir/'plan.json').read_text())
    if frozen['hash']!=flow.digest(frozen['plan']) or frozen['hash']!=baseline['base_hash']:
        raise ValueError('Input case plan changed')
    cases=[case for case in frozen['plan']['cases'] if case['id'] in baseline['case_ids']]
    if len(cases)!=plan['evaluation_inputs']:raise ValueError('Missing evaluation inputs')
    contract={'hierarchy_hash':flow.digest(plan),'model':baseline['model'],'creator_prompt':baseline['creator_prompt'],
              'other_prompt':baseline['other_prompt'],'temperature':baseline['temperature']}
    cache_path=output/'judgments.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {'contract_hash':flow.digest(contract),'scores':json.loads((output/'reused_judgments.json').read_text())}
    if cache['contract_hash']!=flow.digest(contract):raise ValueError('Judgment settings changed')
    scores=cache['scores']
    for key,value in scores.items():
        if key not in hashes or value['hash']!=hashes[key] or type(value['score']) is not int or value['score'] not in (0,1):
            raise ValueError('Invalid or stale judgment')
    pending=[(key,row) for key,row in sorted(pairs.items()) if key not in scores]
    print(f'[hierarchy-judge] total={len(pairs)} cached={len(scores)} new_calls={len(pending)}',flush=True)
    if args.judge and pending:
        client=OpenAI(timeout=60,max_retries=2)
        judges={route:OpenAIJudge(model=contract['model'],client=client,system_prompt=contract['creator_prompt'] if route=='creators' else contract['other_prompt'],schema=BINARY_SCHEMA,temperature=contract['temperature']) for route in ('tags','creators','favorites')}
        def work(item):
            key,row=item
            result=judges[row['route']].judge(row['input'],row['candidate'])
            return key,{'score':result.score,'hash':hashes[key],'source':'new_hierarchy_llm','input_tokens':result.input_tokens,'output_tokens':result.output_tokens}
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            for offset in range(0,len(pending),20):
                futures=[executor.submit(work,item) for item in pending[offset:offset+20]];error=None
                for future in futures:
                    try:key,value=future.result();scores[key]=value
                    except Exception as exc:error=exc
                cache_path.write_text(json.dumps(cache,ensure_ascii=False,indent=2))
                print(f'[hierarchy-judge] saved={len(scores)}/{len(pairs)} elapsed={time.monotonic()-start:.0f}s',flush=True)
                if error:
                    print(f'[hierarchy-judge] stopped: {type(error).__name__} status={getattr(error,"status_code",None)}',flush=True);break
    cache_path.write_text(json.dumps(cache,ensure_ascii=False,indent=2))
    complete=len(scores)==len(pairs)
    summary={'status':'llm_complete_human_unverified' if complete else 'pending_no_full_sample_results',
             'inputs':len(cases),'pairs':len(pairs),'judged_pairs':len(scores),'pending_pairs':len(pairs)-len(scores),
             'groups':score_cases(cases,candidates,scores,frozen['plan']['top_k'],frozen['plan']['seed']) if complete else {},
             'new_tokens':{name:sum(v.get(name,0) for v in scores.values() if v.get('source')=='new_hierarchy_llm') for name in ('input_tokens','output_tokens')},
             'limitations':['Prespecified threshold/weight sensitivity, not gold optimized','Shared inputs, exploratory intervals','No self-selected candidate tags','No human agreement','Synthetic favorites']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','limitations')}),flush=True)


def main():
    """A paid call requires the explicit --judge flag."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('flow-dir','baseline-dir','output-dir'):parser.add_argument('--'+name,type=lambda v:Path(v).expanduser(),required=True)
    parser.add_argument('--judge',action='store_true');execute(parser.parse_args())


if __name__=='__main__':main()
