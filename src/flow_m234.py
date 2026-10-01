"""Compare M2/M3/M4 on identical frozen onboarding and favorite inputs."""
from __future__ import annotations
import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from openai import OpenAI
from src import flow_eval as flow, real_eval as real
from src.flow_experiment import PROMPT, select_pilot, score_cases
from src.clients.openai_judge import OpenAIJudge, BINARY_SCHEMA
from src.config import REAL_BONUS_GRID, LLM_TAG_MAX, OPENAI_JUDGE_MODEL
from src.judge import pair_text_hash
from src.tagging import rank_all
from src.taste_eval import judge_config_hash


def rankings(case, ids, vectors, categories, codes, zero_tags, llm_tags, params, k):
    """Use the same cosine and maximum aggregation; only method tags differ."""
    index = {cid: i for i, cid in enumerate(ids)}
    seeds = case['seeds']
    if len(seeds) != len(set(seeds)) or any(cid not in index for cid in seeds):
        raise ValueError('Unknown or duplicate seeds')
    if case['route'] == 'tags':
        selected = case['tags']
        if seeds or not selected or not set(selected) <= set(codes):
            raise ValueError('Invalid selected tags')
        query = flow.normalized(categories[[codes.index(code) for code in selected]].mean(axis=0))
        cosine = (vectors @ query)[None, :]
        qzero = qllm = [set(selected)]
    elif case['route'] in ('creators', 'favorites'):
        if not seeds or case['tags']:
            raise ValueError('Invalid creator input')
        cosine = vectors[[index[cid] for cid in seeds]] @ vectors.T
        qzero = [zero_tags[cid] for cid in seeds]
        qllm = [llm_tags[cid] for cid in seeds]
    else:
        raise ValueError('Unknown route')
    scores = {'M2': cosine.max(axis=0)}
    for method, query_tags, candidate_tags, bonus in [
        ('M3', qzero, zero_tags, params['bonus_m3']), ('M4', qllm, llm_tags, params['bonus_m4'])]:
        overlap = np.array([[bool(set(tags) & candidate_tags[cid]) for cid in ids] for tags in query_tags])
        scores[method] = (cosine + bonus * overlap).max(axis=0)
    eligible = [i for i, cid in enumerate(ids) if cid not in seeds]
    return {method: [{'id': ids[i], 'score': float(values[i])} for i in sorted(eligible, key=lambda i: (-float(values[i]), ids[i]))[:k]] for method, values in scores.items()}


def evidence(case, data):
    """Build identical blind evidence for every method."""
    channels = {c.id: c for c in data.pool}
    if case['route'] == 'tags':
        return '\n'.join(f'{name}: {description}' for code, name, description in data.categories if code in case['tags'])
    return '\n---\n'.join(channels[cid].text() for cid in case['seeds'])


def reusable(key, query, candidate, data, old, pilot, pilot_contract):
    """Reuse only the same text, model, prompt and source configuration."""
    case_id, cid = key.split('::')
    h = pair_text_hash(query, candidate)
    if case_id.startswith('creator-'):
        qid = case_id.removeprefix('creator-')
        value = old.get(f'{qid}::{cid}')
        expected = judge_config_hash(OPENAI_JUDGE_MODEL, real.BINARY_SYSTEM_PROMPT)
        if value and value.get('hash') == h and value.get('judge') == expected:
            return {**value, 'source': 'existing_creator_llm'}
    else:
        value = pilot.get(key)
        valid_contract = pilot_contract.get('model') == OPENAI_JUDGE_MODEL and pilot_contract.get('prompt') == PROMPT and pilot_contract.get('temperature') == 0
        if valid_contract and value and value.get('hash') == h and pilot_contract.get('pair_text_hashes', {}).get(key) == h:
            return {**value, 'source': 'existing_flow_llm'}
    return None


def execute(args):
    """Prepare frozen comparison, resume optional API judging, report complete groups."""
    data, vectors, categories, llm, previous = flow.load_source(args.data_dir, args.source_dir)
    base = json.loads((args.flow_dir/'plan.json').read_text())
    plan = base['plan']
    if base['hash'] != flow.digest(plan) or plan['source'] != flow.source_snapshot(data,vectors,categories,llm,previous):
        raise ValueError('Frozen source changed')
    params = real.select_parameters(data.pool,vectors,categories,data.codes,data.gold,plan['dev_ids'],llm,LLM_TAG_MAX,REAL_BONUS_GRID)
    zero = {c.id: ranked.assigned(params['tau'], LLM_TAG_MAX) for c,ranked in zip(data.pool, rank_all(vectors,categories,data.codes))}
    ids = [c.id for c in data.pool]
    candidates = {case['id']: rankings(case,ids,vectors,categories,data.codes,zero,llm,params,plan['top_k']) for case in plan['cases']}
    evaluation = select_pilot(plan)+[c for c in plan['cases'] if c['route']=='creators']
    channels={c.id:c for c in data.pool}
    rows={}
    for case in evaluation:
        query=evidence(case,data)
        for selected in candidates[case['id']].values():
            for c in selected:
                key=f'{case["id"]}::{c["id"]}'
                rows[key]={'pair_id':key,'route':case['route'],'input':query,'candidate':channels[c['id']].text(),'score':''}
    contract={'base_hash':base['hash'],'params':params,'case_ids':[c['id'] for c in evaluation],
              'candidate_hash':flow.digest(candidates),'pair_hashes':{key:pair_text_hash(r['input'],r['candidate']) for key,r in sorted(rows.items())},
              'model':OPENAI_JUDGE_MODEL,'creator_prompt':real.BINARY_SYSTEM_PROMPT,'other_prompt':PROMPT,'temperature':0,
              'aggregation':'max_of_per_seed_adjusted_scores','calibration':'transfer_from_fixed_v4_creator_dev_only',
              'decision':'exploratory_M4_minus_M3_at_least_0.05_and_CI_lower_gt_zero; human_gate_0.80_not_verified'}
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    path=output/'evaluation_plan.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:raise ValueError('Comparison contract changed; use a new output directory')
    else:
        flow.write_json(path,contract)
        flow.write_json(output/'candidates.json',{'contract_hash':flow.digest(contract),'candidates':candidates})
        with (output/'human_sheet.csv').open('x',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['pair_id','route','input','candidate','score'])
            writer.writeheader()
            shuffled=list(rows.values()); __import__('random').Random(plan['seed']).shuffle(shuffled)
            writer.writerows(shuffled)
    if json.loads((output/'candidates.json').read_text())!={'contract_hash':flow.digest(contract),'candidates':candidates}:
        raise ValueError('Saved candidate contents changed')
    cache_path=output/'judgments.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {'contract_hash':flow.digest(contract),'scores':{}}
    if cache['contract_hash']!=flow.digest(contract):raise ValueError('Judgment contract mismatch')
    scores=cache['scores']
    old=json.loads((args.source_dir/'judgments.json').read_text())
    pilot_pack=json.loads((args.pilot_dir/'llm_judgments.json').read_text())
    pilot_contract=json.loads((args.pilot_dir/'evaluation_plan.json').read_text())
    if pilot_pack['contract_hash']!=flow.digest(pilot_contract):raise ValueError('Old pilot contract mismatch')
    for key,row in rows.items():
        if key not in scores:
            value=reusable(key,row['input'],row['candidate'],data,old,pilot_pack['judgments'],pilot_contract)
            if value:scores[key]=value
    for key,value in scores.items():
        if key not in rows or value.get('hash')!=contract['pair_hashes'][key] or type(value.get('score')) is not int or value['score'] not in (0,1):
            raise ValueError('Invalid or stale judgment')
    cache_path.write_text(json.dumps(cache,ensure_ascii=False,indent=2))
    pending=[(key,row) for key,row in sorted(rows.items()) if key not in scores]
    print(f'[prepare] inputs={len(evaluation)} pairs={len(rows)} reused={len(scores)} new_calls={len(pending)} params={params}',flush=True)
    if args.judge and pending:
        client=OpenAI(timeout=60,max_retries=2)
        judges={route:OpenAIJudge(model=OPENAI_JUDGE_MODEL,client=client,system_prompt=real.BINARY_SYSTEM_PROMPT if route=='creators' else PROMPT,schema=BINARY_SCHEMA) for route in ('tags','creators','favorites')}
        def work(item):
            key,row=item; result=judges[row['route']].judge(row['input'],row['candidate'])
            return key,{'score':result.score,'hash':contract['pair_hashes'][key],'source':'new_llm','input_tokens':result.input_tokens,'output_tokens':result.output_tokens}
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            for offset in range(0,len(pending),20):
                futures=[executor.submit(work,item) for item in pending[offset:offset+20]]
                error=None
                for future in futures:
                    try:key,value=future.result(); scores[key]=value
                    except Exception as exc:error=exc
                cache_path.write_text(json.dumps(cache,ensure_ascii=False,indent=2))
                print(f'[judge] saved={len(scores)}/{len(rows)} elapsed={time.monotonic()-start:.0f}s',flush=True)
                if error:
                    print(f'[judge] stopped: {type(error).__name__} status={getattr(error,"status_code",None)}',flush=True); break
    complete=[c for c in evaluation if all(f'{c["id"]}::{v["id"]}' in scores for selected in candidates[c['id']].values() for v in selected)]
    result=score_cases(complete,candidates,scores,plan['top_k'],plan['seed']) if complete else {}
    summary={'status':'llm_complete_human_unverified' if len(scores)==len(rows) else 'partial_not_comparable_to_full_sample',
             'params':params,'evaluation_inputs':len(evaluation),'complete_inputs':len(complete),'total_pairs':len(rows),'judged_pairs':len(scores),
             'pending_pairs':len(rows)-len(scores),'groups':result,
             'new_tokens':{name:sum(v.get(name,0) for v in scores.values() if v['source']=='new_llm') for name in ('input_tokens','output_tokens')},
             'limitations':['Exploratory reused inputs','Synthetic favorites','No self-selected candidate tags','No human agreement','Transferred creator-dev parameters, not route-optimized','No popularity baseline']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','limitations')},ensure_ascii=False),flush=True)


def main():
    """Use --judge only with approval for new paid requests."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','source-dir','flow-dir','pilot-dir','output-dir'):
        parser.add_argument('--'+name,type=lambda v:Path(v).expanduser(),required=True)
    parser.add_argument('--judge',action='store_true')
    execute(parser.parse_args())


if __name__=='__main__':main()
