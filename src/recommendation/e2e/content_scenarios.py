"""Synthetic providers; actual Python worker → HTTP → Spring → MySQL contract."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
from src.detail_tagging import automatic_record, digest
from src.recommendation.content_worker import ContentBackend, ContentWorker, ROOT as API
from src.recommendation.e2e.scenarios import Suite, control, receive_control, ControlChannelFailure
from src.recommendation.e2e.runner import ROOT, git_sha
from src.recommendation.http import BackendClientConfig, BackendRequestError
from src.recommendation.storage import _write_json_atomic


def main():
    output = Path(os.environ['CKING_E2E_OUTPUT'])
    report = dict(schemaVersion=1,status='running',completed=False,checks=[],externalConnectionAttempts=0,
                  externalModelCalls=0,llmCommitSha=git_sha(ROOT),beCommitSha=git_sha(Path(os.environ['CKING_E2E_BE_ROOT'])))
    original = socket.socket.connect
    def local(sock,address):
        if address[0] not in {'127.0.0.1','localhost','::1'}:
            report['externalConnectionAttempts'] += 1
            raise AssertionError('External network blocked')
        return original(sock,address)
    def passed(name):
        report['checks'].append(dict(name=name,status='passed'))
        _write_json_atomic(output/'report.json',report)
    try:
        cfg=receive_control()
        cid=cfg['creatorIds']['1']
        client=ContentBackend(BackendClientConfig(cfg['baseUrl'],max_retries=0),access_token=cfg['apiKey'])
        calls=[]
        taxonomy=json.loads((ROOT/'data/creator-subtopics.json').read_text(encoding='utf-8'))
        # Only synthetic test registry is approved. Production remains unseeded.
        taxonomy={**taxonomy,'version':'content-e2e-synthetic-v1',
                  'subtopics':[r for r in taxonomy['subtopics'] if r['code']=='MUSIC_VOCAL']}
        def classify(cid,text,tax,model):
            calls.append(text)
            candidates={'parents':[],'topics':[]}
            if '노래' in text:
                candidates={'parents':[dict(code='MUSIC',evidence='노래',core_evidence='',repetition_evidence=[])],
                            'topics':[dict(code='MUSIC_VOCAL',parent='MUSIC',evidence='노래',core_evidence='',repetition_evidence=[])]}
            return automatic_record(cid,text,candidates,tax,model)
        rec=Suite(cfg,output/'recommendations')
        worker=ContentWorker(client,output/'worker',taxonomy,'synthetic-v1',classify,
                             lambda: 1 if rec.daily.run('apply')['failureCount'] else 0,
                             approved=True,max_calls=30)
        def request(method,path,body=None,token=None):
            return client._request(method,path,body,authenticated=token is not None) if token is None else ContentBackend(
                BackendClientConfig(cfg['baseUrl'],max_retries=0),access_token=token)._request(method,path,body,authenticated=True)
        def reject(method,path,body,status,token=cfg['apiKey']):
            try: request(method,path,body,token)
            except BackendRequestError as error: assert error.status==status
            else: raise AssertionError('Expected rejection')
        def view(public=False):
            return request('GET',f'/api/creators/{cid}/content-tags' if public else '/api/creator/content-tags',token=None if public else cfg['userToken'])
        def edit(text):
            return request('PATCH','/api/creator/space',dict(introText=text,profileImageUrl='profile',bannerImageUrl='banner'),cfg['userToken'])
        def run(source='daily'):
            result=worker.run(source)
            assert result['failureCount']==0, result
            return result
        with patch.object(socket.socket,'connect',local):
            reject('GET',API+'/inputs',None,409)
            assert view(True)['status']=='disabled'
            passed('unapproved_taxonomy_fails_closed')
            control('register',taxonomy=taxonomy,identity=worker.identity,configHash=digest(worker.identity))
            reject('GET',API+'/inputs',None,401,token='wrong')
            reject('GET','/api/creator/content-tags',None,401,token=cfg['apiKey'])
            reject('GET','/api/admin/creator-applications',None,401,token=cfg['apiKey'])
            passed('dedicated_credential_and_owner_routes')
            run('event'); run('event')
            first=control('snapshot')
            assert first['automaticCount']==2 and first['similarCount']==2 and first['interestCount']==17
            assert view(True)['status']=='unconfirmed' and not view(True)['topics']
            mine=view();assert mine['automaticResult']['topics'][0]['code']=='MUSIC_VOCAL'
            passed('initial_v2_delivery_and_all_recommendation_targets')
            before=len(calls);run();assert len(calls)==before
            passed('daily_unchanged_no_model_calls')
            payload=next(json.loads(p.read_text(encoding='utf-8')) for p in (output/'worker/records').glob('*/payload.json') if json.loads(p.read_text(encoding='utf-8'))['creatorId']==cid)
            path=f'{API}/creators/{cid}/automatic'
            assert request('PUT',path,payload,cfg['apiKey'])['status']=='idempotent'
            changed=copy.deepcopy(payload);changed['createdAt']='2026-01-01T00:00:00+00:00'
            reject('PUT',path,changed,409)
            invalid=copy.deepcopy(payload);invalid['record']['result']['topics'][0]['code']='FREE_TAG'
            reject('PUT',path,invalid,400)
            invalid=copy.deepcopy(payload);invalid['record']['result']['parents']*=2
            reject('PUT',path,invalid,400)
            invalid=copy.deepcopy(payload);invalid['record']['result']['topics'][0]['evidence']='absent evidence'
            reject('PUT',path,invalid,400)
            passed('idempotency_conflicting_payload_and_taxonomy_code_validation')
            confirmation=dict(automaticRevision=mine['automaticRevision'],inputHash=mine['inputHash'],parents=['MUSIC'],topics=['MUSIC_VOCAL'])
            reject('PUT','/api/creator/content-tags/confirmation',confirmation,409,token=cfg['otherToken'])
            assert request('PUT','/api/creator/content-tags/confirmation',confirmation,cfg['userToken'])['status']=='creator_confirmed'
            assert view(True)==dict(status='creator_confirmed',parents=['MUSIC'],topics=['MUSIC_VOCAL'])
            request('PUT',path,payload,cfg['apiKey']);assert view(True)['status']=='creator_confirmed'
            passed('owner_confirmation_public_projection_and_same_result_preserves_confirmation')
            # Separate DB transactions prove a competing stale delivery waits for Space edit lock.
            assert control('race',payload=payload)['serialized']
            assert view(True)['status']=='reanalysis_required'
            reject('PUT',path,payload,409)
            reject('PUT','/api/creator/content-tags/confirmation',confirmation,409,token=cfg['userToken'])
            passed('mysql_space_edit_delivery_race_and_stale_confirmation')
            # Claim, then another committed edit invalidates lease generation; ack cannot consume successor.
            job=client.put(API+'/jobs/claim',dict(leaseSeconds=900))
            assert client.put(f'{API}/jobs/{job["jobId"]}/renew',dict(leaseToken=job['leaseToken'],generation=job['generation']))['status']=='RUNNING'
            edit('노래 커버 재수정')
            reject('PUT',f'{API}/jobs/{job["jobId"]}/renew',dict(leaseToken=job['leaseToken'],generation=job['generation']),409)
            reject('PUT',f'{API}/jobs/{job["jobId"]}/ack',dict(leaseToken=job['leaseToken'],generation=job['generation'],status='completed',summary={}),409)
            assert any(j['creator_id']==cid and j['status']=='PENDING' for j in control('snapshot')['jobs'])
            run('event');assert view(True)['status']=='unconfirmed'
            passed('edit_during_claim_keeps_successor_and_event_refreshes_results')
            # A lost receipt after DB commit replays the identical payload, without provider invocation.
            receipt=next(p for p in (output/'worker/records').glob('*/receipt.json') if json.loads((p.parent/'payload.json').read_text(encoding='utf-8'))['inputRevision']==3)
            receipt.unlink();before=len(calls);run();assert len(calls)==before
            passed('restart_after_commit_uses_saved_payload')
            old_payload=next(json.loads(p.read_text(encoding='utf-8')) for p in (output/'worker/records').glob('*/payload.json') if json.loads(p.read_text(encoding='utf-8'))['creatorId']==cid and json.loads(p.read_text(encoding='utf-8'))['inputRevision']==3)
            worker=ContentWorker(client,output/'worker',taxonomy,'synthetic-v2',classify,worker.recommend,approved=True,max_calls=30)
            control('config',configHash=digest(worker.identity))
            run();reject('PUT',path,old_payload,409)
            passed('new_model_config_reclassifies_and_fences_previous_config')
            edit('노래 커버 lease 테스트')
            job=client.put(API+'/jobs/claim',dict(leaseSeconds=900));control('expire')
            reject('PUT',f'{API}/jobs/{job["jobId"]}/ack',dict(leaseToken=job['leaseToken'],generation=job['generation'],status='completed',summary={}),409)
            job2=client.put(API+'/jobs/claim',dict(leaseSeconds=900));assert job2['leaseToken']!=job['leaseToken']
            client.put(f'{API}/jobs/{job2["jobId"]}/ack',dict(leaseToken=job2['leaseToken'],generation=job2['generation'],status='retry',summary={}))
            control('ready');run('event')
            passed('expired_lease_redelivery_and_retry')
            edit('노래 공급자 실패 테스트')
            count=control('snapshot')['automaticCount']
            original_model=worker.classify_fn
            def failed_model(*args):
                raise RuntimeError('synthetic provider outage')
            worker.classify_fn=failed_model
            assert worker.run('event')['stages']['tags']['status']=='failed'
            assert control('snapshot')['automaticCount']==count
            worker.classify_fn=original_model
            control('ready');run('event')
            passed('provider_failure_preserves_success_and_retries_independent_stage')
            edit('')
            before=len(calls);run('event');assert len(calls)==before
            assert view()['status']=='unclassified'
            passed('empty_input_without_provider_call')
            edit('노래 수동 재실행')
            for _ in range(5):
                job=client.put(API+'/jobs/claim',dict(leaseSeconds=900))
                client.put(f'{API}/jobs/{job["jobId"]}/ack',dict(leaseToken=job['leaseToken'],generation=job['generation'],status='retry',summary={}))
                control('ready')
            assert next(j for j in control('snapshot')['jobs'] if j['creator_id']==cid)['status']=='FAILED'
            run('daily')
            assert next(j for j in control('snapshot')['jobs'] if j['creator_id']==cid)['status']=='COMPLETED'
            client.put(f'{API}/jobs/{cid}/retry',{})
            run('event')
            passed('final_failure_daily_reconciliation_and_manual_retry')
            control('disable');assert view(True)['status']=='disabled'
            reject('PUT','/api/creator/content-tags/confirmation',confirmation,409,token=cfg['userToken'])
            passed('taxonomy_revocation_hides_confirmation')
        report.update(status='passed',completed=True,syntheticModelCalls=len(calls))
        return 0
    except Exception as error:
        import traceback
        report.update(status='environment_failure' if isinstance(error,ControlChannelFailure) else 'verification_failure',completed=True,
                      errorType=type(error).__name__,failureLocation=[dict(file=Path(f.filename).name,line=f.lineno) for f in traceback.extract_tb(error.__traceback__)])
        return 1
    finally:
        _write_json_atomic(output/'report.json',report)


if __name__=='__main__':
    raise SystemExit(main())
