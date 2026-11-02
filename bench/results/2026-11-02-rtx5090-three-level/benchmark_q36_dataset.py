#!/usr/bin/env python3
"""Q36 GSM8K/HumanEval collection. Use an otherwise idle GPU for execution."""
import argparse,importlib.util,json,os,pathlib,re,statistics,sys,time,traceback
BASE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path(os.environ.get('STRATA_BENCH_REPO',str(BASE.parents[2])))
EXE=pathlib.Path(os.environ.get('STRATA_BENCH_BUILD',str(ROOT/'build-three-level')))/'strata-q36'
ASSETS=pathlib.Path(os.environ.get('STRATA_BENCH_ASSETS',str(ROOT/'models')))
MODEL=ASSETS/'models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
METRICS=BASE/'metrics_q36.py'
spec=importlib.util.spec_from_file_location('q36_bench_tools',METRICS);helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
spec=importlib.util.spec_from_file_location('q36_protocol',ROOT/'tools/qwen35/q36_harness.py');protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
ARMS={'ar':['--mtp','0'],'two_level':['--mtp','3'],'three_level':['--three-level','--three-level-depth','3','--three-level-loops','6','--preverify-top-h','4']}
ENV={'STRATA_Q36_FIXED_SLOTS':'1','STRATA_Q36_KEEP_PACKED':'1','STRATA_NO_PRESET':'1','STRATA_IO_PREFETCH':'1','STRATA_IO_PF_THREADS':'2','STRATA_IO_STATS':'1'}
NORMAL_GUARD_REASONS={'engine_exited','engine_zombie','pid_reused'}

def read_inputs(path,ctx):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)!=22:raise ValueError('Expected exactly22 shared input records')
    for ds in ('gsm8k','humaneval'):
        for phase,count in (('warmup',3),('timed',8)):
            subset=sorted((r for r in rows if r['dataset']==ds and r['phase']==phase),key=lambda r:r['phase_index'])
            if [r['phase_index'] for r in subset]!=list(range(count)):raise ValueError('Invalid phase count/index for '+ds+'/'+phase)
        warm={r['task_id'] for r in rows if r['dataset']==ds and r['phase']=='warmup'}
        timed={r['task_id'] for r in rows if r['dataset']==ds and r['phase']=='timed'}
        if warm&timed:raise ValueError('Warmup and timed tasks must be disjoint')
    for r in rows:
        if any(k in r for k in ('answer','test','canonical_solution','entry_point')):raise ValueError('Reference data must not be in inputs')
        ids=r['models']['q36']['token_ids']
        if not ids or len(ids)!=r['models']['q36']['token_count']:raise ValueError('Bad Q36 tokens/count')
        if len(ids)+256+8>ctx:raise ValueError('Input +256 output +8 margin exceeds ctx: '+r['task_id'])
    return rows

def engine_args(ctx,arm):
    # Omitting expert-slots preserves Q36's native all-GPU branch and its graphs.
    return ['--model',str(MODEL),'--max-context',str(ctx),'--chunk','128','--no-warmup','--cpu-threads','8','--hybrid-rows','16','--expert-placement','partitioned','--resident-budget-gib','32',*ARMS[arm]]

def environment(guard_log,diagnostic=False):
    e={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','CUDA_','GGML_','BENCH_'))}
    for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER'):
        if k in os.environ:e[k]=os.environ[k]
    e.update(ENV)
    e.update(BENCH_ENGINE_BINARY=str(EXE),BENCH_MEMORY_GUARD_LOG=str(guard_log))
    if diagnostic:e['STRATA_Q36_WINDOW_CHECK']='1'
    return e

def model_geometry():
    sys.path.insert(0,str(ROOT/'tools'))
    from gguf_reader import GGUFFile
    model=GGUFFile(MODEL);arch=model.metadata['general.architecture']
    mtp=int(model.metadata.get(arch+'.nextn_predict_layers',0));blocks=int(model.metadata[arch+'.block_count']);main=blocks-mtp
    expert=int(model.metadata[arch+'.expert_count']);target_bytes=0;mtp_bytes=0
    for tensor in model.tensors:
        match=re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight',tensor.name)
        if match:
            size=tensor.expected_bytes()
            if size is None:raise RuntimeError('Unknown expert tensor byte geometry')
            if int(match[1])<main:target_bytes+=size
            else:mtp_bytes+=size
    return {'target_layers':main,'mtp_layers':mtp,'experts_per_layer':expert,'target_routed_tensor_bytes':target_bytes,'mtp_routed_tensor_bytes':mtp_bytes,'mtp_experts':mtp*expert,'source_loads_mtp_weights_even_when_drafting_off':True,'native_tail_padding_per_layer_bytes':4096}

def prepare_meta(args,rows):
    return {'script':str(pathlib.Path(__file__).resolve()),'script_sha256':helpers.sha(pathlib.Path(__file__)),'binary':str(EXE),'binary_sha256':helpers.sha(EXE),'model':str(MODEL),'inputs':str(args.inputs),'inputs_sha256':helpers.sha(args.inputs),'dataset_manifest_sha256':helpers.sha(BASE/'manifest.json'),'guard_wrapper':str(args.guard_wrapper),'resource_target':{'gpu_gib':32,'dram_target_gib':50,'dram_guard_kind':'soft_watchdog','hard_dram_cap':False,'rss_stop_gib':48,'min_system_available_gib':8,'poll_ms':50,'expert_resident_budget_gib':32,'resident_budget_is_total_ram_cap':False,'expected_cpu_primary_expert_bytes_if_all_gpu':0},'ctx':args.ctx,'chunk':128,'max_new':256,'greedy':True,'eos_behavior':'normal native EOS stopping','adaptive_draft_length':False,'native_two_level_depth':3,'three_level':{'depth':3,'loops':6,'top_h':4},'max_prompt_tokens':max(r['models']['q36']['token_count'] for r in rows),'warmup_per_dataset':3,'timed_per_dataset':8,'warmup_disjoint':True,'environment':ENV,'residency_policy':'Automatic all-GPU placement; require all256 experts/layer in all arms, same packed configuration, no eager fallback. No explicit expert-slots flag.','commands':{arm:[str(args.guard_wrapper),'--serve',*engine_args(args.ctx,arm)] for arm in ARMS},'cache_setup':'With no model readers, model-only posix_fadvise(DONTNEED) before each new engine; load and warmup under the soft guard. No external full-file cache warming.','model_geometry':model_geometry(),'ttft_recorded':False,'ttft_note':'Existing raw collection helper records prompt_ms/decode_ms/wall; TTFT is not recorded.','references':'Only same-run AR generated tokens are used for parity; answer/canonical_solution/test file is never loaded.'}

def guard_finish(path):
    for _ in range(80):
        try:
            r=json.loads(path.read_text())
            if (r.get('termination') or {}).get('reason'):return r
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(.05)
    raise RuntimeError('RAM guardian did not publish a completed record: '+str(path))

def verify_guard(record):
    term=record.get('termination') or {};peaks=record.get('peaks') or {}
    if record.get('guard_kind')!='soft_watchdog' or record.get('hard_cap') is not False:raise RuntimeError('Unexpected guard declaration')
    if term.get('reason') not in NORMAL_GUARD_REASONS or term.get('signal_sent'):raise RuntimeError('RAM watchdog triggered: '+repr(term))
    if peaks.get('swap_bytes',0):raise RuntimeError('Engine used swap')
    if max(peaks.get('rss_bytes',0),peaks.get('hwm_bytes',0))>50*(1<<30):raise RuntimeError('Observed process high-water RSS exceeded50GiB')

def validate_residency(stderr,arm,geometry):
    text=stderr.read_text()
    match=re.search(r'every routed expert in VRAM \(([\d.]+) GiB\)',text)
    if not match:raise RuntimeError('Q36 did not select native all-GPU placement')
    forbidden=('partitioned experts:','hybrid experts:','graph budget','given back so that every routed')
    if any(s in text for s in forbidden):raise RuntimeError('Unexpected hybrid/graph/packed fallback; timings invalid')
    return {'all_gpu':True,'target_layers':geometry['target_layers'],'experts_per_target_layer':geometry['experts_per_layer'],'routed_expert_gib_reported':float(match[1]),'cpu_primary_expert_bytes':0,'no_graph_fallback':True,'packed_copies_kept':True,'mtp_drafting_enabled':arm!='ar','mtp_weight_experts_loaded':geometry['mtp_experts'],'mtp_routed_tensor_bytes':geometry['mtp_routed_tensor_bytes']}

def summary(rows):
    tokens=sum(r['request']['done']['tokens'] for r in rows);prompt_ms=sum(r['request']['done']['prompt_ms'] for r in rows);decode_ms=sum(r['request']['done']['decode_ms'] for r in rows);wall=sum(r['request']['wall_s'] for r in rows)
    accepted=sum(r['request']['done']['accepted'] for r in rows);offered=sum(r['request']['done']['offered'] for r in rows)
    return {'timed_samples':len(rows),'output_tokens':tokens,'total_prompt_ms':prompt_ms,'total_decode_ms':decode_ms,'total_wall_s':wall,'aggregate_decode_output_tok_s':tokens*1000/decode_ms if decode_ms else None,'aggregate_prompt_plus_decode_output_tok_s':tokens*1000/(prompt_ms+decode_ms) if prompt_ms+decode_ms else None,'aggregate_wall_output_tok_s':tokens/wall if wall else None,'median_sample_decode_output_tok_s':statistics.median(r['request']['done']['output_tokens_per_decode_second'] for r in rows),'accepted':accepted,'offered':offered,'acceptance':accepted/offered if offered else None,'stopped_eos_samples':sum(r['request']['done']['finish']=='stop' for r in rows),'read_bytes_per_request':[r['request']['proc_io_delta'].get('read_bytes') for r in rows],'all_exact_ar_tokens':all(r['exact_ar_token_parity'] for r in rows)}

def execute(args,rows,meta):
    if not args.guard_wrapper.is_file() or not os.access(args.guard_wrapper,os.X_OK):raise RuntimeError('Executable RAM guardian is required')
    stamp='%s-%d'%(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),os.getpid());out=BASE/'q36-runs'/(args.action+'-'+stamp);out.mkdir(parents=True)
    (out/'benchmark_q36_dataset.py').write_bytes(pathlib.Path(__file__).read_bytes());(out/'manifest.json').write_text(json.dumps(meta,indent=2))
    (BASE/('q36-latest-'+args.action+'.json')).write_text(json.dumps({'directory':str(out),'results':str(out/'results.json')},indent=2))
    result={'manifest':meta,'action':args.action,'sessions':[],'summary':{},'passed':False,'performance_claim':False}
    def save():(out/'results.json').write_text(json.dumps(result,indent=2))
    save();original_env=dict(os.environ);nvml=helpers.Nvml();references={}
    try:
        datasets=('calibration',) if args.action=='calibrate' else ('gsm8k','humaneval')
        for dataset in datasets:
            for arm in ARMS:
                name=dataset+'-'+arm;print('START '+name,flush=True)
                log=out/(name+'.stderr.log');proto=out/(name+'.protocol.log');proto.write_text('');guard=out/(name+'.ram-guard.json')
                env=environment(guard,diagnostic=args.action=='calibrate' and arm=='three_level')
                session={'dataset':dataset,'arm':arm,'stderr':str(log),'protocol':str(proto),'guard_log':str(guard),'command':meta['commands'][arm],'environment':{k:v for k,v in env.items() if k.startswith(('STRATA_','BENCH_','CUDA_'))},'requests':[]};result['sessions'].append(session);save()
                fd=os.open(MODEL,os.O_RDONLY)
                try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                finally:os.close(fd)
                session['cache_advice']='Model-only DONTNEED, after previous engine/guardian exit and before load'
                # Eng merges os.environ: replace it while creating the guarded child.
                os.environ.clear();os.environ.update(env)
                eng=protocol.Eng.__new__(protocol.Eng)
                try:
                    with helpers.Sampler(0,nvml) as load_sampler:
                        eng.__init__(str(args.guard_wrapper),engine_args(args.ctx,arm),env=env,log=str(log),quiet=True)
                    session['load_memory']=load_sampler.report()
                except BaseException:
                    if hasattr(eng,'p'):
                        eng.close();eng.p.wait(timeout=30);session['exit_code']=eng.p.returncode
                    if hasattr(getattr(eng,'logf',None),'close'):eng.logf.close()
                    session['ram_guard']=guard_finish(guard);save()
                    raise
                finally:os.environ.clear();os.environ.update(original_env)
                session.update(pid=eng.p.pid,ready=eng.ready,info=eng.info,load_s=eng.load_s)
                try:
                    session['residency_at_ready']=validate_residency(log,arm,meta['model_geometry'])
                    selected=[max(rows,key=lambda r:r['models']['q36']['token_count'])] if args.action=='calibrate' else sorted((r for r in rows if r['dataset']==dataset),key=lambda r:(r['phase']=='timed',r['phase_index']))
                    for sample in selected:
                        n=1 if args.action=='calibrate' else 256
                        measured=args.action=='benchmark' and sample['phase']=='timed'
                        request=helpers.gen(eng,sample['models']['q36']['token_ids'],n,proto,nvml,measured)
                        key=(sample['dataset'],sample['task_id'],sample['phase'])
                        if arm=='ar':references[key]=request['tokens']
                        equal=request['tokens']==references[key]
                        item={'dataset':sample['dataset'],'task_id':sample['task_id'],'source_index':sample['source_index'],'phase':sample['phase'],'phase_index':sample['phase_index'],'input_tokens':sample['models']['q36']['token_count'],'ttft_s':None,'request':request,'exact_ar_token_parity':equal};session['requests'].append(item);save()
                        if not equal:raise RuntimeError('AR token mismatch: '+name+'/'+sample['task_id'])
                        if request['done']['tokens']!=len(request['tokens']) or not 1<=len(request['tokens'])<=n:raise RuntimeError('Bad output count')
                        if request['done']['finish'] not in ('length','stop'):raise RuntimeError('Unexpected finish: '+request['done']['finish'])
                        if measured:print('TIMED %s %s tokens=%d decode_ms=%.1f parity=exact'%(name,sample['task_id'],len(request['tokens']),request['done']['decode_ms']),flush=True)
                finally:
                    eng.close();eng.p.wait(timeout=30)
                    if hasattr(eng.logf,'close'):eng.logf.close()
                    session['exit_code']=eng.p.returncode;session['ram_guard']=guard_finish(guard);session['expert_offload_io']={'status':'not_applicable','reason':'All target and loaded MTP expert weights are GPU resident; file-tier shutdown counters are absent.'};session['shutdown']=helpers.parse_shutdown(log);session['shutdown']['scope']='Initialization +3 independent warmup +8 timed requests (benchmark); calibration uses1 n1 probe and optional22shape diagnostic.';save()
                verify_guard(session['ram_guard'])
                if session['exit_code']:raise RuntimeError('Engine exited nonzero')
                session['residency_at_exit']=validate_residency(log,arm,meta['model_geometry'])
                if args.action=='calibrate' and arm=='three_level':
                    session['diagnostic_exact_shapes']=sum('native window check:' in l and 'exact' in l for l in log.read_text().splitlines())
                    # Serve does not call the CLI-only startup diagnostic, so scratch/graph reservation is validated at load;
                    # actual phase graphs are exercised by the three official warmups in the benchmark.
                    session['diagnostic_scope']='n1 serve probe validates allocated25-row scratch/reserve; phase graphs are exercised during warmup.'
                if args.action=='benchmark':result['summary'][name]=summary([r for r in session['requests'] if r['phase']=='timed'])
                save();print('FINISH '+name,flush=True)
        resident=[s['residency_at_exit'] for s in result['sessions']]
        invariant=('all_gpu','target_layers','experts_per_target_layer','cpu_primary_expert_bytes','no_graph_fallback','packed_copies_kept')
        if any(any(r[k]!=resident[0][k] for k in invariant) for r in resident):raise RuntimeError('Target GPU/CPU placement or packed policy differs between arms')
        if args.action=='benchmark':
            spec=importlib.util.spec_from_file_location('shared_prompt_encoding',BASE/'tokenizer_tools.py');encoding=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoding)
            tokenizer,_=encoding.load_tokenizer(ASSETS/'qwen36/tokenizer')
            for session in result['sessions']:
                for item in session['requests']:item['output_text']=tokenizer.decode(item['request']['tokens'])
            result['output_text_decoded_after_all_timings']=True
        result['passed']=True;save();print('COMPLETE '+str(out),flush=True)
    except BaseException as e:result['error']=repr(e);result['traceback']=traceback.format_exc();save();raise
    finally:nvml.close();os.environ.clear();os.environ.update(original_env)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--action',choices=('prepare','calibrate','benchmark'),default='prepare');ap.add_argument('--ctx',type=int,default=512);ap.add_argument('--inputs',type=pathlib.Path,default=BASE/'inputs.jsonl');ap.add_argument('--guard-wrapper',type=pathlib.Path,default=BASE/'ram_guard_exec.py');args=ap.parse_args()
    rows=read_inputs(args.inputs,args.ctx);meta=prepare_meta(args,rows);(BASE/'q36-prepared-config.json').write_text(json.dumps(meta,indent=2))
    print('Prepared Q36:22 prompts,max=%d,ctx=%d,3warmup+8timed per dataset/arm,normal EOS,max_new256.'%(meta['max_prompt_tokens'],args.ctx),flush=True)
    if args.action!='prepare':execute(args,rows,meta)
if __name__=='__main__':main()
