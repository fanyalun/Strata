#!/usr/bin/env python3
"""Q38 dataset collection; preparation is CPU-only. Use an otherwise idle GPU for execution."""
import argparse
import importlib.util
import json
import os
import pathlib
import re
import statistics
import struct
import time
import traceback

BASE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get('STRATA_BENCH_REPO',str(BASE.parents[2])))
ASSETS = pathlib.Path(os.environ.get('STRATA_BENCH_ASSETS',str(ROOT/'models')))
EXE = pathlib.Path(os.environ.get('STRATA_BENCH_BUILD',str(ROOT/'build-three-level'))) / 'strata'
PACK = ASSETS / 'pack-iq2_xs'
MTP = ASSETS / 'mtp-rt'
MODELS = tuple(ASSETS / 'models' / name for name in (
    'Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf',
    'Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf'))
PROFILE = ROOT / 'data/expert-profile.bin'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


helpers = module('q38_benchmark_metrics', BASE / 'metrics_q38.py')
protocol = module('q38_dataset_protocol', ROOT / 'tools/qwen35/q36_harness.py')
input_tools = module('q38_dataset_input_tools', BASE / 'tokenizer_tools.py')
ARMS = {
    'ar': [],
    'two_level': ['--mtp', str(MTP)],
    'three_level': ['--mtp', str(MTP), '--three-level', '--three-level-depth', '3',
                    '--three-level-loops', '6', '--preverify-top-h', '4'],
}
ENV = {
    'STRATA_IQ_MT_MIN': '1', 'STRATA_ROUTE_RESIDENT': '0',
    'STRATA_ROUTE_TAIL_SKIP': '0', 'STRATA_ROUTE_PRIOR': '0',
    'STRATA_PREFILL_CPU_SHARE': '0', 'STRATA_PREFILL_WARM': '0',
    'STRATA_IO_PREFETCH': '1', 'STRATA_IO_PF_THREADS': '2', 'STRATA_IO_STATS': '1',
    'STRATA_FS_SLOTS': '0', 'STRATA_SPEC_COST': '0', 'STRATA_PCIE_BALANCE': '0',
}
NORMAL_GUARD_REASONS = {'engine_exited', 'engine_zombie', 'pid_reused'}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_inputs(path, ctx):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 22:
        raise ValueError('Expected exactly22 shared records')
    for dataset in ('gsm8k', 'humaneval'):
        for phase, indices in (('warmup', list(range(3))), ('timed', list(range(3, 11)))):
            selected = [r for r in rows if r['dataset'] == dataset and r['phase'] == phase]
            if [r['source_index'] for r in selected] != indices:
                raise ValueError('Unexpected original row selection: ' + dataset + '/' + phase)
            if [r['phase_index'] for r in selected] != list(range(len(indices))):
                raise ValueError('Unexpected phase order')
    for row in rows:
        if any(key in row for key in ('answer', 'canonical_solution', 'test', 'entry_point')):
            raise ValueError('Reference data must not be in engine inputs')
        encoded = row['models']['q38']
        if len(encoded['token_ids']) != encoded['token_count'] or not encoded['token_ids']:
            raise ValueError('Invalid Q38 input token count')
        if encoded['token_ids'] != row['models']['q36']['token_ids']:
            raise ValueError('Shared chat wrapper has different token IDs')
        if encoded['token_count'] + 256 + 1 >= ctx:
            raise ValueError('Input/output/anchor exceeds native context: ' + row['task_id'])
    return rows


def engine_args(ctx, arm, cache):
    return [
        '--pack', str(PACK), '--native', str(MODELS[0]),
        '--expert-profile', str(PROFILE), '--expert-cache', str(cache),
        '--vram-reserve-mib', '1024', '--resident-budget-gib', '32',
        '--memory-limit-mib', '51200', '--expert-placement', 'partitioned',
        '--max-context', str(ctx), '--prefill', '128', '--greedy', '--spec', '4',
        '--spec-min-p', '0', '--adapt-swaps', '0', '--adapt-every', '0',
        '--pcie-frac', '0', '--prompt-cache', '0', '--suffix-draft', '0',
        '--lookup-chain', '0', *ARMS[arm],
    ]


def environment(guard_log):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('STRATA_', 'CUDA_', 'GGML_', 'BENCH_'))}
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(ENV)
    env.update(BENCH_ENGINE_BINARY=str(EXE), BENCH_MEMORY_GUARD_LOG=str(guard_log))
    return env


def profile_pairs():
    raw = PROFILE.read_bytes()
    magic, version, layers, experts, want, count = struct.unpack_from('<4s5I', raw)
    if magic != b'STRP' or version != 1 or layers != 48 or experts != 512:
        raise ValueError('Unexpected native expert profile header')
    pairs = [struct.unpack_from('<HH', raw, 24 + 4 * i) for i in range(count)]
    if len(set(pairs)) != len(pairs) or len(pairs) != layers * experts:
        raise ValueError('Benchmark requires one complete unique native profile')
    if any(l >= layers or e >= experts for l, e in pairs) or count > want:
        raise ValueError('Invalid native profile pairs')
    return pairs


def expected_mapping(slots):
    pairs = profile_pairs()[:slots]
    if len(pairs) != slots:
        raise ValueError('More actual cache slots than ranked pairs')
    raw = b''.join(struct.pack('<HH', *pair) for pair in pairs)
    import hashlib
    blobs = {}
    for line in (PACK / 'native_experts.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            blobs[int(fields[0])] = int(fields[4])
    cache_bytes = sum((blobs[layer] + 255) // 256 * 256 for layer, _ in pairs)
    return {'pair_encoding': 'little-endian uint16 layer, uint16 expert in slot order',
            'profile_prefix_pairs': slots, 'mapping_sha256': hashlib.sha256(raw).hexdigest(),
            'cache_bytes_from_native_layout': cache_bytes,
            'cache_mib_from_native_layout': cache_bytes / (1 << 20),
            'scope': 'Native fill maps profile prefix into slots; verified full prefill count and unchanged profile. '
                     'Derived from native deterministic placement, not a GPU weight readback.',
            'pairs': pairs}


def parse_residency(path, requested):
    text = path.read_text()
    matches = re.findall(r'expert cache (\d+) slots, ([\d.]+) GiB of VRAM', text)
    if not matches:
        raise RuntimeError('Missing native main-cache allocation record')
    slots, gib = int(matches[-1][0]), float(matches[-1][1])
    filled = re.findall(r'pre-filled (\d+) of (\d+) slots from the profile', text)
    if not filled or tuple(map(int, filled[-1])) != (slots, slots):
        raise RuntimeError('Native profile did not fill every actual cache slot')
    auto = re.findall(r'expert cache auto:[^\n]*?-> (\d+) slots', text)
    if str(requested) == 'auto':
        if not auto:
            raise RuntimeError('Cannot recover auto cache budget request')
        budget_request = int(auto[-1])
    else:
        budget_request = int(requested)
        if auto:
            raise RuntimeError('Fixed-cache request unexpectedly used automatic sizing')
    if 'trying a smaller expert cache' in text or 'shrinking the expert cache' in text:
        raise RuntimeError('Native cache shrank; preserve run and recalibrate fixed placement')
    ram = re.findall(r'resident RAM mode: ([\d.]+) GiB of experts in RAM[^\n]*, (\d+) in the GPU cache; adaptive swaps off', text)
    if not ram or int(ram[-1][1]) != slots:
        raise RuntimeError('Missing fixed CPU/GPU complementary placement record')
    mapping = expected_mapping(slots)
    mapping.pop('pairs')
    if abs(mapping['cache_bytes_from_native_layout'] / (1 << 30) - gib) > .0051:
        raise RuntimeError('Profile prefix byte size disagrees with native cache allocation')
    return {'budget_request': budget_request, 'actual_slots': slots,
            'cache_gib_native_rounded': gib, 'cache_mib_native_rounded': gib * 1024,
            'ram_expert_gib_native_rounded': float(ram[-1][0]),
            'profile_sha256': helpers.sha(PROFILE), **mapping,
            'native_lines': [line for line in text.splitlines() if any(s in line for s in (
                'expert cache auto:', 'expert cache ', 'pre-filled ', 'resident RAM mode:',
                'cache complement ready:', 'expert tiers:', 'complement checksum'))]}


def guard_finish(path):
    for _ in range(80):
        try:
            record = json.loads(path.read_text())
            if (record.get('termination') or {}).get('reason'):
                return record
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(.05)
    raise RuntimeError('RAM guardian has no completed record: ' + str(path))


def verify_guard(record):
    term = record.get('termination') or {}
    peaks = record.get('peaks') or {}
    if record.get('guard_kind') != 'soft_watchdog' or record.get('hard_cap') is not False:
        raise RuntimeError('Unexpected memory guardian declaration')
    if term.get('reason') not in NORMAL_GUARD_REASONS or term.get('signal_sent'):
        raise RuntimeError('Memory watchdog triggered: ' + repr(term))
    if peaks.get('swap_bytes', 0):
        raise RuntimeError('Engine used swap')
    if max(peaks.get('rss_bytes', 0), peaks.get('hwm_bytes', 0)) > 50 * (1 << 30):
        raise RuntimeError('Process RSS exceeded50GiB')


def gen(eng, ids, n, protocol_log, nvml, measured, tokenizer):
    command = 'GEN %d temperature=0 %s' % (n, ','.join(map(str, ids)))
    before = helpers.proc_io(eng.p.pid)
    rss_before = helpers.proc_rss(eng.p.pid)
    tokens, lines, ttft, done_line, error = [], [], None, None, None
    start = time.monotonic()
    with protocol_log.open('a') as log, helpers.Sampler(eng.p.pid, nvml) as sampler:
        log.write('> ' + command + '\n')
        log.flush()
        try:
            eng.send(command)
            while True:
                line = eng.line()
                lines.append(line)
                log.write('< ' + str(line) + '\n')
                log.flush()
                if line is None:
                    raise RuntimeError('Engine exited before DONE')
                if line.startswith('ERR'):
                    raise RuntimeError(line)
                if line.startswith('T '):
                    if ttft is None:
                        ttft = time.monotonic() - start
                    tokens.append(int(line.split()[1]))
                if line.startswith('DONE'):
                    done_line = line
                    break
        except BaseException as exc:
            error = repr(exc)
    after = helpers.proc_io(eng.p.pid)
    value = {'measured': measured, 'command': command, 'tokens': tokens,
             'text': tokenizer.decode(tokens), 'done_line': done_line,
             'ttft_s': ttft, 'wall_s': time.monotonic() - start,
             'proc_io_before': before, 'proc_io_after': after,
             'proc_io_delta': {k: after[k] - v for k, v in before.items()
                               if isinstance(v, int) and isinstance(after.get(k), int)},
             'rss_before': rss_before, 'memory': sampler.report()}
    if error:
        value['error'] = error
        return value
    fields = done_line.split()
    if len(fields) < 16:
        value['error'] = 'Native DONE has fewer than16 fields'
        return value
    done = dict(zip(('tokens', 'prompt_tokens', 'prompt_ms', 'decode_ms', 'finish',
                     'accepted', 'offered', 'reused_prompt_inputs', 'cache_hits',
                     'lookups', 'ram_blobs', 'file_blobs', 'logical_file_mb',
                     'fresh_prompt_inputs', 'offload_entries'), fields[1:16]))
    for key in done:
        if key != 'finish':
            done[key] = float(done[key]) if key in ('prompt_ms', 'decode_ms', 'logical_file_mb') else int(done[key])
    done['acceptance'] = done['accepted'] / done['offered'] if done['offered'] else None
    done['output_tokens_per_decode_second'] = done['tokens'] * 1000 / done['decode_ms'] if done['decode_ms'] else None
    done['tail_fields'] = fields[16:]
    value['done'] = done
    return value


def summary(rows):
    tokens = sum(row['request']['done']['tokens'] for row in rows)
    prompt_ms = sum(row['request']['done']['prompt_ms'] for row in rows)
    decode_ms = sum(row['request']['done']['decode_ms'] for row in rows)
    wall = sum(row['request']['wall_s'] for row in rows)
    accepted = sum(row['request']['done']['accepted'] for row in rows)
    offered = sum(row['request']['done']['offered'] for row in rows)
    speeds = [row['request']['done']['output_tokens_per_decode_second'] for row in rows]
    return {'timed_samples': len(rows), 'output_tokens': tokens,
            'total_prompt_ms': prompt_ms, 'total_decode_ms': decode_ms, 'total_wall_s': wall,
            'aggregate_decode_output_tok_s': tokens * 1000 / decode_ms if decode_ms else None,
            'aggregate_prompt_plus_decode_output_tok_s': tokens * 1000 / (prompt_ms + decode_ms) if prompt_ms + decode_ms else None,
            'aggregate_wall_output_tok_s': tokens / wall if wall else None,
            'median_sample_decode_output_tok_s': statistics.median(s for s in speeds if s is not None),
            'median_ttft_s': statistics.median(row['request']['ttft_s'] for row in rows),
            'accepted': accepted, 'offered': offered, 'acceptance': accepted / offered if offered else None,
            'normal_eos_stops': sum(row['request']['done']['finish'] == 'stop' for row in rows),
            'exact_ar_samples': sum(row['exact_ar_token_parity'] is True for row in rows),
            'all_exact_ar_tokens': all(row['exact_ar_token_parity'] is True for row in rows),
            'proc_read_bytes': sum(row['request']['proc_io_delta'].get('read_bytes', 0) for row in rows),
            'peak_sampled_rss_bytes': max(row['request']['memory']['peak_sampled_rss_bytes'] or 0 for row in rows),
            'peak_gpu_device_used_bytes': max(row['request']['memory']['peak_gpu_device_used_bytes'] or 0 for row in rows)}


def prepare_meta(args, rows):
    cache = 'auto' if args.action != 'benchmark' else None
    calibration = None
    if args.action == 'benchmark':
        calibration = json.loads(args.calibration.read_text())
        if not calibration.get('passed'):
            raise RuntimeError('A successful same-binary calibration is required')
        for key, expected in (('binary_sha256', helpers.sha(EXE)),
                              ('inputs_sha256', helpers.sha(args.inputs)),
                              ('expert_profile_sha256', helpers.sha(PROFILE))):
            if calibration['manifest'].get(key) != expected:
                raise RuntimeError('Calibration asset differs: ' + key)
        cache = calibration['placement']['budget_request']
    elif args.expert_cache != 'auto':
        cache = int(args.expert_cache)
        if cache <= 0:
            raise ValueError('Explicit expert-cache request must be positive')
    return {
        'script': str(pathlib.Path(__file__).resolve()),
        'script_sha256': helpers.sha(pathlib.Path(__file__)),
        'binary': str(EXE), 'binary_sha256': helpers.sha(EXE),
        'inputs': str(args.inputs), 'inputs_sha256': helpers.sha(args.inputs),
        'dataset_manifest_sha256': helpers.sha(BASE / 'manifest.json'),
        'expert_profile': str(PROFILE), 'expert_profile_sha256': helpers.sha(PROFILE),
        'models': [{'path': str(path), 'bytes': path.stat().st_size} for path in MODELS],
        'model_hash_source': 'models.json; no full GGUF rehash before benchmarks',
        'guard_wrapper': str(args.guard_wrapper),
        'guard_wrapper_sha256': helpers.sha(args.guard_wrapper) if args.guard_wrapper.is_file() else None,
        'resource_target': {'physical_gpu_gib': 32, 'dram_target_gib': 50,
                            'dram_guard_kind': 'soft_watchdog', 'hard_dram_cap': False,
                            'rss_stop_gib': 48, 'min_system_available_gib': 8, 'poll_ms': 50,
                            'resident_expert_budget_gib': 32, 'engine_memory_limit_mib': 51200,
                            'resident_budget_is_total_ram_cap': False},
        'ctx': args.ctx, 'prefill': 128, 'max_new': 256, 'greedy': True,
        'eos_behavior': 'normal native EOS stopping', 'adaptive_draft_length': False,
        'native_two_level_spec_rows': 4, 'native_two_level_max_drafts': 3,
        'three_level': {'depth': 3, 'loops': 6, 'top_h': 4},
        'warmup_per_dataset': 3, 'timed_per_dataset': 8, 'warmup_disjoint': True,
        'max_prompt_tokens': max(r['models']['q38']['token_count'] for r in rows),
        'cache_budget_request': cache,
        'calibration': str(args.calibration) if calibration else None,
        'expected_placement': calibration['placement'] if calibration else None,
        'environment': ENV,
        'commands': {arm: [str(args.guard_wrapper), '--serve', *engine_args(args.ctx, arm, cache)] for arm in ARMS},
        'cache_setup': 'With all prior model readers exited, targeted DONTNEED on both model shards before each new process. '
                       'Then native load and3 independent warmups; no external full-model warming.',
        'references': 'Same-batch AR outputs only. No answer, canonical_solution, test or entry_point is loaded.',
        'io_scope': helpers.SCOPE,
    }


def advise_models():
    readers = []
    for process in pathlib.Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            maps = (process / 'maps').read_text()
            open_model = any(str(path) in maps for path in MODELS)
            if not open_model:
                for descriptor in (process / 'fd').iterdir():
                    try:
                        if os.readlink(descriptor) in {str(path) for path in MODELS}:
                            open_model = True
                            break
                    except OSError:
                        continue
            if open_model:
                readers.append(int(process.name))
        except (OSError, ProcessLookupError):
            continue
    if readers:
        raise RuntimeError('Model still has mapped readers before DONTNEED: ' + repr(readers))
    reports = []
    for path in MODELS:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            reports.append({'file': str(path), 'success': True})
        finally:
            os.close(fd)
    return {'advice': 'Model-shard DONTNEED before native load, with prior engine/guardian exited',
            'fully_cold_guaranteed': False, 'mapped_or_open_model_readers': readers, 'files': reports}


def same_placement(actual, expected):
    keys = ('budget_request', 'actual_slots', 'profile_sha256', 'mapping_sha256',
            'cache_gib_native_rounded', 'ram_expert_gib_native_rounded', 'cache_bytes_from_native_layout')
    return all(actual[k] == expected[k] for k in keys)


def execute(args, rows, meta):
    if not args.guard_wrapper.is_file() or not os.access(args.guard_wrapper, os.X_OK):
        raise RuntimeError('Executable memory guardian is required')
    stamp = '%s-%d' % (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), os.getpid())
    out = BASE / 'q38-runs' / (args.action + '-' + stamp)
    out.mkdir(parents=True)
    (out / 'benchmark_q38_dataset.py').write_bytes(pathlib.Path(__file__).read_bytes())
    write_json(out / 'manifest.json', meta)
    write_json(BASE / ('q38-latest-' + args.action + '.json'), {'directory': str(out), 'results': str(out / 'results.json')})
    result = {'manifest': meta, 'action': args.action, 'sessions': [], 'summary': {},
              'passed': False, 'all_exact_ar_tokens': None, 'performance_claim': False}

    def save():
        write_json(out / 'results.json', result)

    save()
    original_env = dict(os.environ)
    original_cwd = os.getcwd()
    nvml = helpers.Nvml()
    tokenizer, _ = input_tools.load_tokenizer(PACK / 'tokenizer')
    references = {}
    try:
        os.chdir(ROOT)
        datasets = ('calibration',) if args.action == 'calibrate' else ('gsm8k', 'humaneval')
        for dataset in datasets:
            arms = ('three_level',) if args.action == 'calibrate' else tuple(ARMS)
            for arm in arms:
                name = dataset + '-' + arm
                print('START ' + name, flush=True)
                log = out / (name + '.stderr.log')
                wire = out / (name + '.protocol.log')
                guard = out / (name + '.ram-guard.json')
                wire.write_text('')
                env = environment(guard)
                session = {'dataset': dataset, 'arm': arm, 'stderr': str(log),
                           'protocol': str(wire), 'guard_log': str(guard),
                           'command': meta['commands'][arm], 'requests': [],
                           'environment': {k: v for k, v in env.items()
                                           if k.startswith(('STRATA_', 'BENCH_', 'CUDA_'))}}
                result['sessions'].append(session)
                save()
                session['cache_advice'] = advise_models()
                load_gpu_sampler = helpers.Sampler(0, nvml)
                load_gpu_sampler.__enter__()
                os.environ.clear()
                os.environ.update(env)
                eng = protocol.Eng.__new__(protocol.Eng)
                try:
                    eng.__init__(str(args.guard_wrapper), engine_args(args.ctx, arm, meta['cache_budget_request']),
                                 env=env, log=str(log), quiet=True)
                except BaseException:
                    if hasattr(eng, 'p'):
                        eng.close()
                        eng.p.wait(timeout=30)
                    if hasattr(getattr(eng, 'logf', None), 'close'):
                        eng.logf.close()
                    if guard.exists():
                        session['ram_guard'] = guard_finish(guard)
                    load_gpu_sampler.__exit__()
                    session['whole_session_gpu_memory'] = load_gpu_sampler.report()
                    save()
                    raise
                finally:
                    os.environ.clear()
                    os.environ.update(original_env)
                session.update(pid=eng.p.pid, ready=eng.ready, info=eng.info, load_s=eng.load_s)
                try:
                    session['placement_at_ready'] = parse_residency(log, meta['cache_budget_request'])
                    if meta['expected_placement'] and not same_placement(session['placement_at_ready'], meta['expected_placement']):
                        raise RuntimeError('Placement differs from frozen three-level calibration')
                    if args.action == 'calibrate':
                        selected = [max((r for r in rows if r['dataset'] == ds),
                                        key=lambda r: r['models']['q38']['token_count'])
                                    for ds in ('gsm8k', 'humaneval')]
                    else:
                        selected = [r for r in rows if r['dataset'] == dataset]
                    for sample in selected:
                        measured = args.action == 'benchmark' and sample['phase'] == 'timed'
                        request = gen(eng, sample['models']['q38']['token_ids'], 256, wire, nvml, measured, tokenizer)
                        key = (sample['dataset'], sample['task_id'], sample['phase'])
                        if arm == 'ar' and 'error' not in request:
                            references[key] = request['tokens']
                        equal = request['tokens'] == references.get(key) if args.action == 'benchmark' else None
                        item = {k: sample[k] for k in ('dataset', 'task_id', 'source_index', 'phase', 'phase_index')}
                        item.update(input_tokens=sample['models']['q38']['token_count'], request=request,
                                    exact_ar_token_parity=equal)
                        if equal is False:
                            ref = references[key]
                            mismatch = next((i for i, (a, b) in enumerate(zip(ref, request['tokens'])) if a != b), min(len(ref), len(request['tokens'])))
                            item['first_ar_mismatch'] = {'index': mismatch, 'ar_tokens': ref,
                                                       'ar_count': len(ref), 'actual_count': len(request['tokens'])}
                        session['requests'].append(item)
                        save()
                        if 'error' in request:
                            raise RuntimeError(request['error'])
                        done = request['done']
                        if done['tokens'] != len(request['tokens']) or not 1 <= len(request['tokens']) <= 256:
                            raise RuntimeError('Native token count disagrees with emitted tokens')
                        if done['finish'] not in ('length', 'stop'):
                            raise RuntimeError('Unexpected native finish: ' + done['finish'])
                        print('%s %s %s tokens=%d decode_ms=%.1f parity=%s' %
                              ('TIMED' if measured else 'PROBE' if args.action == 'calibrate' else 'WARMUP',
                               name, sample['task_id'], len(request['tokens']), done['decode_ms'], equal), flush=True)
                finally:
                    eng.close()
                    eng.p.wait(timeout=30)
                    if hasattr(eng.logf, 'close'):
                        eng.logf.close()
                    session['exit_code'] = eng.p.returncode
                    session['ram_guard'] = guard_finish(guard)
                    session['shutdown'] = helpers.parse_shutdown(log)
                    session['shutdown']['scope'] = 'Load + two unmeasured longest-prompt probes during calibration'
                    if args.action == 'benchmark':
                        session['shutdown']['scope'] = 'Load +3 independent warmup +8 timed requests; no extra requests'
                    load_gpu_sampler.__exit__()
                    session['whole_session_gpu_memory'] = load_gpu_sampler.report()
                    save()
                verify_guard(session['ram_guard'])
                if session['exit_code']:
                    raise RuntimeError('Engine exited nonzero')
                session['placement_at_exit'] = parse_residency(log, meta['cache_budget_request'])
                if not same_placement(session['placement_at_ready'], session['placement_at_exit']):
                    raise RuntimeError('Placement changed during session')
                if args.action == 'calibrate':
                    result['placement'] = session['placement_at_exit']
                    mapping = expected_mapping(result['placement']['actual_slots'])
                    write_json(out / 'gpu-profile-prefix-mapping.json', mapping)
                    result['placement']['mapping_file'] = str(out / 'gpu-profile-prefix-mapping.json')
                else:
                    timed = [r for r in session['requests'] if r['phase'] == 'timed']
                    result['summary'][name] = summary(timed)
                save()
                print('FINISH ' + name, flush=True)
        if args.action == 'benchmark':
            result['all_exact_ar_tokens'] = all(r['exact_ar_token_parity'] is True for s in result['sessions'] for r in s['requests'])
            result['passed'] = result['all_exact_ar_tokens']
        else:
            result['passed'] = True
        save()
        print('COMPLETE ' + str(out), flush=True)
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        nvml.close()
        os.environ.clear()
        os.environ.update(original_env)
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'calibrate', 'benchmark'), default='prepare')
    parser.add_argument('--ctx', type=int, default=512)
    parser.add_argument('--inputs', type=pathlib.Path, default=BASE / 'inputs.jsonl')
    parser.add_argument('--guard-wrapper', type=pathlib.Path, default=BASE / 'ram_guard_exec.py')
    parser.add_argument('--expert-cache', default='auto', help='Calibration budget request, default auto; benchmark reads saved calibration')
    parser.add_argument('--calibration', type=pathlib.Path, help='Successful calibration results.json required for benchmark')
    args = parser.parse_args()
    if args.action == 'benchmark' and not args.calibration:
        parser.error('--calibration is required for benchmark')
    rows = read_inputs(args.inputs, args.ctx)
    profile_pairs()
    meta = prepare_meta(args, rows)
    write_json(BASE / 'q38-prepared-config.json', meta)
    print('Prepared Q38:22 prompts,max=%d,ctx=%d,3warmup+8timed/dataset/arm,max_new256,normal EOS.' %
          (meta['max_prompt_tokens'], args.ctx), flush=True)
    if args.action != 'prepare':
        execute(args, rows, meta)


if __name__ == '__main__':
    main()
