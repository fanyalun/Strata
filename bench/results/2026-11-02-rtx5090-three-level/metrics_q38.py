import ctypes,ctypes.util,hashlib,pathlib,re,threading,time

ENV={'STRATA_IQ_MT_MIN':'1','STRATA_ROUTE_RESIDENT':'0','STRATA_ROUTE_TAIL_SKIP':'0','STRATA_ROUTE_PRIOR':'0','STRATA_PREFILL_CPU_SHARE':'0','STRATA_PREFILL_WARM':'0','STRATA_IO_PREFETCH':'1','STRATA_IO_PF_THREADS':'2','STRATA_IO_STATS':'1'}

SCOPE='Native per-request source reports cover activity since the preceding report; the first includes initialization. Later reports include prompt plus decode. Exact measured-request OS storage bytes are sampled from /proc/PID/io around GEN; internal MB/ms values retain native rounding.'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()

def proc_io(pid):
    try:return {k:int(v) for k,v in (line.split(':',1) for line in pathlib.Path('/proc/%d/io'%pid).read_text().splitlines())}
    except Exception as e:return {'error':repr(e)}

def proc_rss(pid):
    try:
        vals={}
        for line in pathlib.Path('/proc/%d/status'%pid).read_text().splitlines():
            k,_,v=line.partition(':')
            if k in ('VmRSS','VmHWM','RssAnon','RssFile','VmLck'):vals[k+'_bytes']=int(v.split()[0])*1024
        return vals
    except Exception as e:return {'error':repr(e)}

class Nvml:
    class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self):
        self.error=None
        try:
            self.lib=ctypes.CDLL(ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
            if self.lib.nvmlInit_v2():raise RuntimeError('nvmlInit_v2 failed')
            self.handle=ctypes.c_void_p()
            # This environment has one physical GPU. Preserve its full-device memory scope explicitly.
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle)):raise RuntimeError('NVML device0 unavailable')
        except Exception as e:self.error=repr(e)
    def used(self):
        if self.error:return None
        m=self.Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(m)):return None
        return int(m.used)
    def close(self):
        if not self.error:self.lib.nvmlShutdown()

class Sampler:
    def __init__(self,pid,nvml):
        self.pid=pid;self.nvml=nvml;self.samples=[];self.stop=threading.Event()
        self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set():
            self.samples.append({'at':time.monotonic(),**proc_rss(self.pid),'gpu_device_used_bytes':self.nvml.used()})
            self.stop.wait(.05)
    def __enter__(self):self.t.start();return self
    def __exit__(self,*args):self.stop.set();self.t.join(timeout=5)
    def report(self):
        good=[s for s in self.samples if 'VmRSS_bytes' in s];gpu=[s['gpu_device_used_bytes'] for s in self.samples if s['gpu_device_used_bytes'] is not None]
        return {'interval_s':.05,'samples':len(self.samples),'peak_sampled_rss_bytes':max((s['VmRSS_bytes'] for s in good),default=None),'process_lifetime_hwm_bytes':max((s.get('VmHWM_bytes',0) for s in good),default=None),'peak_gpu_device_used_bytes':max(gpu,default=None),'vram_scope':'Full physical GPU0 used bytes, including driver; exclusive inference required, not per-process NVML accounting.','nvml_error':self.nvml.error}

def gen(eng,ids,n,protocol_log,nvml,measured):
    cmd='GEN %d %s'%(n,','.join(map(str,ids)))
    before=proc_io(eng.p.pid);rss_before=proc_rss(eng.p.pid);start=time.monotonic();tokens=[];lines=[]
    with Sampler(eng.p.pid,nvml) as sampler:
        eng.send(cmd)
        while True:
            line=eng.line();lines.append(line)
            if line is None:raise RuntimeError('engine exited before DONE')
            if line.startswith('ERR'):raise RuntimeError(line)
            if line.startswith('T '):tokens.append(int(line.split()[1]))
            if line.startswith('DONE'):break
    wall=time.monotonic()-start;after=proc_io(eng.p.pid)
    f=line.split()
    done={'tokens':int(f[1]),'prompt_tokens':int(f[2]),'prompt_ms':float(f[3]),'decode_ms':float(f[4]),'finish':f[5],'accepted':int(f[6]),'offered':int(f[7]),'reused_prompt_inputs':int(f[8]),'fresh_prompt_inputs':int(f[14])}
    done['output_tokens_per_decode_second']=done['tokens']*1000/done['decode_ms'] if done['decode_ms']>0 else None
    done['acceptance']=done['accepted']/done['offered'] if done['offered'] else None
    with open(protocol_log,'a') as log:
        log.write('> '+cmd+'\n');log.writelines('< '+str(l)+'\n' for l in lines)
    return {'measured':measured,'command':cmd,'tokens':tokens,'done_line':line,'done':done,'wall_s':wall,'proc_io_before':before,'proc_io_after':after,'proc_io_delta':{k:after[k]-v for k,v in before.items() if isinstance(v,int) and isinstance(after.get(k),int)},'rss_before':rss_before,'memory':sampler.report()}

def parse_shutdown(path):
    txt=path.read_text();lines=txt.splitlines();records=[]
    for line in lines:
        if 'file tier I/O this request:' not in line:continue
        r={'raw':line}
        for pattern,keys in [
            (r'OS read ([\d.]+) MB from storage \((\d+) major faults\) for ([\d.]+) MB of expert reads',('os_storage_mb_since_previous_report','major_faults_since_previous_report','logical_expert_mb_since_previous_report')),
            (r'io prefetch: (\d+) read ahead \(([\d.]+) MB by pread\), (\d+) used, (\d+) unused, (\d+) skipped \(cached\), (\d+) dropped',('prefetch_blobs','pread_mb','prefetch_used','prefetch_unused','prefetch_cached_skips','prefetch_dropped')),
            (r'decode waited ([\d.]+) ms in (\d+) fetches',('decode_wait_ms','decode_fetches')),
            (r'read ahead ([\d.]+) MB, ([\d.]+) MB of it used; (\d+) late',('prefetch_mb','prefetch_used_mb','prefetch_late'))]:
            m=re.search(pattern,line)
            if m:r.update({key:float(value) for key,value in zip(keys,m.groups())})
        records.append(r)
    observed=re.findall(r'(?:whole-blob preads on|ones read by) (\d+) I/O threads',txt)
    config={'prefetch':1 if 'io prefetch:' in txt else None,'threads':int(observed[-1]) if observed else None,'stats':1 if 'page cache held' in txt else None,'requested_environment':ENV,'observation':'threads from native startup; prefetch/stats from native request report'}
    return {'scope':SCOPE,'source_io':{'records':records,'scope':'Native reports since preceding report; first includes initialization. Rounded MB/ms retained as reported. Exact request OS read_bytes comes from proc_io_delta.'},'source_config':config,'source_tiers':{'native_lines':[line for line in lines if 'strata serve: expert tiers:' in line]},'three_level_stage_lines':[line for line in lines if 'strata three-level:' in line]}
