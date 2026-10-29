"""Process-level measured costs; tensor payload is supplied separately, never wire bytes."""
import threading
import time
from datetime import datetime,timezone
import psutil


class CostProfile:
    def __init__(self, interval=.02):
        self.process=psutil.Process(); self.interval=interval
        self.stop=threading.Event(); self.active=False; self.wall=0.; self.cpu=0.
        self.peak=0; self.samples=0

    def _cpu(self):
        x=self.process.cpu_times(); return x.user+x.system

    def _sample(self):
        if self.active:
            self.peak=max(self.peak,self.process.memory_info().rss); self.samples+=1

    def _loop(self):
        while not self.stop.wait(self.interval): self._sample()

    def resume(self):
        if self.active: raise RuntimeError('Profiler already active')
        self.t=time.perf_counter(); self.c=self._cpu(); self.active=True; self._sample()

    def pause(self):
        if not self.active: return
        self._sample(); self.wall+=time.perf_counter()-self.t; self.cpu+=self._cpu()-self.c; self.active=False

    def __enter__(self):
        self.timestamp=datetime.now(timezone.utc).isoformat()
        self.baseline=self.process.memory_info().rss
        self.resume(); self.thread=threading.Thread(target=self._loop,daemon=True); self.thread.start()
        return self

    def __exit__(self,*args):
        self.pause(); self.stop.set(); self.thread.join()

    def observe(self,observer,*args):
        self.pause()
        try: observer(*args)
        finally: self.resume()

    def result(self,steps=0,exposures=0,rounds=0,up=0,down=0,training_rounds=0):
        if self.active: raise RuntimeError('Read profile after exit')
        return {'wall_seconds':self.wall,'process_cpu_seconds':self.cpu,
                'cpu_core_equivalent_percent':100*self.cpu/self.wall if self.wall else 0.,
                'rss_baseline_bytes':self.baseline,'rss_sampled_max_bytes':self.peak,
                'rss_samples':self.samples,'sampling_interval_seconds':self.interval,'timestamp_utc':self.timestamp,
                'optimizer_steps':steps,'sample_exposures':exposures,'recovery_rounds':rounds,'training_rounds':training_rounds,
                'payload_bytes_up':up,'payload_bytes_down':down,'wire_bytes_measured':False,
                'scope':'sequential process; CPU includes monitor; sampled RSS not allocator peak; excludes paused diagnostics'}
