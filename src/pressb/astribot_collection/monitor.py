"""One-second resource evidence; independent of simulator stepping/render timing."""
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

class ResourceMonitor:
    def __init__(self,path,queues=lambda:{}):
        self.path=Path(path);self.queues=queues;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,name='collection-resources',daemon=True)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*_):self.stop.set();self.thread.join(timeout=5)
    def run(self):
        import psutil
        with self.path.open('a',buffering=1) as out:
            while not self.stop.is_set():
                start=time.monotonic()
                row=dict(unix_s=time.time(),monotonic_s=start,cpu_percent=psutil.cpu_percent(),
                    available_ram_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(self.path.parent).free,
                    disk_write_bytes=psutil.disk_io_counters().write_bytes,queues=self.queues())
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.encoder,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],text=True,timeout=3)
                    row['gpus']=[dict(zip(['index','compute_percent','encoder_percent','memory_used_mib','memory_total_mib'],
                        [float(x.strip()) for x in line.split(',')])) for line in raw.strip().splitlines()]
                except Exception as exc:row['gpu_error']=str(exc)
                out.write(json.dumps(row)+'\n')
                self.stop.wait(max(0.,1.-(time.monotonic()-start)))
