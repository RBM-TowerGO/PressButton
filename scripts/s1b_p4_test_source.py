#!/usr/bin/env python3
"""Loopback-only finite motion-test source; contains no robot SDK."""
import argparse,json,math,os,socket,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--joint',type=int,required=True);p.add_argument('--amplitude',type=float,required=True);p.add_argument('--duration',type=float,required=True);p.add_argument('--fault',choices=['network_disconnect','queue_exhausted','service_failure'],required=True);a=p.parse_args()
    q=json.loads(Path(a.base).read_text())
    if len(q)!=10 or a.joint not in (0,1,2,3,4,5,6,8,9) or not 0<abs(a.amplitude)<=.006:raise ValueError('Invalid bounded test source')
    with socket.socket() as server:
        server.bind(('127.0.0.1',0));server.listen(1);print(json.dumps(dict(port=server.getsockname()[1],pid=os.getpid())),flush=True)
        connection,address=server.accept()
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            if connection.recv(1)!=b'S':return
            start=time.monotonic();ordinal=0
            while True:
                t=time.monotonic()-start
                if t>=a.duration/4:
                    if a.fault=='service_failure':os._exit(17)
                    if a.fault=='network_disconnect':return
                    time.sleep(2);return
                target=list(q);target[a.joint]+=a.amplitude*.5*(1-math.cos(2*math.pi*t/a.duration))
                connection.sendall((json.dumps(dict(sequence=ordinal,target=target,source_ns=time.monotonic_ns()))+'\n').encode())
                ordinal+=1;wait=start+ordinal/250-time.monotonic()
                if wait>0:time.sleep(wait)
if __name__=='__main__':main()
