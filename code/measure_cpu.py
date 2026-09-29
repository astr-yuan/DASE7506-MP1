"""Run the unchanged evaluator and measure the child process peak working set (Windows)."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time

class Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]

class ProcessEntry(ctypes.Structure):
    _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ProcessID',wintypes.DWORD),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),('dwFlags',wintypes.DWORD),('szExeFile',wintypes.WCHAR*260)]

kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessEntry)]
kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessEntry)]
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.CloseHandle.argtypes=[wintypes.HANDLE]

def descendants(root):
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    entry=ProcessEntry(); entry.dwSize=ctypes.sizeof(entry)
    parents={}
    try:
        valid=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while valid:
            parents[entry.th32ProcessID]=entry.th32ParentProcessID
            valid=kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    found={root}
    while True:
        expanded=found|{pid for pid,parent in parents.items() if parent in found}
        if expanded==found:
            return found
        found=expanded

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--split', choices=['validation','test'], default='validation')
    a=p.parse_args()
    if sys.platform != 'win32':
        p.error('This memory wrapper uses Windows APIs; on Linux use /usr/bin/time -v.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'evaluate.py','--checkpoint',a.checkpoint,'--device','cpu','--precision','fp32','--threads','4','--split',a.split,'--output',str(a.output)]
    api=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    api.restype=wintypes.BOOL
    peaks={}
    start=time.perf_counter()
    with a.output.with_suffix('.stdout.txt').open('w') as out:
        proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            for pid in descendants(proc.pid):
                handle=kernel.OpenProcess(0x410,False,pid)
                if not handle:
                    continue
                try:
                    counter=Counters(); counter.cb=ctypes.sizeof(counter)
                    if api(handle,ctypes.byref(counter),counter.cb):
                        peaks[pid]=max(peaks.get(pid,0),counter.PeakWorkingSetSize)
                finally:
                    kernel.CloseHandle(handle)
            time.sleep(.02)
    if proc.returncode:
        raise RuntimeError(a.output.with_suffix('.stdout.txt').read_text())
    peak=sum(peaks.values())
    if not peak:
        raise RuntimeError('Failed to measure process working set.')
    result={'command':cmd,'peak_working_set_bytes':peak,'peak_ram_gib':peak/2**30,'process_seconds':time.perf_counter()-start,'per_process_peak_bytes':peaks,'measurement':'Conservative sum of per-process peak working sets for launcher and all descendants; Windows Toolhelp32 + GetProcessMemoryInfo, 20ms polling; includes loading/tokenization/scoring','score':json.loads(a.output.read_text())}
    a.output.with_suffix('.resource.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
