"""Opt-in subprocess supervision test: SIMBENCH_RUNTIME_TESTS=1 enables it."""
import json,os,tempfile,threading,unittest
from pathlib import Path
from simbench.config import load

@unittest.skipUnless(os.environ.get('SIMBENCH_RUNTIME_TESTS')=='1','set SIMBENCH_RUNTIME_TESTS=1 to initialize workers')
class RuntimeTests(unittest.TestCase):
    def test_timeout_restart_and_interruption(self):
        from simbench.runtime import Lane
        c=load(Path(__file__).resolve().parents[1]/'configs/smoke_cpu.json')
        stop=threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);lane=Lane(0,None,out,c,stop)
            request=dict(operation='sleep_test',sleep_s=2,folder=str(out),result=str(out/'r.json'))
            try:
                result=lane.invoke(request,.2);self.assertEqual(result['status'],'timeout');first=lane.proc.pid
                result=lane.invoke(request|dict(sleep_s=.01),2);self.assertEqual(result['status'],'success');self.assertNotEqual(first,lane.proc.pid)
                # Ensure a new timeout cannot inherit a preceding successful result.
                result=lane.invoke(request,.2);self.assertEqual(result['status'],'timeout');self.assertNotIn('worker_wall_s',result)
                lane.start();timer=threading.Timer(.2,stop.set);timer.start()
                result=lane.invoke(request,10);timer.join();self.assertEqual(result['status'],'interrupted')
            finally:stop.set();lane.stop()

if __name__=='__main__':unittest.main()
