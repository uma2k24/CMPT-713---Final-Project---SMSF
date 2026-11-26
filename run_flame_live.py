"""Verify the available workflow, export real FLAME previews, and launch localhost UI."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'geometry/live_output'
PORT=8765


def check():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    modules=['models.model_a.test_model','data.test_generate','models.model_b.test_model_b',
             'evaluation.test_metrics','models.model_c.test_objective','evaluation.test_independent_review',
             'geometry.test_flame','data.test_paraphrase_pipeline','models.model_b.test_provisional_data',
             'evaluation.test_ai_validation','geometry.test_live_engine','geometry.test_semantic_calibration']
    result=subprocess.run([sys.executable,'-m','unittest',*modules,'-v'],cwd=ROOT,capture_output=True,text=True)
    (OUTPUT/'verification.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Checks failed; inspect geometry/live_output/verification.log')
    from data.import_paraphrase_reviews import verify_inputs
    from models.model_b.data import sha256
    verify_inputs()
    training=json.loads((ROOT/'experiments/provisional_v1/summary.json').read_text())
    for run in training['runs']:
        if sha256(ROOT/run['checkpoint']/'weights.pt')!=run['checkpoint_sha256']:
            raise ValueError('Trained checkpoint changed')
    evaluation=json.loads((ROOT/'experiments/ai_validation_v1/results.json').read_text())
    for run in evaluation['per_run']:
        path=ROOT/f'experiments/ai_validation_v1/predictions/{run["model"]}_{run["split"]}.jsonl'
        if sha256(path)!=run['prediction_sha256']:
            raise ValueError('Saved evaluation predictions changed')
    record=dict(status='pass',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        test_modules=modules,trained_checkpoints_verified=len(training['runs']),
        evaluation_prediction_files_verified=len(evaluation['per_run']),
        human_review_complete=False,test_human_collected=False,training_repeated=False)
    (OUTPUT/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('PASS: implementation checks, original data hashes, 12 trained checkpoints and 21 evaluation files',flush=True)


def serve(port):
    from geometry.live_engine import LiveEngine
    engine=LiveEngine()
    class Handler(BaseHTTPRequestHandler):
        def local(self):
            host=self.headers.get('Host','')
            origin=self.headers.get('Origin')
            return host in (f'127.0.0.1:{port}',f'localhost:{port}') and (not origin or origin in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'))
        def send(self,value,status=200,mime='application/json'):
            content=json.dumps(value,allow_nan=False).encode() if mime=='application/json' else value
            self.send_response(status)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(content)
        def do_GET(self):
            if not self.local():
                return self.send({'error':'Local access only'},403)
            if self.path=='/api/health':
                return self.send(dict(status='ready',app='cmpt713-flame-live-v1',device=engine.device,geometric_calibration='v1'))
            if self.path=='/api/init':
                return self.send(engine.initialize())
            files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/controls.js':('controls.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if self.path in files:
                name,mime=files[self.path]
                return self.send((ROOT/'geometry/viewer'/name).read_bytes(),mime=mime)
            return self.send({'error':'Not found'},404)
        def do_POST(self):
            if not self.local() or self.headers.get('Content-Type')!='application/json':
                return self.send({'error':'Local JSON requests only'},403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<16000:
                    raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length))
                if self.path=='/api/predict':
                    return self.send(engine.predict(body.get('text'),body.get('model','C')))
                if self.path=='/api/geometry':
                    return self.send(engine.raw_geometry(body))
                if self.path=='/api/semantic':
                    return self.send(engine.semantic_geometry(body.get('controls')))
                return self.send({'error':'Not found'},404)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
                return self.send({'error':str(error)},400)
            except Exception as error:
                print('REQUEST ERROR: '+repr(error),flush=True)
                return self.send({'error':'Inference failed; inspect local server.log'},500)
    print(f'FLAME LIVE READY http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


def launch(port):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=2) as response:
            if json.load(response).get('app')=='cmpt713-flame-live-v1':
                print(f'Already running: http://127.0.0.1:{port}',flush=True)
                return
    except Exception:
        pass
    with (OUTPUT/'server.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--serve','--port',str(port)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    (OUTPUT/'server_process.json').write_text(json.dumps(dict(pid=process.pid,port=port,url=f'http://127.0.0.1:{port}'),indent=2))
    for _ in range(45):
        if process.poll() is not None:
            raise RuntimeError('Server exited; inspect geometry/live_output/server.log')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as response:
                if json.load(response).get('status')=='ready':
                    print(f'RUNNING http://127.0.0.1:{port} (PID {process.pid})',flush=True)
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError('Server is still starting; inspect server.log and health endpoint')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--serve',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--port',type=int,default=PORT)
    args=parser.parse_args()
    if args.check:
        check()
    if args.export:
        from geometry.live_engine import LiveEngine
        LiveEngine().export(OUTPUT)
    if args.launch:
        launch(args.port)
    if args.serve:
        serve(args.port)
