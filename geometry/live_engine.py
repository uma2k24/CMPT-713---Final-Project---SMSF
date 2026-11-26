"""Real FLAME2019 decoding plus frozen A/B/C inference for a local viewer.

Six geometric proxies are fitted to anatomical measurements. Human perceptual
validation remains pending. Raw PCs and the legacy jaw projection remain separate.
"""
import json
import math
import threading
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from geometry.flame_backend import FlameGeometry, ROOT, RINGNET_FLAME2019_SHA256
from models.model_a.model import predict as predict_a
from models.model_b.model import load_checkpoint, Predictor, ATTRIBUTES
from models.model_b.data import sha256
from models.model_b.train import seed_everything
from geometry.semantic_calibration import SemanticCalibration

REFERENCE_JAW = .175
JAW_DELTA_SCALE = .175


def scalar(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in [{low}, {high}]')
    return float(value)


def jaw_projection(values):
    if not isinstance(values, (list,tuple)) or len(values)!=6:
        raise ValueError('Six semantic values required')
    values=[scalar(v,-1,1,'semantic control') for v in values]
    return dict(reference_jaw_radians=REFERENCE_JAW,
        target_jaw_radians=REFERENCE_JAW+JAW_DELTA_SCALE*values[4],
        rendered_control_indices=[4], unrendered_control_indices=[0,1,2,3,5],
        semantic_mapping_complete=False, calibrated=False,
        mapping='Illustrative signed jaw projection: 0.175 rad reference + 0.175 rad * predicted jaw control',
        raw_controls=values)


class LiveEngine:
    def __init__(self):
        seed_everything(13)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.lock=threading.Lock()
        self.flame=FlameGeometry(ROOT.parent/'FLAME/FLAME2019/FLAME2019/generic_model.pkl').to(self.device)
        if self.flame.model_sha256!=RINGNET_FLAME2019_SHA256:
            raise ValueError('Expected the pinned FLAME2019 generic model')
        self.calibration=SemanticCalibration(self.flame)
        self.predictors={}
        self.checkpoints={}
        summary=json.loads((ROOT/'experiments/provisional_v1/summary.json').read_text())
        for family,lam in [('B',0.),('C',summary['selected_c_lambda'])]:
            # Fixed representative seed, not selected from test performance.
            run=next(r for r in summary['runs'] if r['seed']==13 and r['lambda_value']==lam)
            path=ROOT/run['checkpoint']
            if sha256(path/'weights.pt')!=run['checkpoint_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            model,metadata=load_checkpoint(path,device=self.device)
            tokenizer=AutoTokenizer.from_pretrained(path/'tokenizer',local_files_only=True)
            self.predictors[family]=Predictor(model,tokenizer,metadata,device=self.device)
            self.checkpoints[family]=dict(path=str(path.relative_to(ROOT)),sha256=run['checkpoint_sha256'],seed=13,
                lambda_value=lam,classification_threshold=metadata['classification_threshold'])
        self.faces=self.flame.faces.cpu().tolist()
        self.neutral=self.mesh(0.)
        self.reference=self.mesh(REFERENCE_JAW)
        self.bounds=[np.min(self.neutral,axis=0).tolist(), np.max(self.neutral,axis=0).tolist()]

    @torch.no_grad()
    def mesh(self,jaw,expression=None):
        jaw=scalar(jaw,0,.5,'jaw angle')
        exp=torch.zeros(1,100,device=self.device)
        if expression is not None:
            if not isinstance(expression,list) or len(expression)!=100:
                raise ValueError('Expected 100 expression PC coefficients')
            exp[0]=torch.tensor([scalar(v,-3,3,'expression coefficient') for v in expression],device=self.device)
        pose=torch.tensor([[jaw,0.,0.]],device=self.device)
        vertices=self.flame(expression=exp,jaw_pose=pose)[0][0].cpu().numpy()
        if not np.isfinite(vertices).all():
            raise ValueError('Nonfinite mesh')
        return np.round(vertices,7).tolist()

    def initialize(self):
        return dict(app='cmpt713-flame-live-v1',device=self.device,model='FLAME2019',
            flame_sha256=self.flame.model_sha256,faces=self.faces,neutral=self.neutral,reference=self.reference,
            bounds=self.bounds,attributes=list(ATTRIBUTES),checkpoints=self.checkpoints,
            reference_jaw_radians=REFERENCE_JAW,semantic_mapping_complete=False,ringnet_predictor_used=False,
            geometric_mapping_available=True,calibration=self.calibration.specification())

    def predict(self,text,family='C',mapping='geometric'):
        if not isinstance(text,str) or not text.strip() or len(text)>500:
            raise ValueError('Enter an instruction of 1 to 500 characters')
        if family not in ('A','B','C'):
            raise ValueError('Choose model A, B or C')
        with self.lock:
            prediction=(predict_a([text]) if family=='A' else self.predictors[family].predict([text]))[0]
            if mapping=='geometric':
                vertices,projection=self.calibration.decode(prediction['y'])
                mesh=np.round(vertices,7).tolist()
            elif mapping=='jaw_only':
                projection=jaw_projection(prediction['y'])
                mesh=self.mesh(projection['target_jaw_radians'])
            else:
                raise ValueError('Unknown geometry mapping')
        return dict(text=text,model=family,prediction=prediction,projection=projection,vertices=mesh,
            checkpoint=self.checkpoints.get(family),official_experiment=False,
            notice=('All six controls use geometric proxies; smile means lip-corner lift. Human perceptual validation remains pending.'
                    if mapping=='geometric' else 'Legacy jaw-only projection; other controls are not rendered.'))

    def semantic_geometry(self,controls):
        with self.lock:
            vertices,record=self.calibration.decode(controls)
        return dict(vertices=np.round(vertices,7).tolist(),projection=record,mode='direct_geometric_controls')

    def raw_geometry(self,body):
        jaw=scalar(body.get('jaw_radians',0.),0,.5,'jaw angle')
        index=body.get('pc_index',0)
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<100:
            raise ValueError('PC index must be 0 to 99')
        coefficient=scalar(body.get('coefficient',0.),-3,3,'PC coefficient')
        expression=[0.]*100
        expression[index]=coefficient
        with self.lock:
            mesh=self.mesh(jaw,expression)
        return dict(vertices=mesh,jaw_radians=jaw,pc_index=index,coefficient=coefficient,
                    mode='raw_FLAME_parameters',semantic_labels_assigned=False)

    def export(self,destination):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        from geometry.run_demo import panel,save_obj
        destination=Path(destination)
        destination.mkdir(parents=True,exist_ok=True)
        prompts=['Open the jaw slightly.','Open the jaw much.','Close the jaw slightly.','Close the jaw much.']
        predictions=[self.predict(text,'C',mapping='jaw_only') for text in prompts]
        reference=np.array(self.reference)
        faces=np.array(self.faces)
        bounds=(reference[:,0].min()-.015,reference[:,0].max()+.015,reference[:,1].min()-.025,reference[:,1].max()+.015)
        fig,axes=plt.subplots(1,5,figsize=(15,4.7),facecolor='#f5f8fb')
        panel(axes[0],reference,faces,'Open-jaw reference',bounds)
        for ax,result in zip(axes[1:],predictions):
            panel(ax,np.array(result['vertices']),faces,result['text']+'\nPredicted jaw '+f'{result["prediction"]["y"][4]:+.3f}',bounds)
        fig.suptitle('FLAME2019 | frozen Model C | illustrative jaw projection',fontsize=16)
        fig.text(.5,.025,'Other five semantic controls are not rendered. Reference = 0.175 rad; signed jaw delta scale = 0.175 rad.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.015,right=.985,bottom=.11,top=.82,wspace=.04)
        fig.savefig(destination/'model_c_jaw_preview.png',dpi=150)
        plt.close(fig)
        frames=[]
        for result in predictions:
            # Every GIF frame is decoded with FLAME at its actual intermediate angle.
            phases=np.sin(np.linspace(0,np.pi,18))**2
            angles=REFERENCE_JAW+(result['projection']['target_jaw_radians']-REFERENCE_JAW)*phases
            with torch.no_grad():
                poses=torch.zeros(len(angles),3,device=self.device)
                poses[:,0]=torch.tensor(angles,device=self.device)
                meshes=self.flame(jaw_pose=poses)[0].cpu().numpy()
            for vertices in meshes:
                fig,axes=plt.subplots(1,2,figsize=(7.2,4),facecolor='#f5f8fb',dpi=90)
                panel(axes[0],reference,faces,'Reference',bounds)
                panel(axes[1],vertices,faces,'Model C jaw preview',bounds)
                fig.suptitle(result['text'],fontsize=13)
                fig.text(.5,.025,'FLAME2019 | jaw only | illustrative mapping',ha='center',fontsize=9)
                fig.subplots_adjust(left=.02,right=.98,bottom=.10,top=.87)
                fig.canvas.draw()
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
                plt.close(fig)
        frames[0].save(destination/'model_c_flame_animation.gif',save_all=True,append_images=frames[1:],duration=90,loop=0)
        save_obj(destination/'reference.obj',reference,faces)
        for i,result in enumerate(predictions):
            save_obj(destination/f'model_c_prompt_{i+1}.obj',np.array(result['vertices']),faces)
        record=dict(flame_sha256=self.flame.model_sha256,device=self.device,checkpoints=self.checkpoints,
            mapping=jaw_projection([0.]*6),predictions=[{k:v for k,v in r.items() if k!='vertices'} for r in predictions],
            mesh_vertex_count=len(reference),mesh_face_count=len(faces),animation_frames=len(frames),
            animation_method='Each frame independently FLAME-decoded at an interpolated jaw angle',
            ringnet_predictor_used=False,semantic_mapping_complete=False)
        (destination/'run.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print('Exported FLAME PNG, GIF, OBJ meshes and provenance to '+str(destination),flush=True)
