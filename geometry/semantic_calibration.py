"""Version-pinned geometric proxies for six semantic controls on FLAME2019.

Fits the actual 100-dimensional expression space. No mesh vertices are edited
after FLAME decoding. Calibration targets are engineering choices, not human
perceptual labels or learned FACS action units.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from geometry.flame_backend import ROOT, ModelUnpickler, RINGNET_FLAME2019_SHA256

DIRECTORY=ROOT/'geometry/calibration'
LANDMARK_SHA256='2881bdad8e61e87aad83a3df2e04655c534694089f68ca7144e7350da9e8ad62'
FEATURE_NAMES=['brow_left','brow_right','eyelid_gap_left','eyelid_gap_right','mouth_width',
               'lip_corner_lift_left','lip_corner_lift_right','cheek_patch_left','cheek_patch_right']
# Modest engineering displacement budgets, in native FLAME units (metres).
SCALES=np.array([.004,.004,.002,.002,.006,.003,.003,.002,.002])
FEATURE_CONTROL=[0,0,1,1,2,3,3,5,5]
REFERENCE_JAW=.175
JAW_SCALE=.175


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def landmarks(model):
    path=DIRECTORY/'assets/flame_static_embedding.pkl'
    if sha256(path)!=LANDMARK_SHA256:
        raise ValueError('Landmark embedding hash mismatch')
    with path.open('rb') as stream:
        data=ModelUnpickler(stream,encoding='latin1').load()
    face=np.asarray(data['lmk_face_idx'],dtype=np.int64).reshape(-1)
    bary=np.asarray(data['lmk_b_coords'],dtype=np.float64)
    if face.shape!=(51,) or bary.shape!=(51,3) or not np.isfinite(bary).all() or not np.allclose(bary.sum(1),1,atol=1e-6):
        raise ValueError('Expected 51 validated static barycentric landmarks')
    faces=model.faces.cpu().numpy()
    if face.min()<0 or face.max()>=len(faces) or bary.min() < -1e-6:
        raise ValueError('Invalid landmark locations')
    matrix=np.zeros((51,len(model.v_template)))
    for i in range(51):
        np.add.at(matrix[i],faces[face[i]],bary[i])
    return matrix


class SemanticCalibration:
    def __init__(self,model):
        if model.model_sha256!=RINGNET_FLAME2019_SHA256:
            raise ValueError('This calibration requires the exact FLAME2019 generic model')
        self.model=model
        self.device=model.v_template.device
        self.landmark_matrix=landmarks(model)
        with torch.no_grad():
            self.reference=model(jaw_pose=torch.tensor([[REFERENCE_JAW,0,0]],device=self.device))[0][0].cpu().numpy().astype(np.float64)
        self.lmk=self.landmark_matrix@self.reference
        self.measurement=np.zeros((9,len(self.reference),3),dtype=np.float64)
        def lm(indices):
            return self.landmark_matrix[np.array(indices)-17].mean(0)
        self.measurement[0,:,1]=lm(range(17,22))-lm([36,39])
        self.measurement[1,:,1]=lm(range(22,27))-lm([42,45])
        self.measurement[2,:,1]=lm([37,38])-lm([40,41])
        self.measurement[3,:,1]=lm([43,44])-lm([46,47])
        sign=np.sign(self.lmk[54-17,0]-self.lmk[48-17,0])
        self.measurement[4,:,0]=sign*(lm([54])-lm([48]))
        self.measurement[5,:,1]=lm([48])-lm([51,57])
        self.measurement[6,:,1]=lm([54])-lm([51,57])
        self.cheek_patches=[]
        for row,eye,corner in [(7,[36,39],48),(8,[42,45],54)]:
            eye_pos=self.lmk[np.array(eye)-17].mean(0)
            mouth=self.lmk[corner-17]
            # Fixed reference-defined surface patch below/lateral to the eye.
            center=.60*eye_pos+.40*mouth
            center[0]+=np.sign(center[0])*.008
            eligible=(self.reference[:,1] < eye_pos[1]-.004)&(self.reference[:,1]>mouth[1]+.004)&(self.reference[:,2]>center[2]-.012)&(self.reference[:,0]*np.sign(center[0])>0)
            ids=np.flatnonzero(eligible)
            if len(ids)<12:
                raise ValueError('Could not identify a cheek surface patch')
            ids=ids[np.argsort(np.linalg.norm(self.reference[ids]-center,axis=1))[:16]]
            patch=np.zeros(len(self.reference));patch[ids]=1/len(ids)
            self.measurement[row,:,1]=patch-lm([27,28])
            self.cheek_patches.append(ids.tolist())
        self.W=self.measurement.reshape(9,-1)
        lip_measurement=np.zeros((3,len(self.reference),3))
        for i,(upper,lower) in enumerate([(61,67),(62,66),(63,65)]):
            lip_measurement[i,:,1]=lm([upper])-lm([lower])
        self.lip_W=lip_measurement.reshape(3,-1)
        self.minimum_lip_gap=.0002
        self.reference_features=self.W@self.reference.ravel()
        self.target_matrix=np.zeros((9,6))
        for i,j in enumerate(FEATURE_CONTROL):
            self.target_matrix[i,j]=SCALES[i]
        self.fields=self._target_fields()
        self.cache={}

    def _target_fields(self):
        """Smooth desired motions near features; no direct displacement is rendered."""
        v=self.reference
        fields=np.zeros((len(v),3,6))
        def patch(point,sigma):
            return np.exp(-np.sum((v-point)**2,axis=1)/(2*sigma*sigma))
        def pos(indices):
            return self.lmk[np.array(indices)-17].mean(0)
        for ids in (range(17,22),range(22,27)):
            fields[:,1,0]+=.004*patch(pos(ids),.012)
        for upper,lower in [([37,38],[40,41]),([43,44],[46,47])]:
            fields[:,1,1]+=.001*patch(pos(upper),.005)-.001*patch(pos(lower),.005)
        for index in (48,54):
            point=pos([index])
            fields[:,0,2]+=.003*np.sign(point[0])*patch(point,.012)
            fields[:,1,3]+=.003*patch(point,.010)
        for ids in self.cheek_patches:
            fields[:,1,5]+=.002*patch(v[ids].mean(0),.013)
        return fields.reshape(-1,6)

    @torch.no_grad()
    def basis_at(self,jaw):
        # Fixed jaw rotations make this decoder affine in expression coefficients.
        jaw=float(jaw)
        key=round(jaw,8)
        if key in self.cache:
            return self.cache[key]
        expressions=torch.cat([torch.zeros(1,100),torch.eye(100)]).to(self.device)
        poses=torch.zeros(101,3,device=self.device);poses[:,0]=jaw
        vertices=self.model(expression=expressions,jaw_pose=poses)[0].cpu().numpy().astype(np.float64)
        baseline=vertices[0].ravel()
        D=(vertices[1:]-vertices[0]).reshape(100,-1).T
        # Mean full-mesh displacement penalizes leakage. Small coefficient ridge
        # prevents the nine feature constraints using poorly supported directions.
        gram=D.T@D/len(baseline)
        ridge=.02*np.trace(gram)/100
        H=gram+ridge*np.eye(100)
        factor=cho_factor(H)
        A=self.W@D
        HA=cho_solve(factor,A.T)
        constraint=A@HA
        condition=float(np.linalg.cond(constraint))
        if not np.isfinite(condition) or condition>1e10:
            raise ValueError('Ill-conditioned anatomical feature fit')
        gain=HA@np.linalg.inv(constraint)
        local=cho_solve(factor,D.T@self.fields/len(baseline))
        mapping=local+gain@(self.target_matrix-A@local)
        offset=gain@(self.reference_features-self.W@baseline)
        inverse=cho_solve(factor,np.eye(100))
        null_inverse=inverse-HA@np.linalg.solve(constraint,HA.T)
        lip_A=self.lip_W@D
        value=dict(baseline=baseline,D=D,mapping=mapping,offset=offset,condition=condition,ridge=ridge,
                   H=H,lip_A=lip_A,lip_baseline=self.lip_W@baseline,
                   lip_gain=null_inverse@lip_A.T)
        # Bound the exploratory cache instead of keeping arbitrary text requests.
        if len(self.cache)>12:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key]=value
        return value

    @torch.no_grad()
    def decode(self,controls):
        if isinstance(controls,torch.Tensor):
            controls=controls.detach().cpu().numpy()
        values=np.asarray(controls)
        if values.shape!=(6,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or (np.abs(values)>1).any():
            raise ValueError('Six finite numeric controls in [-1,1] required')
        values=values.astype(np.float64)
        jaw=REFERENCE_JAW+JAW_SCALE*values[4]
        fit=self.basis_at(jaw)
        expression=fit['offset']+fit['mapping']@values
        # Solve the tiny convex active-set problem in the nullspace of the nine
        # semantic constraints. This preserves targets while preventing measured
        # inner-lip crossing; it is not full surface collision detection.
        gap=fit['lip_baseline']+fit['lip_A']@expression
        active=[]
        if np.min(gap)<self.minimum_lip_gap:
            candidates=[]
            for count in range(1,4):
                for subset in itertools.combinations(range(3),count):
                    ids=list(subset)
                    gain=fit['lip_gain'][:,ids]
                    system=fit['lip_A'][ids]@gain
                    if np.linalg.cond(system)>1e10:
                        continue
                    delta=gain@np.linalg.solve(system,self.minimum_lip_gap-gap[ids])
                    if np.min(gap+fit['lip_A']@delta)>=self.minimum_lip_gap-1e-8:
                        candidates.append((float(delta@fit['H']@delta),delta,ids))
            if not candidates:
                raise ValueError('No feasible inner-lip separation fit for requested controls')
            _,delta,active=min(candidates,key=lambda item:item[0])
            expression=expression+delta
        if np.max(np.abs(expression))>3.0:
            raise ValueError('Requested combination exceeds the calibrated expression coefficient envelope (3.0); reduce its intensity')
        mesh=self.model(expression=torch.tensor(expression[None],dtype=torch.float32,device=self.device),
                        jaw_pose=torch.tensor([[jaw,0,0]],device=self.device))[0][0].cpu().numpy()
        measured=self.W@mesh.ravel()
        residual=(measured-self.reference_features-self.target_matrix@values)/SCALES
        if not np.isfinite(mesh).all() or np.max(np.abs(residual))>.03:
            raise ValueError('Actual decoded mesh failed the 3% geometric target tolerance')
        lip_gaps=self.lip_W@mesh.ravel()
        if np.min(lip_gaps)<self.minimum_lip_gap-1e-6:
            raise ValueError('Actual decoded mesh failed inner-lip separation tolerance')
        record=dict(calibration='FLAME2019-geometric-proxies-v1',flame_sha256=self.model.model_sha256,
            landmark_sha256=LANDMARK_SHA256,requested_controls=values.tolist(),applied_controls=values.tolist(),
            rendered_control_indices=list(range(6)),unrendered_control_indices=[],
            target_jaw_radians=jaw,reference_jaw_radians=REFERENCE_JAW,
            expression_coefficients=expression.tolist(),max_abs_coefficient=float(np.max(np.abs(expression))),
            normalized_feature_residual=residual.tolist(),max_normalized_feature_residual=float(np.max(np.abs(residual))),
            inner_lip_gaps_m=lip_gaps.tolist(),active_lip_constraints=active,
            geometric_mapping_complete=True,semantic_mapping_complete=False,human_perceptual_validation=False,
            proxy_definitions='Brow elevation, eyelid gap, lip width, lip-corner lift, jaw rotation, cheek-patch elevation; all relative to a fixed open-jaw reference')
        return mesh,record

    def specification(self):
        return dict(version='FLAME2019-geometric-proxies-v1',model_sha256=self.model.model_sha256,
            landmark_sha256=LANDMARK_SHA256,
            landmark_source='https://github.com/soubhiksanyal/RingNet/blob/master/flame_model/flame_static_embedding.pkl',
            static_landmark_convention='51 inner iBUG landmarks, global indices 17 through 67',
            feature_names=FEATURE_NAMES,feature_scales_m=SCALES.tolist(),feature_controls=FEATURE_CONTROL,
            reference_jaw_radians=REFERENCE_JAW,jaw_scale_radians=JAW_SCALE,
            cheek_vertex_ids=self.cheek_patches,reference_features_m=self.reference_features.tolist(),
            ridge_fraction_of_mean_basis_energy=.02,max_abs_expression_coefficient=3.,
            minimum_inner_lip_gap_m=self.minimum_lip_gap,inner_lip_landmark_pairs=[[61,67],[62,66],[63,65]],
            fit='Spatially regularized FLAME expression fit with nine exact linear geometric constraints at the requested jaw angle',
            scales_status='Engineering choices; not human-calibrated intensity or perceptual measurements',
            semantic_mapping_complete=False,human_perceptual_validation=False)
