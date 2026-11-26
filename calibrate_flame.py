"""Build, inspect and verify geometric-proxy calibration for the pinned FLAME2019."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry.flame_backend import FlameGeometry,ROOT
from geometry.semantic_calibration import SemanticCalibration,DIRECTORY,SCALES,FEATURE_NAMES
from geometry.run_demo import panel,save_obj


def main(inspect=False,animate=False):
    model=FlameGeometry(ROOT.parent/'FLAME/FLAME2019/FLAME2019/generic_model.pkl').to('cuda' if torch.cuda.is_available() else 'cpu')
    calibration=SemanticCalibration(model)
    DIRECTORY.mkdir(exist_ok=True)
    faces=model.faces.cpu().numpy();ref=calibration.reference
    bounds=(ref[:,0].min()-.012,ref[:,0].max()+.012,ref[:,1].min()-.025,ref[:,1].max()+.012)
    fig,ax=plt.subplots(figsize=(8,9))
    panel(ax,ref,faces,'Official landmarks and derived cheek patches',bounds)
    ax.scatter(calibration.lmk[:,0],calibration.lmk[:,1],c='#ef4366',s=12)
    for i,p in enumerate(calibration.lmk,17):
        ax.text(p[0],p[1],str(i),fontsize=7,color='#600025')
    for ids in calibration.cheek_patches:
        ax.scatter(ref[ids,0],ref[ids,1],c='#15ad8b',s=18)
    fig.savefig(DIRECTORY/'landmark_overlay.png',dpi=160);plt.close(fig)
    if inspect:
        print(json.dumps(calibration.specification(),indent=2),flush=True)
        return
    records=[];meshes=[];all_axis_meshes=[]
    quality=[]
    def assess(mesh,case):
        points=calibration.landmark_matrix@mesh
        eye_gaps=[float(points[u-17,1]-points[l-17,1]) for u,l in [(37,41),(38,40),(43,47),(44,46)]]
        inner_lip_gap=min(float(points[u-17,1]-points[l-17,1]) for u,l in [(61,67),(62,66),(63,65)])
        displacement=np.linalg.norm(mesh-ref,axis=1)
        # Eyelid pairs and displacement are independent of fitted measurements.
        # Lip gaps check the three additional inequality constraints.
        quality.append(dict(case=case,minimum_individual_eyelid_gap_m=min(eye_gaps),
            inner_lip_gap_m=inner_lip_gap,max_vertex_displacement_m=float(displacement.max()),
            neck_rms_displacement_m=float(np.sqrt(np.mean((mesh[ref[:,1]<-.10]-ref[ref[:,1]<-.10])**2)))))
    for j in range(6):
        column=[];axis_meshes=[]
        for level in [-1,-.75,-.5,-.25,0,.25,.5,.75,1]:
            controls=np.zeros(6);controls[j]=level
            mesh,record=calibration.decode(controls)
            axis_meshes.append(mesh)
            records.append(dict(kind='axis',attribute=j,level=level,**record))
            assess(mesh,f'axis_{j}_{level}')
            if level in (-1,0,1):
                column.append(mesh)
                save_obj(DIRECTORY/f'control_{j}_{"minus" if level<0 else "plus" if level>0 else "zero"}.obj',mesh,faces)
        meshes.append(column)
        all_axis_meshes.append(axis_meshes)
    names=['Eyebrow elevation','Eye openness','Mouth width','Lip-corner lift (smile proxy)','Jaw opening','Cheek patch elevation']
    fig,axes=plt.subplots(3,6,figsize=(18,12))
    for j in range(6):
        for i,level in enumerate([-1,0,1]):
            panel(axes[i,j],meshes[j][i],faces,f'{names[j]}\n{level:+d}',bounds)
    fig.suptitle('FLAME2019: measurement-fitted geometric controls\nEngineering amplitudes; not human-perceptual calibration',fontsize=17)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(DIRECTORY/'control_sweeps.png',dpi=145);plt.close(fig)
    if animate:
        from PIL import Image
        frames=[]
        for i,level in enumerate([-1,-.75,-.5,-.25,0,.25,.5,.75,1]):
            fig,axes=plt.subplots(2,3,figsize=(10,9),dpi=90)
            for j,ax in enumerate(axes.flat):
                panel(ax,all_axis_meshes[j][i],faces,f'{names[j]}: {level:+.2f}',bounds)
            fig.suptitle('Six separate FLAME2019 control sweeps\nGeometric proxies; human validation pending',fontsize=14)
            fig.tight_layout(rect=(0,0,1,.94));fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
            plt.close(fig)
        frames[0].save(DIRECTORY/'control_sweeps.gif',save_all=True,append_images=frames[1:]+frames[-2:0:-1],duration=220,loop=0)
    # Fixed random test suite, independent of fit constraints and of NLP labels.
    rng=np.random.default_rng(713)
    rejected=[]
    for i,controls in enumerate(rng.uniform(-1,1,(100,6))):
        try:
            mesh,record=calibration.decode(controls)
            records.append(dict(kind='mixed',case=i,**record))
            assess(mesh,f'mixed_{i}')
        except ValueError as error:
            rejected.append(dict(case=i,controls=controls.tolist(),reason=str(error)))
    checks=dict(status='pass' if not rejected else 'restricted_envelope',
        axis_cases=54,mixed_cases=100,mixed_rejected=rejected,
        max_normalized_feature_residual=max(r['max_normalized_feature_residual'] for r in records),
        max_abs_expression_coefficient=max(r['max_abs_coefficient'] for r in records),
        independent_geometry_checks=dict(
            minimum_individual_eyelid_gap_m=min(q['minimum_individual_eyelid_gap_m'] for q in quality),
            minimum_inner_lip_gap_m=min(q['inner_lip_gap_m'] for q in quality),
            maximum_vertex_displacement_m=max(q['max_vertex_displacement_m'] for q in quality),
            maximum_neck_rms_displacement_m=max(q['neck_rms_displacement_m'] for q in quality),
            full_collision_detection_performed=False),
        human_perceptual_validation=False,records=records)
    fit=calibration.basis_at(.175)
    np.savez(DIRECTORY/'reference_coefficients.npz',mapping=fit['mapping'],offset=fit['offset'],
             model_sha256=model.model_sha256,landmark_sha256=calibration.specification()['landmark_sha256'])
    (DIRECTORY/'geometry_quality.json').write_text(json.dumps(quality,indent=2))
    (DIRECTORY/'specification.json').write_text(json.dumps(calibration.specification(),indent=2))
    (DIRECTORY/'validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({k:v for k,v in checks.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--animate',action='store_true')
    args=parser.parse_args()
    main(args.inspect,args.animate)
