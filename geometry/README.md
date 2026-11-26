# PyTorch FLAME integration

## Six-control geometric calibration — November 24, 2026

The live viewer now maps all six predicted controls to actual FLAME2019 meshes. Open `http://127.0.0.1:8765` and select **Six controls** for direct sliders, or **Text edits** for frozen A/B/C inference. `py -3 run_flame_live.py --check --launch` verifies the workflow and starts the local server.

`semantic_calibration.py` fits anatomical measurement proxies within the 100-dimensional expression basis at each requested jaw angle. It preserves nine target measurements and enforces three inner-lip separation constraints. Signed zero is an explicitly open-jaw reference, not FLAME's closed-jaw neutral. Smile is lip-corner lift; cheek raise is elevation of two documented surface patches. Engineering amplitude choices still require human perceptual validation. Some combined requests exceed the coefficient envelope and are rejected explicitly.

Run `py -3 calibrate_flame.py --animate` to reproduce the sweeps, coefficient export, landmark overlay, OBJ meshes, PNG/GIF previews and numeric validation under `geometry/calibration/`. Read [the calibration report](../reports/FLAME_CALIBRATION_2026-11-24.md) for measurements, provenance, results and limitations. This implementation supersedes the historical unmapped-control boundary below for the live viewer; the legacy illustrative adapter retains its original behavior.

The local FLAME 2019 geometry model now runs on CUDA with the installed PyTorch 2.9.1 / Python 3.12 environment and is the default after the user's RingNet compatibility clarification. FLAME 2020 and 2023 are also tested as distinct geometry models. FLAME model assets remain in the user's sibling `FLAME/` directory. The code reads them without changing them.

```powershell
py -3 -m geometry.run_demo
py -3 -m unittest geometry.test_flame -v
```

Current outputs under `geometry/output_2019/`: a neutral OBJ, four jaw-opening OBJs, `jaw_preview.png`, an expression-PC inspection sheet, and `run.json` with model hash and actual numeric checks. Earlier 2023 demonstration outputs remain under `geometry/output/` and are not interchangeable parameter bases. Static software previews need no interactive OpenGL window. The texture space is optional and is not used by these untextured geometry checks.

## Implementation and provenance

The upstream repository is cloned under `external/FLAME_PyTorch`, commit `5a85f60a77f9e0454109e79de0a6517c4b95d371`. Our geometry-only adapter follows its shape/expression concatenation, joint order and SMPL-X linear blend skinning. It originally omitted landmark computation; the new calibration layer separately loads the pinned official RingNet static embedding. It does not claim that the unmodified upstream landmark demo ran.

`smplx==0.1.28` is installed locally under `external/python_packages` with no dependency upgrades. Existing PyTorch/NumPy packages were retained. The legacy Chumpy model storage is read through a restricted numeric-state adapter; only stored arrays are accepted, and unknown pickle globals/computed Chumpy expressions are rejected. No legacy Chumpy installation or Python downgrade was needed.

Sources: [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch), [SMPL-X](https://github.com/vchoutas/smplx). Preserve upstream notices. Local model Readme documents and hashes are recorded in `research/FLAME_PREFLIGHT.json`. Using the model in a report requires the cited FLAME paper and applicable model terms. No model files are redistributed by this workflow.

## Historical semantic adapter boundary

The standard FLAME2019 `generic_model.pkl` is selected, SHA-256 `e7f4f6ac7aa97045b3989893386aa94dc5245ded7105bbbf44116bfdf801a050`. The [RingNet setup](https://github.com/soubhiksanyal/RingNet#download-models) specifies FLAME2019 for decoding RingNet outputs. This workflow does not run a RingNet image encoder or use its pretrained predictions. Its direct FLAME decoder supports different geometry bases, but any external coefficient bundle must carry the exact model hash. `decode_parameter_bundle` rejects mismatches and RingNet-tagged bundles with any non-2019 model. Coefficients/calibrations cannot be transferred across versions by changing a filename.

The standard 2023 model was used only for the earlier separate smoke demo. Its bundled README describes jaw rotation as separate from expression; `flame2023_no_jaw.pkl` mixes that variation into expression. The jaw demo rejects the no_jaw variant.

The only current semantic adapter is an explicitly illustrative positive jaw-opening scale: control 1.0 maps to 0.35 radians about the jaw's local x axis. This scale is not calibrated. Nonzero eyebrow, eye-opening, mouth-width, smile or cheek controls raise an error. Negative jaw changes also raise an error until a valid reference-state policy exists. They are never silently dropped, clamped or assigned to arbitrary expression PCs.

The expression preview labels raw PCs by index, not by semantic names. Next calibration requires selected anatomical landmarks/measurements, target displacements for each control, a documented fit over the expression basis, checks of cross-attribute effects, and human visual validation. Until then, this is a geometry/software demonstration, not six-control grounding evidence or a completed face-editing interface.
