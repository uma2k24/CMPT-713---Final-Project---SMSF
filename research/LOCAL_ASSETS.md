# Local CelebA-Dialog and FLAME asset review

Inspected November 23, 2026 after the user added the downloads. Evidence: `LOCAL_ASSET_INVENTORY.json`, the coordinator's metadata-only inventory script, and `LOCAL_ASSET_README.txt` extracted from the supplied texture README. Original downloads were preserved. No model was executed, no pickle was loaded, no external assets were downloaded, and no license agreement was accepted on the user's behalf.

## What is present

| Asset | Observed local contents | Current use |
|---|---|---|
| `../CelebA-Dialog (HQ)/` | 30,000 JPGs; captions, requests and annotated-request JSON; fine-grained and binary classification labels; identity/mapping metadata; image and mask archives | Supplementary language/resource analysis; not a release of this project's six-control gold data |
| `../TextureSpace/FLAME_texture.npz` | 1,264,785,156 bytes; UV coordinates `vt` (5118,2), texture face indices `ft` (9976,3), mean RGB texture (512,512,3), and 200 texture PCA directions (512,512,3,200) | Optional appearance model after compatible geometry and rendering are available |
| `../TextureSpace/Readme.pdf` | Describes the MPI FLAME texture space and its accompanying terms | Exact local provenance/terms evidence for the supplied texture |

The three CelebA-Dialog JSON files each contain **29,999 filename keys**, while the image inventory contains 30,000 JPGs. A filename-key join and explicit missing-ID audit are needed before reuse. The inventory stores JSON SHA-256 values; it does not claim a bytewise image/archive integrity audit or full NPZ-array validation.

No FLAME geometry model was found in the inspected project files. This is a statement about the inspected location, not about downloads elsewhere on the user's computer. UV coordinates and texture triangle indices are not a deformable 3D head model. The official texture-sampling example supplies a geometry model such as `generic_model.pkl` separately from `FLAME_texture.npz`; its geometry parameters include shape, expression and pose. [Official FLAME implementation](https://github.com/TimoBolkart/TF_FLAME)

The official dataset repository identifies HQ as the 30,000-image subset, distinct from the 202,599-image standard release. Its fine-grained attributes are Bangs, Eyeglasses, Beard, Smiling and Age; the local field names include `No_Beard` and `Young`. The extra 40 binary labels are image attributes. [Official CelebA-Dialog repository](https://github.com/ziqihuangg/CelebA-Dialog)

## Compatibility with the frozen task

The local annotated requests contain `text`, `request_mode`, `attribute`, `attr_label`, `score_change_direction`, `score_change_value` and `target_score`. These describe the source dataset's discrete attribute scale and image context. They are not automatically signed changes from this project's neutral reference, and a scalar source score must not be divided by a convenient constant and relabeled as six-dimensional gold.

| Project control | Relevant local source material | Why direct label transfer is invalid |
|---|---|---|
| Eyebrow elevation | Binary appearance labels such as Arched_Eyebrows | Brow appearance is not a signed elevation edit |
| Eye openness | Binary Narrow_Eyes | A binary image state is not an edit magnitude or preservation decision |
| Mouth width | Mouth-related captions and binary labels | Mouth opening, lip size and smile wording can refer to different controls |
| Smile intensity | Fine-grained Smiling requests and labels | Closest semantic overlap, but source states/targets and scale differ |
| Jaw opening | Some smile descriptions and Mouth_Slightly_Open | Visible mouth opening is not calibrated jaw articulation |
| Cheek raise | High_Cheekbones and related appearance labels | Stable anatomy is not an instruction to raise the cheeks |

This mapping is a research assessment based on the inspected fields and project specification. For example, an inspected smiling request also describes an open mouth and visible teeth; treating its single Smiling label as a pure smile-intensity edit would discard potentially relevant jaw/mouth meaning. Actual target labels require annotation against the frozen guidelines, not keyword matching.

CelebA-Dialog can support a separately labeled supplementary corpus or qualitative language analysis. It does **not** fulfill the proposal's Test-Human requirement: that set requires newly independent instructions from people unfamiliar with our templates and two independent project-author annotations. It also supplies no verification of our generated paraphrase pairs. Keep it separate from the custom Train/Val/test release until an explicit experimental amendment defines its role, annotation and leakage controls. If a supplementary study is added, preserve filename/identity provenance and check image/identity overlap rather than assuming text rows are independent.

## Asset-specific terms and citations

The supplied texture README identifies Creative Commons BY-NC-SA 4.0, non-commercial use, and attribution to StyleGAN/FFHQ and the texture-construction repository. The current official texture page also states attribution/change notice/share-alike conditions and additional use restrictions. These are texture terms, not evidence that any geometry release has the same license. No texture derivatives were produced or redistributed during this inspection. [Official texture terms](https://flame.is.tue.mpg.de/texturelicense.html), [texture-construction repository](https://github.com/HavenFeng/photometric_optimization)

CelebA-Dialog's official agreement limits use to non-commercial research and restricts copying, publication and distribution, with an internal same-site exception. Keep the source assets out of a public code/data bundle; the project can record local paths and aggregate metadata. Any eventual derived-data release needs its own terms review. [Official dataset agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html)

## Next actions

1. Complete the filename-key/count audit before reading CelebA-Dialog into any pipeline. Use source IDs rather than positional joins.
2. Keep the primary synthetic candidates, paraphrase-review queue and human collection protocol unchanged. The downloads do not unblock G2 by themselves.
3. For optional geometry work, locate the exact FLAME geometry model and its release terms, then establish the required topology/landmark/rendering assets. The supplied texture alone is insufficient.
4. Calibrate each of the six semantic controls to geometry with explicit feasibility checks and fixed identity/camera settings. Do not call the semantic output vector native FLAME coefficients.
5. Finish the core NLP comparison before spending milestone time on optional textured rendering. No neural, human-language or geometry performance result is implied by having these files locally.
# Completed filename join audit

## Superseding geometry update

The user subsequently supplied FLAME2019, FLAME2020 and FLAME2023 geometry files under the sibling `FLAME/` directory; texture data moved to `FLAME/TextureSpace/`. The previous texture-only finding below describes the earlier inventory. Current model hashes are in `FLAME_PREFLIGHT.json`. FLAME2019 generic is selected for the current demo after the user's RingNet compatibility clarification, while independent tests cover all three releases as distinct models. `geometry/README.md` explains the decoder and coefficient-version checks. No RingNet image predictor was run.

The 6,909 smiling requests are now exported as original-label supplementary references under `celeba_references/`; they remain outside custom Train/Val/Test-Human and have no inferred six-control gold.

Coordinator reran `research/inventory_assets.py` after the initial review. All three text JSON key sets are identical. The only image missing text annotations is `5380.jpg`; no text key lacks an image. Annotated request counts: Young 6,918; Bangs 6,974; Eyeglasses 6,940; Smiling 6,909; No_Beard 2,258. Request modes: target 10,070; change_definite 9,928; change_indefinite 10,001. The inventory includes IDs of the 6,909 smiling requests as a reference index, without relabeling or copying them into Train/Val/Test-Human.

