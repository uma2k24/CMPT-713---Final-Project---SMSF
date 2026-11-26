# Research kickoff

Researcher | 2026-11-23 | Setup complete; implementation/data acquisition not started.

The initial search verifies the proposal's foundations. It is a bounded review, not evidence that no competing work exists. No assets were downloaded and no account/license agreements were accepted.

## Proposal references

| Reference and verified primary source | Finding and use in this project | Limit of transfer |
| --- | --- | --- |
| Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21(140):1–67. [Paper page](https://www.jmlr.org/papers/v21/20-074.html); [official code](https://github.com/google-research/text-to-text-transfer-transformer) | Establishes T5 pretraining/transfer as the foundation for B/C. | Our pooled encoder with classifier/regressor heads is a task adaptation; the paper's text-to-text framing does not itself validate these geometry heads. |
| Jiang et al. (2021), *Talk-to-Edit: Fine-Grained Facial Editing via Dialog*, ICCV. [Author project](https://www.mmlab-ntu.com/project/talkedit/); [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf); [official code](https://github.com/yumingj/Talk-to-Edit) | Uses dialogue and a learned semantic field for fine-grained face-image manipulation; motivates intensity and preservation. | Its StyleGAN latent editing and image attribute labels do not supply our six-dimensional semantic gold vectors. |
| Li et al. (2017), *Learning a Model of Facial Shape and Expression from 4D Scans*, TOG 36(6). [Official project](https://flame.is.tue.mpg.de/); [paper URL](https://download.is.tue.mpg.de/flame/flame_paper.pdf) | FLAME combines learned shape/expression spaces and articulated facial pose. It supports optional geometry demonstrations. | Independent semantic dimensions need calibration to statistical shape/expression/pose controls. The paper download exceeded the browser fetch limit; this kickoff relies on the official project abstract, not an asserted full-paper reading. |
| Keysers et al. (2020), *Measuring Compositional Generalization: A Comprehensive Method on Realistic Data*, ICLR. [Google Research](https://research.google/pubs/measuring-compositional-generalization-a-comprehensive-method-on-realistic-data/); [paper](https://arxiv.org/pdf/1912.09713); [CFQ repository](https://github.com/google-research/google-research/tree/master/cfq) | Motivates separating familiar primitives from unseen compounds and measuring split composition rather than using random text splitting alone. | Our held-out attribute-combination test is inspired by this idea; it is not automatically a CFQ maximum-compound-divergence split. |
| Yin et al. (2023), *Consistency Regularization Training for Compositional Generalization*, ACL, pp.1294–1308. [Metadata](https://aclanthology.org/2023.acl-long.72/); [paper](https://aclanthology.org/2023.acl-long.72.pdf) | Studies representation consistency across samples and prediction consistency under dropout, evaluated in semantic parsing/translation. | Our squared continuous-vector agreement on different paraphrases is an adaptation. It is not the same loss as their contrastive representation and output-distribution regularization. B/C need equal data and comparable budgets to isolate it. |

## Dataset decision

Recommendation (research judgment): build the proposal's custom paired instruction/control corpus as the primary NLP benchmark. Existing face datasets below are supplementary evidence or optional geometry resources. None verified here directly supplies all six controls with the proposed continuous signed labels and independently authored editing instructions.

| Candidate / exact official source | Available supervision and suitable use | Access/terms checked | Decision |
| --- | --- | --- | --- |
| **Custom synthetic + Test-Human corpus**, specified by our proposal | Exact six-control vectors from grammar; verified paraphrases; independent human instructions with dual annotation. | Must record generation provenance, real human verification and permission to use/release human contributions. These steps are pending. | **Primary**. Follow contract counts: 800/100/100 plus 100 Test-Comp bases; 150–200 human instructions. |
| [CelebA-Dialog](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html); [official dataset repo](https://github.com/ziqihuangg/CelebA-Dialog) | Face images, captions and editing requests with five degree-labelled attributes: Bangs, Eyeglasses, Beard, Smiling, Age. Useful to study natural phrasing/intensity, especially smiling. | Official page provides download route; non-commercial research and redistribution restrictions apply. Do not bundle it with our released dataset. | **Optional reference**. Most attributes do not match ours; labels are not continuous 3D parameters. Any reuse/derived release needs specific terms review. |
| [Registered D3DFACS via FLAME](https://flame.is.tue.mpg.de/); [data terms](https://flame.is.tue.mpg.de/datalicense.html); [author data summary](https://sites.google.com/site/bolkartt/data) | 3D facial-expression sequences, with FLAME-topology registrations. Potential geometric examples for action-based controls. | Registered data is limited to non-commercial scientific research with access and distribution conditions. Model and data terms differ. | **Optional calibration evidence**. No verified instruction-to-six-vector supervision. Validate any mapping rather than equating an AU label with a semantic control. |
| [CoMA](https://coma.is.tue.mpg.de/); [terms](https://coma.is.tue.mpg.de/license.html) | 20,466 meshes from 12 subjects with extreme expressions. Possible geometry sanity checks after NLP work. | Registration and license acceptance required; terms restrict use and distribution. | **Defer**. No paired editing instructions; would introduce fitting/annotation work. |
| [FaMoS / TEMPEH](https://tempeh.is.tue.mpg.de/); [terms](https://tempeh.is.tue.mpg.de/license.html) | Dynamic 3D head sequences from 95 subjects with FLAME-topology registrations. Broad geometry examples. | Research download requires sign-up/terms; separate data/model/software restrictions recorded on official site. | **Defer**. More geometry than the NLP milestone requires; not a six-control instruction dataset. |
| [CFQ](https://github.com/google-research/google-research/tree/master/cfq) | Natural-language questions with logical-query targets; compositional split design examples. | Public repository and download route verified. Asset-specific reuse license not audited in this kickoff because no acquisition is recommended. | **Methodology only**; not facial supervision. |

FLAME is a model/resource, not an instruction dataset. The official site lists a FLAME 2023 Open release with CC-BY-4.0 and also links an academic license for earlier releases. **Do not assume all FLAME versions share one license.** Freeze the exact model file and its accompanying terms before use; model, D3DFACS data and textures must be checked separately. No license acceptance occurred here.

## Additional related-work correction

A later *Talk-to-Edit: Fine-Grained 2D and 3D Facial Editing via Dialog* journal extension exists. [Author publication page](https://ziqihuangg.github.io/) links [author PDF](https://ziqihuangg.github.io/papers/2023TPAMI-TalktoEdit3D.pdf) and the publisher; DOI [10.1109/TPAMI.2023.3347299](https://doi.org/10.1109/TPAMI.2023.3347299). The author page dates acceptance to 2023; the issue is 2024. The author PDF exceeded the browser fetch limit, so detailed method comparison is pending full-paper inspection. This already rules out an unqualified claim that all prior Talk-to-Edit work is only 2D.

Safer project positioning: evaluate continuous semantic-control grounding, paraphrase invariance and held-out edit combinations under one controlled A/B/C comparison. Whether this exact combination is new requires a broader literature search; no first-ever claim is established.

## Handoffs and gates

- **Data Builder:** use custom gold vectors; do not import CelebA-Dialog labels as our values. Keep semantic controls distinct from physical feasibility. Explicitly separate smile intensity and mouth width in annotation examples.
- **Agent B / Agent C:** Yin is conceptual motivation, not an identical continuous paraphrase objective. Specify reductions and supervised exposure equally.
- **Eval:** audit primitive/compound coverage, group leakage and preservation failures. Pairwise consistency alone cannot establish grounding quality.
- **Doc:** cite the specific publication versions. Record research limitations and optional geometry status.

## Pending work

1. Read the full FLAME and later Talk-to-Edit papers through an available document route before detailed geometric/novelty claims.
2. Freeze intensity conventions and sign semantics with Data Builder; agree on a geometric feasibility note.
3. Once rendering is actually scheduled, select one FLAME release, verify exact accompanying terms, then design a small calibrated mapping with visual checks.
4. Expand novelty search around language-conditioned semantic facial controls and paraphrase consistency. Do not delay the primary custom NLP benchmark for large geometry downloads.

Acceptance status: all five proposal references resolved to primary sources; initial dataset decision made; exact asset access, calibration and exhaustive novelty review remain open. No human dataset or model result exists from this research step.
