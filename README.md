# world_model
object-centric world model


# Awesome Object-Centric Counterfactual World Models (OCCWM)

A curated list of papers and resources on **Object-Centric Counterfactual World Models**,
covering world models, object-centric representations, causal and counterfactual reasoning,
benchmarks, and implementations.

> This repository is a companion to the survey  
> **"Object-Centric Counterfactual World Models: Foundations, Methods, and Recent Advances"**.

---

## Table of Contents

- [Foundations](#foundations)
- [Core OCCWM Methods](#core-occwm-methods)
- [Methods by Benchmark](#methods-by-benchmark)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [Libraries and Tooling](#libraries-and-tooling)
- [Surveys and Related Awesome Lists](#surveys-and-related-awesome-lists)

---

| Benchmark / Dataset                                                                 | Domain                 | Counterfactual signal                                             | Typical metrics                            | Pitfalls / failure modes                                                                                 |
|-------------------------------------------------------------------------------------|------------------------|-------------------------------------------------------------------|--------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Counterfactual-native / intervention-grounded**                                   |                        |                                                                   |                                            |                                                                                                          |
| [CLEVRER](https://clevrer.csail.mit.edu/)                                           | Synthetic video physics | Programmatic “what-if” video QA; alternative event programs       | QA accuracy; event graph consistency       | Models can exploit QA shortcuts; correct answers ≠ correct underlying dynamics or causal structure      |
| [PHYRE](https://github.com/facebookresearch/phyre)                                  | 2D physics + planning  | Explicit action-as-intervention in physics puzzles                | Success rate; attempts-to-solve; sample eff.| Stylized 2D; low visual realism; perception and planning tightly coupled                                |
| [Kubric / MOVi](https://github.com/google-research/kubric)                          | Synthetic multi-object videos | Direct edits to simulator parameters yield paired rollouts | Segmentation / depth / flow; CF consistency | Simulator bias vs. real videos; need to report train/test distribution shifts explicitly                |
| [CausalWorld](https://github.com/rr-learning/CausalWorld)                           | Robotic manipulation sim | do-interventions on causal task variables & environment params  | Success rate; return; OOD generalization   | Sim-to-real gap; causal variables are “too clean” compared to real robot sensory streams                |
| Causal3DIdent                                                                       | Static images (3D scenes) | Controlled latent-factor shifts; paired views of scenes        | Identifiability; factor recovery metrics    | Mostly static (no dynamics); limited contact physics or long-horizon reasoning                          |
| **Physics plausibility (possible vs. impossible)**                                  |                        |                                                                   |                                            |                                                                                                          |
| [Physion](https://physion-benchmark.github.io/)                                     | Synthetic video physics | Protocol-defined interventions & alternative rollouts             | Prediction error; human-alignment scores    | Heuristics can work surprisingly well; intervention protocols differ between papers                     |
| IntPhys                                                                             | Synthetic video physics | Possible vs. impossible physical events                           | AUC / accuracy                             | Scores can be gamed by low-level cues (e.g., motion statistics)                                         |
| IntPhys 2                                                                           | Synthetic video physics | Harder violation-of-expectation style tests                       | AUC / accuracy                             | Still synthetic; unclear how performance transfers to real-world physics                                |
| **Real-world objectness (stress tests, not CF GT)**                                 |                        |                                                                   |                                            |                                                                                                          |
| [DAVIS](https://davischallenge.org/davis2017/code.html)                             | Real video segmentation | Occlusion / cut–paste protocols sometimes used as “pseudo-CF”    | Jaccard (J), F-score, identity switches    | Strongly supervised; no ground-truth causal counterfactuals; dominated by low-level segmentation issues |
| [YouTube-VOS](https://youtube-vos.org/)                                             | Real video segmentation | Long-horizon object ID stress; identity-preservation protocols   | J, F; ID switches                          | Category imbalance; annotation noise; identity drift dominates causal/physical reasoning                |
| **Embodied / RL (implicit CF via alternative actions / domains)**                   |                        |                                                                   |                                            |                                                                                                          |
| ALE (Atari)                                                                         | Pixel-based RL         | Different action sequences as implicit counterfactuals            | Return; sample efficiency; OOD performance | Highly stylized; reward hacking; partial observability; overfitting to specific game seeds              |
| dm\_control                                                                         | Continuous control RL  | Alternative actions; param changes in simulator                   | Return; success rate; robustness           | Physics mismatch; performance dominated by reward shaping & low-level control issues                    |
| [Procgen](https://github.com/openai/procgen)                                        | Procedural RL          | Train/test seeds as different “worlds” (environment CFs)          | Return; generalization gap                 | Agents can overfit RNG artifacts; narrow coverage of real-world skills                                  |
| [Meta-World](https://meta-world.github.io/)                                         | Robotic manipulation   | Task / goal variations as CF task settings                        | Success rate; average return               | Task leakage between train/test; protocols vary widely across papers                                    |
| [RLBench](https://github.com/stepjam/RLBench)                                       | Robotic manipulation   | Task variations and multi-step goals                              | Success rate; imitation metrics            | Vision confounds and domain randomization can obscure dynamics modeling quality                         |
| [robosuite](https://github.com/ARISE-Initiative/robosuite)                          | Robotic manipulation   | Task + embodiment variations; domain shifts                       | Success rate; return; robustness           | Results sensitive to simulator details; reproducibility issues across forks / versions                  |
| RL Unplugged                                                                        | Offline RL (multi-domain) | Dataset shifts as alternative “worlds”                         | Normalized return; offline policy metrics  | Offline evaluation noisy; policy selection & overfitting to validation sets                             |

### World Models
- [2018] Ha & Schmidhuber – *World Models* (arXiv:1803.10122)  
- [2019] Hafner et al. – *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer)

## Object-Centric World Models
### Table 1. Recent Object-Centric World Models (OCWM, ~2019–2025)

## Object-Centric World Models (OCWM)

| Year | Paper Title | Code |
|------|-------------|------|
| 2019 | [SCALOR: Generative World Models with Scalable Object-Centric Representations](https://arxiv.org/abs/1910.02384) | – |
| 2019 | [C-SWM: Contrastive Learning of Structured World Models](https://arxiv.org/abs/1911.12247) | [Link](https://github.com/tkipf/c-swm)|
| 2020 | [G-SWM: Improving Generative Imagination in Object-Centric World Models](https://arxiv.org/abs/2007.09571) | [Link](https://github.com/zhixuan-lin/G-SWM) |
| 2020 | [OP3: Object-Oriented Perception, Prediction, and Planning](https://arxiv.org/abs/2007.05309) | [Link](https://github.com/jcoreyes/OP3) |
| 2020 | [SILOT: Spatially Invariant Learning of Object Tracking](https://arxiv.org/abs/2001.04918) | [Link](https://github.com/e2crawfo/silot) |
| 2021 | [SAVi: Conditional Object-Centric Learning from Video](https://arxiv.org/abs/2111.13723) | [Link](https://github.com/google-research/slot-attention-video) |
| 2022 | [SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos](https://arxiv.org/abs/2206.07764) | – |
| 2022 | [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861) | [Link](https://github.com/pairlab/SlotFormer) |
| 2023 | [SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models](https://arxiv.org/abs/2305.11281) | [Link](https://github.com/Wuziyi616/SlotDiffusion) |
| 2023 | [VideoSAUR: Object-Centric Learning for Real-World Videos](https://arxiv.org/abs/2306.04829) | [Link](https://github.com/martius-lab/videosaur) |
| 2024 | [Learning Physical Dynamics for Object-Centric Visual Prediction](https://arxiv.org/abs/2403.10079) | – |
| 2025 | [Dyn-O: Structured Object-Centric World Models](https://arxiv.org/abs/2503.02161) | [Link](https://github.com/wangzizhao/dyn-o) |
| 2025 | [SlotPi: Physics-informed Object-Centric Reasoning](https://arxiv.org/abs/2506.10778) | – |
| 2025 | [FOCUS: Object-Centric World Models for Robotic Manipulation](https://arxiv.org/abs/2310.19586) | [Link](https://github.com/StefanoFerraro/FOCUS) |

## Counterfactual World Models (CF-WM)

| Year | Paper Title | Code |
|------|-------------|------|
| 2021 | [Counterfactual Generative Networks (CGN)](https://arxiv.org/abs/2101.06046) | [Link](https://github.com/autonomousvision/counterfactual_generative_networks) |
| 2023 | [CWM: Unifying (Machine) Vision via Counterfactual World Modeling](https://arxiv.org/abs/2306.01828) | [Link](https://github.com/neuroailab/CounterfactualWorldModels) |
| 2024 | [CWM-Physics: Understanding Physical Dynamics with Counterfactual World Modeling](https://arxiv.org/abs/2312.06721) | [Link](https://github.com/rahulvenkk/cwm_dynamics) |
| 2025 | [Opt-CWM: Learning Motion Concepts by Optimizing Counterfactuals](https://arxiv.org/abs/2503.19953) | [Link](https://github.com/neuroailab/Opt_CWM) |
| 2025 | [KL-Tracing: Zero-Shot Optical Flow via Counterfactual Tracing](https://arxiv.org/abs/2507.09082) | [Link](https://neuroailab.github.io/projects/kl_tracing/) |
| 2025 | [Point Prompting: Counterfactual Tracking with Video Diffusion Models](https://arxiv.org/abs/2510.11715) | [Link](https://point-prompting.github.io) |
| 2025 | [CWMDT: Digital Twin-conditioned Counterfactual Video Diffusion](https://arxiv.org/abs/2511.17481) | – |
| 2025 | [Intuitive Physics via JEPA Pretraining](https://arxiv.org/abs/2502.11831) | [Link](https://github.com/facebookresearch/jepa-intuitive-physics) |

## Object-Centric Counterfactual World Models (OCCWM)

| Year | Paper Title | Code |
|------|------------------------|-----------|
| 2020 | [G-SWM: Improving Generative Imagination in Object-Centric World Models](https://arxiv.org/pdf/2010.02054) | [Link](https://github.com/zhixuan-lin/G-SWM) |
| 2020 | [CLEVRER: A Diagnostic Dataset for Video Reasoning](https://arxiv.org/abs/1910.01442) | [Link](https://github.com/chuangg/CLEVRER)|
| 2021 | [Physion: Evaluating Physical Prediction from Vision in Humans and Machines](https://arxiv.org/abs/2106.08261) | [Link](https://github.com/cogtoolslab/physics-benchmarking-neurips2021)|
| 2023 | [Unifying (Machine) Vision via Counterfactual World Modeling](https://arxiv.org/pdf/2306.01828) | [Link](https://github.com/neuroailab/CounterfactualWorldModels) |
| 2023 | [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861) | [Link](https://github.com/pairlab/SlotFormer) |
| 2023 | [SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models](https://arxiv.org/abs/2305.11281) | [Link](https://github.com/Wuziyi616/SlotDiffusion) |
| 2024 | [Understanding Physical Dynamics with Counterfactual World Modeling](https://arxiv.org/abs/2312.06721) | [Link](https://github.com/neuroailab/cwm_dynamics) |
| 2024 | [Object-Centric Temporal Consistency via Conditional Autoregressive Inductive Biases](https://arxiv.org/abs/2410.15728) | [Link](https://github.com/Cmeo97/CA-SA) |
| 2024 | [COIL: UNSUPERVISED OBJECT INTERACTION LEARNING WITH COUNTERFACTUAL DYNAMICS MODELS](https://openreview.net/pdf?id=dYjH8Nv81K) | – |
| 2025 | [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831) | [Link](https://github.com/facebookresearch/jepa-intuitive-physics) |
| 2025 | [CWMDT: Counterfactual World Modeling with Digital Twins-conditioned Video Diffusion](https://arxiv.org/pdf/2511.17481)| – |



### Causal and Counterfactual Foundations
- [2009] Pearl – *Causality* (book)  
- [2021] Schölkopf et al. – *Toward Causal Representation Learning*  
- [2024] Komanduri et al. – *From Identifiable Causal Representations to Controllable Counterfactual Generation*

---

## Core OCCWM Methods

### Vision Counterfactual World Models
- [2023] Bear et al. – *Unifying (Machine) Vision via Counterfactual World Modeling (CWM)*  
  - type: CF-WM, mask-based; domain: vision; bench: CLEVR-like, Physion-style  
- [2023] Venkatesh et al. – *Understanding Physical Dynamics with Counterfactual World Modeling*  
  - type: CF-WM for physics; bench: Physion

### Object-Centric World Models with Interventions
- [2020] G-SWM – *Improving Generative Imagination in Object-Centric World Models*  
- [2019] OP3 – *Entity Abstraction in Visual Model-Based RL*  

(… more entries …)

---

## Methods by Benchmark

### CLEVRER
- G-SWM (object-centric WM, synthetic physics)
- CWM (CF-WM with structured masking)
- ...

### Physion
- Physical CWM
- ...

### Kubric / MOVi
- Slot Attention
- DINOSAUR, SAVi++, VideoSAUR
- ...

(… continue for PHYRE, IntPhys, CausalWorld …)

---

## Benchmarks and Datasets
- CLEVRER – counterfactual video QA  
- Physion – intuitive physics prediction  
- PHYRE – physics+planning puzzles  
- Kubric / MOVi – synthetic video scenes with object GT  
- IntPhys / IntPhys2 – possible vs impossible physics  
- CausalWorld – robotic manipulation with do-interventions  

---

## Libraries and Tooling
- object-centric-library – OC models under distribution shifts  
- Awesome-World-Models, Awesome-World-Models-for-robots, etc. (related lists)

---

## Surveys and Related Awesome Lists
- Hitchhiker's Guide to World Models (paper + GitHub)  
- Awesome-World-Models (general)  
- Awesome-Causal-Learning / Awesome-Causality-in-CV  
