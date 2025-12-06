# Awesome Object-Centric Counterfactual World Models (OCCWM)

A curated list of papers and resources on **Object-Centric Counterfactual World Models**,
covering world models, object-centric representations, causal and counterfactual reasoning,
benchmarks, and implementations.

> This repository is a companion to the survey  
> **"Object-Centric Counterfactual World Models: Foundations, Methods, and Recent Advances"**.

---

## Table of Contents

- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [Methods by Benchmark](#methods-by-benchmark)
- [Typical Intervention Types in OCCWM](#intervention-in-OCCWM)

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

## Object-Centric Counterfactual World Models (OCCWM, 2020-2025)  Object-Centric (OC) Counterfactual (CF)

| Year | Paper | OC? | CF? | Category | Benchmarks / Datasets | Code |
|------|--------|-----|-----|----------|------------------------|------|
| 2019 | [SCALOR](https://arxiv.org/abs/1910.02384) | ✔️ | ✖️ | OCWM | Synthetic videos | – |
| 2019 | [C-SWM](https://arxiv.org/abs/1911.12247) | ✔️ | ✖️ | OCWM | Gridworld, Atari, physics | [Link](https://github.com/tkipf/c-swm) |
| 2020 | [G-SWM](https://arxiv.org/abs/2007.09571) | ✔️ | Weak | OCWM / OCCWM | Synthetic physics | [Link](https://github.com/zhixuan-lin/G-SWM) |
| 2020 | [OP3](https://arxiv.org/abs/2007.05309) | ✔️ | ✖️ | OCWM | MuJoCo visual control | [Link](https://github.com/jcoreyes/OP3) |
| 2020 | [SILOT](https://arxiv.org/abs/1911.09033) | ✔️ | ✖️ | OCWM | Synthetic moving objects | [Link](https://github.com/e2crawfo/silot) |
| 2020 | [CLEVRER](https://arxiv.org/abs/1910.01442) | ✔️ | ✔️ | OCCWM | CLEVRER CF-QA | [Link](https://github.com/chuangg/CLEVRER) |
| 2021 | [CGN](https://arxiv.org/abs/2101.06046) | ✖️ | ✔️ | CF-WM | C-MNIST, ImageNet-10 | [Link](https://github.com/autonomousvision/counterfactual_generative_networks) |
| 2021 | [Physion](https://arxiv.org/abs/2106.08261) | ✔️ | ✔️ | OCCWM | Physion | [Link](https://github.com/cogtoolslab/physics-benchmarking-neurips2021) |
| 2021 | [SAVi](https://openreview.net/forum?id=aD7uesX1GF_) | ✔️ | ✖️ | OCWM | CATER, MOVi | [Link](https://github.com/google-research/slot-attention-video) |
| 2022 | [SAVi++](https://arxiv.org/abs/2206.07764) | ✔️ | ✖️ | OCWM | MOVi, real videos | – |
| 2022 | [SlotFormer](https://arxiv.org/abs/2210.05861) | ✔️ | ✔️ | OCCWM | CLEVRER, Physion, PHYRE | [Link](https://github.com/pairlab/SlotFormer) |
| 2023 | [CWM](https://arxiv.org/abs/2306.01828) | ✖️ | ✔️ | CF-WM | COCO, DAVIS, Kinetics | [Link](https://github.com/neuroailab/CounterfactualWorldModels) |
| 2023 | [SlotDiffusion](https://arxiv.org/abs/2305.11281) | ✔️ | ✔️ | OCCWM | CLEVRER, Physion | [Link](https://github.com/Wuziyi616/SlotDiffusion) |
| 2023 | [VideoSAUR](https://arxiv.org/abs/2306.04829) | ✔️ | ✖️ | OCWM | MOVi → YTVIS, DAVIS | [Link](https://github.com/martius-lab/videosaur) |
| 2024 | [CWM-Physics](https://arxiv.org/abs/2312.06721) | ✔️ | ✔️ | OCCWM | Physion, Physion++ | [Link](https://github.com/rahulvenkk/cwm_dynamics) |
| 2024 | [CA-SA](https://arxiv.org/abs/2410.15728) | ✔️ | Weak | OCWM | MOVi, synthetic | [Link](https://github.com/Cmeo97/CA-SA) |
| 2024 | [COIL](https://openreview.net/pdf?id=dYjH8Nv81K) | ✔️ | ✔️ | OCCWM | MuJoCo manipulation | – |
| 2025 | [JEPA-Physics](https://arxiv.org/abs/2502.11831) | ✖️ | ✔️ | CF-WM / OCCWM | IntPhys, InfLevel | [Link](https://github.com/facebookresearch/jepa-intuitive-physics) |
| 2025 | [Opt-CWM](https://arxiv.org/abs/2503.19953) | ✖️ | ✔️ | CF-WM | Kinetics motion | [Link](https://github.com/neuroailab/Opt_CWM) |
| 2025 | [KL-Tracing](https://arxiv.org/abs/2507.09082) | ✖️ | ✔️ | CF-WM | Sintel, KITTI Flow | [Link](https://neuroailab.github.io/projects/kl_tracing/) |
| 2025 | [PointPrompting](https://arxiv.org/abs/2510.11715) | ✖️ | ✔️ | CF-WM | DAVIS, YT-VOS, TAP-Vid | [Link](https://point-prompting.github.io) |
| 2025 | [Dyn-O](https://arxiv.org/abs/2503.02161) | ✔️ | ✖️ | OCWM | Sim physics tasks | [Link](https://github.com/wangzizhao/dyn-o) |
| 2025 | [SlotPi](https://arxiv.org/abs/2506.10778) | ✔️ | Weak | OCWM | Synthetic physics | – |
| 2025 | [FOCUS](https://arxiv.org/abs/2310.19586) | ✔️ | Weak | OCWM | ManiSkill2, robosuite | [Link](https://github.com/StefanoFerraro/FOCUS) |
| 2025 | [CWMDT](https://arxiv.org/abs/2511.17481) | ✔️ | ✔️ | OCCWM | Digital Twin envs | – |

---

## Typical Intervention Types in OCCWM

| Intervention Type | Target Level | Implementation in OCCWM | Best Suited For | Typical Failure Mode |
|-------------------|--------------|---------------------------|------------------|-----------------------|
| **Object presence** | Object set | Add/remove object slot; mask out region; swap instance in simulator or generator | Testing object permanence, existence-based causality, counterfactual removal effects | Model confuses background and object; slot identity collapse; unrealistic collisions or shadows |
| **Attribute change** | Object attributes (color, shape, mass, friction, velocity) | Edit attribute latents; change simulator parameters for selected instances | Disentanglement of appearance vs. dynamics; visual vs. physical causal factors | Leakage across factors (shape change alters pose); dynamics ignore changed mass; texture-only shortcuts |
| **Relational change** | Edges/relations between objects (contact, support, occlusion) | Toggle edges in relational module; modify contact constraints; alter relative pose in simulator | Causal chains between objects; stability, support, collision outcomes | Incorrect relation inference; over-smoothing interactions; brittle under unseen configurations |
| **Causal context change** | Global context (gravity, lighting, domain, rules) | Edit context latents; change environment parameters; domain or task switches | Generalization across “worlds”; robustness to dynamics shifts; deconfounding evaluations | State and context entangled; performance collapses under unseen contexts; spurious correlation with visual style |
| **Action-level intervention** | Agent actions or policies | Roll out alternative action sequences; do-interventions on policy or control inputs | Planning & control; policy counterfactuals; explaining decisions with alternative trajectories | Model exploits reward structure instead of causal dynamics; compounding rollout errors; off-policy instability |

---
