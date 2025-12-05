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

## Foundations

### World Models
- [2018] Ha & Schmidhuber – *World Models* (arXiv:1803.10122)  
- [2019] Hafner et al. – *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer)

### Object-Centric Representation Learning

## Object-Centric Counterfactual World Models — Papers & Code

| Year | Paper Title | Paper Link | Code Link |
|------|-------------|------------|-----------|
| 2023 | Unifying (Machine) Vision via Counterfactual World Modeling (CWM) | https://arxiv.org/abs/2310.01408 | https://github.com/neuroailab/CounterfactualWorldModels |
| 2024 | Understanding Physical Dynamics with Counterfactual World Modeling (CWM-Physics / CWM-Dynamics) | https://arxiv.org/abs/2312.06721 | https://github.com/neuroailab/cwm_dynamics |
| 2023 | SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models | https://arxiv.org/abs/2210.05861 | https://github.com/pairlab/SlotFormer |
| 2023 | SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models | https://arxiv.org/abs/2305.11281 | https://github.com/Wuziyi616/SlotDiffusion |
| 2020 | G-SWM: Improving Generative Imagination in Object-Centric World Models | https://arxiv.org/abs/2010.02050 | https://github.com/google-research/google-research/tree/master/g-swmm |
| 2019 | OP3: Object-Oriented Planning and Prediction | https://arxiv.org/abs/1912.08737 | https://github.com/deepmind/deepmind-research/tree/master/op3 |
| 2021 | Physion: Evaluating Physical Prediction from Vision in Humans and Machines | https://arxiv.org/abs/2106.08261 | Benchmark code: https://github.com/cogtoolslab/physics-benchmarking-neurips2021 <br> Evaluator: https://github.com/neuroailab/physion_evaluator |
| 2020 | CLEVRER: A Diagnostic Dataset for Video Reasoning | https://arxiv.org/abs/1910.01442 | Official dataset/tools: https://github.com/clevrer/clevrer |
| 2024 | CA-SA: Conditional Autoregressive Inductive Biases for Object-Centric Temporal Consistency | https://arxiv.org/abs/2410.15728 | https://github.com/Cmeo97/CA-SA |
| 2025 | Intuitive physics understanding emerges from self-supervised pretraining on natural videos (JEPA-Intuitive-Physics) | https://arxiv.org/abs/2502.11831 | https://github.com/facebookresearch/jepa-intuitive-physics |
| 2024 | COIL: Counterfactual Object Interaction Learning | (arXiv link 待补) | (code link 待补) |
| 20xx | CWMDT: Counterfactual World Modeling with Digital Twins | (link 待补) | (code link 待补) |

- [2019] MONet – *Unsupervised Scene Decomposition and Representation*  
- [2019] IODINE – *Multi-Object Representation Learning with Iterative VI*  
- [2020] Slot Attention – *Object-Centric Learning with Slot Attention*  

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
