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
