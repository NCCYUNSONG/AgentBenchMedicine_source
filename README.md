# AgentBenchMedicine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<!-- You can add CI badges, code coverage badges, etc. here -->

## Overview

**AgentBenchMedicine** provides three open-source benchmarks for evaluating multimodal and text-only large-language-model (LLM) agents in clinical reasoning tasks:

1. **AgentClinic** — simulates a multi-agent clinical diagnosis workflow via dialogues among four roles: **doctor**, **patient**, **measurement**, and **moderator**.
2. **MedAgentsBench** — a "hard" subset of complex multiple-choice medical questions requiring multi-step reasoning.
3. **Humanity’s Last Exam (HLE)** — a benchmark of highly complex and nuanced Biology/Medicine cases, presented as multiple-choice and short-answer questions.

---

## Table of Contents

* [Benchmarks and Datasets](#benchmarks-and-datasets)
  * [AgentClinic](#agentclinic)
  * [MedAgentsBench](#medagentsbench)
  * [Humanity’s Last Exam](#humanitys-last-exam)
* [Models and Agent Systems](#models-and-agent-systems)
  * [LLMs](#llms)
  * [Manus & OpenManus Variants](#manus--openmanus-variants)
* [Running the Benchmarks](#running-the-benchmarks)
  * [AgentClinic CLI](#agentclinic-cli)
  * [MedAgentsBench CLI](#medagentsbench-cli)
  * [Humanity’s Last Exam CLI](#humanitys-last-exam-cli)
* [License](#license)

---

## Benchmarks and Datasets

### AgentClinic

Simulates the clinical diagnosis process as Objective Structured Clinical Examinations (OSCEs) via a four-agent dialogue:

* **Doctor**: the LLM or agent under test, interacts with patient and measurement agents to arrive at a diagnosis within a fixed inference budget.
* **Patient, Measurement & Moderator**: driven by LLM for role-playing, providing history, lab results, and dialogue control.
* **Inference cap**: 20 message exchanges per scenario.

#### Datasets

1. **AgentClinic-MedQA**
   214 board-exam–style scenarios (MedQA corpus), each with history, symptoms, labs, and gold-standard diagnosis.

2. **AgentClinic-MIMIC-IV**
   200 de-identified inpatient cases from MIMIC-IV, adapted to OSCE format.

3. **AgentClinic-NEJM**
   120 image-centric cases (radiographs, clinical photos) from NEJM Case Challenges.

---

### MedAgentsBench

A "HARD" subset for complex medical QA, covering eight source datasets. Only questions with <50% correct rate in baseline LLMs were retained, ensuring high difficulty and requiring multi-step reasoning.

* **Total items**: 862 multiple-choice questions.
* **Sources**: MedQA, PubMedQA, MedMCQA, MedExQA, MMLU-Pro, MedBullets, MMLU, MedXpertQA (Reasoning & Understanding splits).

---

### Humanity’s Last Exam

A recently developed benchmark comprising highly complex and nuanced cases across multiple academic areas, presented as multiple-choice and short-answer questions. Created by a consortium of academics and domain experts, each question avoids shortcut cues and requires in-depth domain knowledge and deep reasoning. Frontier LLMs have consistently demonstrated low accuracy on this benchmark.

* **Biology/Medicine Subset**:
  - **222 text-only tasks**
  - **58 multimodal tasks**

---

## Models and Agent Systems

### LLMs

* **Text-only tasks**:
  * Llama-4-Maverick-17B-128E-Instruct-FP8
  * Qwen-3-235B-A22B-FP8
  * GPT-4.1

* **Multimodal tasks** (with vision support):
  * Llama-4-Maverick-17B-128E-Instruct-FP8
  * Gemma-3-27B-IT-Q8
  * GPT-4.1

### Manus & OpenManus Variants

* **Manus** (closed-source; invite-only) evaluated on MedAgentsBench and HLE via public web UI.
* **OpenManus** (https://github.com/mannaandpoem/OpenManus) with a layered ReAct/ToolCall architecture, plus browsing, code execution, and search.

| Configuration              | AgentClinic Text-only | AgentClinic Multimodal | MedAgentsBench | HLE Text-only | HLE Multimodal |
|----------------------------|:---------------------:|:----------------------:|:--------------:|:-------------:|:--------------:|
| **Original OpenManus**     |          ✔️           |          ✔️            |       ✔️       |      ✔️       |      ✔️        |
| **OpenManus Customized 1** |  ✔️ (explicit role)   |          n/a           |       n/a      |      n/a      |      n/a       |
| **OpenManus Customized 2** |  ✔️ (max tool usage)  |          n/a           |       n/a      |      n/a      |      n/a       |
| **vOpenManus**             |          n/a          |         ✔️             |       n/a      |      n/a      |      ✔️        |
| **Manus**                  |          n/a          |          n/a           |       ✔️      |      ✔️       |      ✔️        |

---

## Running the Benchmarks

### AgentClinic CLI

Use the following Python scripts to run different AgentClinic configurations. Adjust flags as needed:

  --doctor_llm llama4_maverick #set model used for doctor
  --patient_llm gpt-4o-mini #set model used for patient
  --measurement_llm gpt-4o-mini #set model used for measurement
  --moderator_llm gpt-4o-mini #set model used for moderator
  --num_scenarios 200 #set scenario number
  --agent_dataset MedQA #set dataset
  --doctor_image_request False #whether enable multimodality 
  --total_inferences 20 #set inference time limit
```bash
# Text-only: Original OpenManus
python agentclinicmanus_original.py --agent_dataset MedQA_Ext

# Text-only: MedAssist
python agentclinicmanus_customized1.py --doctor_llm openmanus --agent_dataset MedQA_Ext

# Text-only: MedAssist_Tool+
python agentclinicmanus_customized2.py --doctor_llm openmanus --agent_dataset MedQA_Ext
```

**Multimodal (images given at first)**

```bash
# LLM baseline
python agentclinic_vllmf.py --agent_dataset NEJM_Ext --doctor_image_request True

# Original OpenManus
python agentclinicmanus_vf_original.py --doctor_llm openmanus --agent_dataset NEJM_Ext --doctor_image_request True

# vOpenManus 
python agentclinicmanus_vf.py --doctor_llm openmanus --agent_dataset NEJM_Ext --doctor_image_request True
```

**Multimodal (images requested on demand)**

```bash
# LLM baseline
python agentclinic_vllmr.py --agent_dataset NEJM_Ext --doctor_image_request True

# Original OpenManus
python agentclinicmanus_vr_original.py --doctor_llm openmanus --agent_dataset NEJM_Ext --doctor_image_request True

# vOpenManus
python agentclinicmanus_vr.py --doctor_llm openmanus --agent_dataset NEJM_Ext --doctor_image_request True
```

### MedAgentsBench CLI

```bash
# LLM baseline
python run_llm.py \
  --model_name llama-4 \
  --dataset_name medexqa \
  --dataset_dir ../../data/medexqa/ \
  --split test_hard \
  --output_files_folder ./output/ \
  --num_processes 4 \
  --start_pos 0 \
  --end_pos -1

# OpenManus baseline
python run_openmanus.py \
  --dataset_name medexqa \
  --dataset_dir ../../data/medexqa/ \
  --split test_hard \
  --output_files_folder ./output_openmanus/ \
  --num_processes 4
```

### Humanity’s Last Exam CLI

```bash
# LLM baseline (Text-only)
python hle_llm.py \

# LLM baseline (Multimodal)
python hle_llm_img.py \

# OpenManus baseline (Text-only)
python hle.py \

# vOpenManus (Multimodal)
python hle_img.py \
```

---

## License

This project is licensed under the [MIT License](LICENSE).
