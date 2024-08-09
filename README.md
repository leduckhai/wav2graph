# EntityKG (Entity Knowledge Graph)

![EntityKG](EntityKG.png)

This repository contains the codebase for the **wav2graph** paper:

**wav2graph**: A Framework for Supervised Learning Knowledge Graph from Speech

https://arxiv.org/pdf/2408.04174.pdf


## Project Overview

The **wav2graph** paper presents a novel approach to constructing entity knowledge graphs from speech data. This repository provides the necessary scripts, configurations, and setup instructions to reproduce the experiments discussed in the paper.

## Setup

To set up the environment and run the experiments, follow the steps below:

### 1. Create a Virtual Environment

Before you start, create a Python virtual environment and install the required dependencies.

```bash
pip install -r requirements.txt
```

### 2. Configure Hugging Face Token

You will need a Hugging Face API token to access certain resources used in this project. Insert your Hugging Face token into the relevant YAML configuration files.

### 3. Run the Experiments

Once the environment is set up and the configurations are complete, you can run the experiments using the provided script.

```bash
sh run.sh
```

## Cite our work

```bibtex
@misc{leduc2024wav2graphframeworksupervisedlearning,
      title={wav2graph: A Framework for Supervised Learning Knowledge Graph from Speech}, 
      author={Khai Le-Duc and Quy-Anh Dang and Tan-Hanh Pham and Truong-Son Hy},
      year={2024},
      eprint={2408.04174},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04174}, 
}
```
