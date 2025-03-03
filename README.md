# Tools for applying SAE-based latent space steering to Gemma2 family models and monitoring output logits and latents using hookpoints

### Purpose:

To induce model drift, where the model 'unexpectedly' looses performance over time, SAE-based latent space steering is utilized.

The tools of this repository allow monitoring of different hook points, using both the raw logits and SAE interpretations provided by GemmaScope, to assess downstream effect within the models, when such steering based model drift occurs.

Hypothesis: there may be ways of predicting when a model drifts (output text changes from correct to incorrect), without actually looking at the output text. 

### How to use:

conda_requirements.txt can be used to install all necessary conda and pip libs with the correct versions using conda venv. 
Note that the PyTorch Version may need to be adapted to your version of CUDA. 

The following order of tools can be used to evaluate the publicly available USMLE and HotPotQA Datasets (see https://github.com/lchen001/LLMDrift for these)

1. LetsDriftGemma2_DatasetSteered_v2_looped.ipynb:
    This notebook allows prompting of a chosen LLM with a loop of prompts from a dataset, to return one unsteered answer and then a parametrizable amount of steered answers. The metadata and answers are automatically written into .jsonl file format.
    Previous results are available at request. Note that 30 prompts can generate up to 100GB of data due to the amount of logits recorded by the hookpoints in the current configuration.
   
    Instructions for configurations are given inside the file using headings and descriptive comments.

2. EVAL_steered_jsonl_v2.ipynb:
    Part of the evaluation toolchain, this notebook will open a dialog, allowing a human evaluator to judge the LLM's answers by entering 1 for a displayed correct or 0 for a displayed incorrect answer. Reference answers are also provided by this tool.

3. EVAL_steered_latents_v2.ipynb:
    Part of the evaluation toolchain, this notebooks shows extensive stastical comparisons performed on the logits and latents stored in the output .jsonl files. 
