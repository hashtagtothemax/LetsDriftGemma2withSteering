# Conda Env == lets_drift_env_V1 
#(Which is a clone of arena_env1_1_1c)
# Imports for the Gemma Model
try:
    import torch as t
    device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
    from huggingface_hub import login
    login()  # This will prompt for your HuggingFace token in a text box
    print("starts with a j..jeje .. f_JTQGGDeycTxPCozthrzqWQhmLpOkaVKXRq")

    # SAE lens and TransformerLens
    from sae_lens import (
        SAE,
        ActivationsStore,
        HookedSAETransformer,
        LanguageModelSAERunnerConfig,
        SAEConfig,
        SAETrainingRunner,
        upload_saes_to_huggingface,
    )
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    from transformer_lens import ActivationCache, HookedTransformer, utils
    from transformer_lens.hook_points import HookPoint
    # from transformer_lens import HookedTransformer
    from sae_lens import SAE, HookedSAETransformer
    import pandas as pd


    # for steering impln and evaluation metrics
    from rich import print as rprint
    from rich.table import Table
    from tqdm.auto import tqdm
    from functools import partial
    from jaxtyping import Float
    from torch import Tensor

    import torch.nn.functional as F

    # Plot line chart of latent activations
    import plotly.express as px

    # instantiate an object to hold activations from a dataset for finding max activation of latent
    from sae_lens import ActivationsStore

    # For display of neuronpedia dashboards
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

    # For evaluation metrics
    import json
    from pathlib import Path

    print("Successfully imported all the imports")
except Exception as e:
    print(f"Error occurred during imports: {str(e)}")