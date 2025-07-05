import json
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from rouge_score import rouge_scorer
import argparse
import warnings
from phase1_prompting_experiment import PromptingExperiment
warnings.filterwarnings('ignore')