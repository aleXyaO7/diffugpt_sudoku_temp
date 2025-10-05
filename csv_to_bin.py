# python csv_to_bin.py --model-name-or-path gpt2-model-bs1024-lr1e-3-ep100-20250910-035030 --dataset sudoku_test

import pandas as pd
import numpy as np

from packages.packed_dataset import PackedDatasetBuilder
from packages.llmtuner.tuner.core import get_train_args
from packages.llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from packages.custom_tokenizer import CustomTokenizer

model_directory = 'gpt2-model-bs1024-lr1e-3-ep100-20250910-035030'
data_directory = 'data/test'
dataset_name = 'sudoku_test'
chunk_size = 16400
sep_token = 1
vocab_size = 31

model_args, diffusion_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
dataset = get_dataset(model_args, data_args)
tokenizer = CustomTokenizer.from_pretrained(model_directory)
dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")
total_dataset = np.array(dataset['input_ids']).flatten()

dataset_builder = PackedDatasetBuilder(data_directory, dataset_name, chunk_size, sep_token, vocab_size=vocab_size)
dataset_builder.add_adjusted_array(total_dataset)
