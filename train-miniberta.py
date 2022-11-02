import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
logging.basicConfig(level=logging.ERROR)
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import wandb
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import random

wandb.init(project="cds", entity="adityay")

def load_sentences_from_file(file_path: Path,
                             include_punctuation: bool = True,
                             allow_discard: bool = False,
                             ) -> List[str]:
    """
    load sentences for language modeling from text file
    """

    print(f'Loading {file_path}', flush=True)

    res = []
    num_too_small = 0
    
    with open(file_path, 'r') as line_by_line_file:
    # with file_path.open('r') as line_by_line_file:

        for sentence in line_by_line_file.readlines():

            if not sentence:  # during probing, parsing logic above may produce empty sentences
                continue

            sentence = sentence.rstrip('\n')

            # check  length
            if sentence.count(' ') < 3 - 1 and allow_discard:
                num_too_small += 1
                continue

            if not include_punctuation:
                sentence = sentence.rstrip('.')
                sentence = sentence.rstrip('!')
                sentence = sentence.rstrip('?')

            res.append(sentence)

    if num_too_small:
        print(f'WARNING: Skipped {num_too_small:,} sentences which are shorter than {3}.')

    return res

from itertools import islice

def make_sequences(sentences: List[str],
                   num_sentences_per_input: int,
                   ) -> List[str]:

    gen = (bs for bs in sentences)

    # combine multiple sentences into 1 sequence
    res = []
    while True:
        sentences_in_sequence: List[str] = list(islice(gen, 0, num_sentences_per_input))
        if not sentences_in_sequence:
            break
        sequence = ' '.join(sentences_in_sequence)
        res.append(sequence)

    print(f'Num total sequences={len(res):,}', flush=True)
    return res

def get_perplexity(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input.cuda(), labels=labels.cuda()).loss
    return np.exp(loss.item())

from datasets import Dataset, DatasetDict

from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed, TrainingArguments

def get_scores_on_paradigm(model, tokenizer, file_path):
    with open(file_path) as f:
        data = list(f)
    
    acc = 0
    for item in data:
        line = json.loads(item)
        good = line["sentence_good"]
        bad = line["sentence_bad"]
        good_score = get_perplexity(sentence=good, model=model, tokenizer=tokenizer)
        bad_score = get_perplexity(sentence=bad, model=model, tokenizer=tokenizer)
        if bad_score >= good_score:
            acc += 1
    
    acc = acc / len(data)
    return acc

def main():

    rep = 0
    path_out = '/scratch/pbsjobs/axy327/' + str(rep)

    print(f'replication={rep}')

    training_args = TrainingArguments(
        report_to=None,
        output_dir=str(path_out),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        max_steps=160_000,
        warmup_steps=24_000,
        seed=rep,
        save_steps=40_000
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    set_seed(rep)

    logger.info("Loading data")
    data_path = 'text_spok.txt'  # we use aonewsela for reference implementation
    sentences = load_sentences_from_file(data_path,
                                         include_punctuation=True,
                                         allow_discard=True)
    data_in_dict = {'text': make_sequences(sentences, 1)}
    datasets = DatasetDict({'train': Dataset.from_dict(data_in_dict)})
    print(datasets['train'])
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    logger.info("Loading tokenizer")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files="text_spok.txt", vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    ])
    tokenizer.save_model("Babyberta")
    tokenizer.save("byte-level-BPE.tokenizer.json")
    tokenizer = RobertaTokenizerFast(vocab_file=None,
                                     merges_file=None,
                                     tokenizer_file=str('byte-level-BPE.tokenizer.json')
    )
    logger.info("Initialising Roberta from scratch")
    config = RobertaConfig(vocab_size=52_000,
                           hidden_size=256,
                           num_hidden_layers=8,
                           num_attention_heads=8,
                           intermediate_size=1024,
                           initializer_range=0.02,
                           )
    model = RobertaForMaskedLM(config)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    text_column_name = "text"

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=128,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
    logger.info("Tokenising data")
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[text_column_name],
        load_from_cache_file=True,
    )
    
    train_dataset = tokenized_datasets["train"]
    print(f'Length of train data={len(train_dataset)}')

    # Data collator will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm_probability=0.15)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Training
    trainer.train()
    trainer.save_model()  # Saves the tokenizer too

    print(get_perplexity(sentence='London is the capital of Great Britain.', model=model, tokenizer=tokenizer))
    print(get_perplexity(sentence='London is the capital of South America.', model=model, tokenizer=tokenizer))
    # path = "tests/wh_vs_that_with_gap_long_distance.jsonl"
    paths = glob.glob("tests/*.jsonl")
    for path in paths:
        acc = get_scores_on_paradigm(model, tokenizer, path)
        print(path + " " + str(acc*100))

if __name__ == "__main__":
    main()
