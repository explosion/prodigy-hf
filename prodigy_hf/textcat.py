import time
import random
from typing import List, Dict, Iterable, Optional, Literal
from pathlib import Path

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.utils.logging import set_verbosity_error as set_transformers_verbosity_error

from prodigy.components.db import connect
from prodigy.core import recipe, Arg
from prodigy.util import log


def get_label_names(examples: List[Dict], variant: Literal["binary", "multi"]) -> List[str]:
    """We have to assume exclusive textcat here. So the first example contains all labels."""
    if variant == "multi":
        return [ex['id'] for ex in examples[0]['options']]
    return ["accept", "reject"]


def into_hf_format(train_examples: List[Dict], valid_examples: List[Dict], variant: Literal["binary", "multi"]):
    """Turn the examples into variables/format that Huggingface expects."""
    label_names = get_label_names(train_examples, variant)
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}

    def generator(examples) -> Iterable[Dict]:
        for ex in examples:
            yield = {
                "text": ex["text"],
                "label": label2id[ex["answer"]] if variant =="binary" else label2id[ex["accept"][0]]
            }

    train_out = list(generator(train_examples))
    valid_out = list(generator(valid_examples))
    return train_out, valid_out, label_names, id2label, label2id


def filter_examples(examples: List[Dict], variant: Literal["binary", "multi"]):
    for ex in examples:
        if (ex['answer'] != 'ignore'): 
            yield ex
 
def validate_examples(examples: List[Dict], dataset:str, variant: Literal["binary", "multi"]) -> None:
    """Just make sure that we don't have non-NER tasks in here."""
    log(f"RECIPE: Validating examples for textcat task for {dataset} dataset.")
    label_names = get_label_names(examples, variant=variant)
    for ex in examples:
        if variant == "multi":
            options = [opt['id'] for opt in ex['options']]
            assert set(options) == set(label_names),  f"Found an example that has different labels {ex} than expected {label_names}."
        else:
            assert ex["answer"] in ["accept", "reject"], f"Found an example {ex} that doesn't have binary answer option."
    log("RECIPE: Validation complete.")


def produce_train_eval_datasets(datasets: str, eval_split: Optional[float] = None):
    """Handle all the eval: and --eval-split logic here."""
    db = connect()
    train_examples = []
    valid_examples = []
    variant = None
    for dataset in datasets.split(","):
        examples = db.get_dataset_examples(dataset.replace("eval:", ""))
        if variant is None:
            variant = "multi" if 'options' in examples[0] else "binary"
            log(f"RECIPE: Assuming {variant=}.")
        examples = list(filter_examples(examples, variant=variant))
        validate_examples(examples, dataset, variant=variant)
        if "eval:" in dataset:
            valid_examples.extend(examples)
        else:
            train_examples.extend(examples)
    if len(valid_examples) == 0:
        log("RECIPE: No eval set specified via `eval:<name>`.")
        if eval_split:
            log(f"RECIPE: Using--eval-split={eval_split}.")
            random.seed(42)
            random.shuffle(train_examples)
            cutoff = int(len(train_examples) * eval_split)
            train_examples, valid_examples = train_examples[cutoff:], train_examples[:cutoff]
        else:
            log("RECIPE: No --eval-split specified. Will evaluate on train set.")
            valid_examples = train_examples
    log(f"RECIPE: Created train/valid split. #train={len(train_examples)} #valid={len(valid_examples)}")
    return train_examples, valid_examples, variant


def build_metrics_func(label_list):
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Taken from https://huggingface.co/docs/transformers/tasks/sequence_classification#evaluate"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return compute_metrics

@recipe(
    "hf.train.textcat",
    # fmt: off
    datasets=Arg(help="Datasets with NER annotations to train model for"),
    out_dir=Arg(help="Folder to save trained model into"),
    epochs=Arg("--epochs", "-e", help="Number of epochs to finetune"),
    model_name=Arg("--model-name", "-m", help="HFmodel to use as a base model"),
    batch_size=Arg("--batch-size", "-bs", help="Batch size."),
    eval_split=Arg("--eval-split", "-es", help="If no evaluation sets are provided for a component, split off a a percentage of the training examples for evaluation."),
    learning_rate=Arg("--learning-rate", "-lr", help="Learning rate."),
    verbose=Arg("--verbose", "-v", help="Output all the logs/warnings from Huggingface libraries."),
    # fmt: on
)
def hf_train_textcat(datasets: str,
                     out_dir: Path,
                     epochs: int = 10,
                     model_name: str = "distilbert-base-uncased",
                     batch_size: int = 8,
                     eval_split: Optional[float] = None,
                     learning_rate: float = 2e-5,
                     verbose:bool = False):
    log("RECIPE: train.hf.ner started.")
    if not verbose:
        set_transformers_verbosity_error()
        disable_progress_bar()

    train_examples, valid_examples, variant = produce_train_eval_datasets(datasets, eval_split)
    gen_train, gen_valid, label_list, id2lab, lab2id = into_hf_format(train_examples, valid_examples, variant)

    prodigy_dataset = DatasetDict(
        train=Dataset.from_list(gen_train),
        eval=Dataset.from_list(gen_valid)
    )

    log("RECIPE: Applying tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_dataset = prodigy_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(id2lab), id2label=id2lab, label2id=lab2id
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_metrics_func(label_list),
    )
    log("RECIPE: Starting training.")
    tic = time.time()
    trainer.train()
    toc = time.time()
    log(f"RECIPE: Total training time: {round(toc - tic)}s.")
