import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import evaluate
import numpy as np
import spacy
from datasets import Dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
from prodigy.components.db import connect
from prodigy.components.validate import validate
from prodigy.components.stream import get_stream
from prodigy.components.preprocess import add_tokens
from prodigy.components.decorators import support_both_streams
from prodigy.core import Arg, recipe
from prodigy.types import NerExample
from prodigy.util import log
from spacy.language import Language
from spacy.tokens import Doc
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.pipelines.token_classification import TokenClassificationPipeline
from transformers.utils.logging import set_verbosity_error as set_transformers_verbosity_error


def get_label_names(examples: List[Dict], labels: List[str]=None) -> List[str]:
    """Go through all the examples, grab all labels from spans and convert to BI-labels."""
    names = {span['label'] for ex in examples for span in ex.get('spans', [])}
    if labels:
        names = [n for n in names if n in labels]
    result = []
    for name in names:
        result.append(f"B-{name}")
        result.append(f"I-{name}")
    return ['O'] + result


def into_hf_format(train_examples: List[Dict], valid_examples: List[Dict]):
    """Turn the examples into variables/format that Huggingface expects."""
    label_names = get_label_names(train_examples)
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}

    def generator(examples) -> Iterable[Dict]:
        for ex in examples:
            tokens = [tok['text'] for tok in ex["tokens"]]
            ner_tags = [0 for _ in tokens]
            for span in ex.get("spans", []):
                ner_tags[span['token_start']] = label2id[f"B-{span['label']}"]
                for i in range(span['token_start'] + 1, span['token_end'] + 1):
                    ner_tags[i] = label2id[f"I-{span['label']}"]

            yield {
                "text": ex["text"],
                "tokens": tokens,
                "ner_tags": ner_tags,
                "ner_labels": [id2label[t] for t in ner_tags]
            }
    train_out = list(generator(train_examples))
    valid_out = list(generator(valid_examples))
    return train_out, valid_out, label_names, id2label, label2id


def validate_examples(examples: List[Dict], dataset:str) -> None:
    """Just make sure that we don't have non-NER tasks in here."""
    log(f"RECIPE: Validating examples for NER task for {dataset} dataset.")
    for ex in examples:
        validate(NerExample, ex, error_msg=f"Found an invalid example for NER: {ex} from {dataset}.")
    log("RECIPE: Validation complete.")


def produce_train_eval_datasets(datasets: str, eval_split: Optional[float] = None):
    """Handle all the eval: and --eval-split logic here."""
    db = connect()
    train_examples = []
    valid_examples = []
    for dataset in datasets.split(","):
        examples = db.get_dataset_examples(dataset.replace("eval:", ""))
        if not examples:
            raise ValueError(f"It seems dataset {dataset} has 0 examples in it.")
        validate_examples(examples, dataset)
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
    return train_examples, valid_examples

def tokenize_and_align_labels(examples, tokenizer):
    """Taken from https://huggingface.co/docs/transformers/tasks/token_classification#load-wnut-17-dataset"""
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def build_metrics_func(label_list):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification#load-wnut-17-dataset"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lab] for (p, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }

    return compute_metrics

@recipe(
    "hf.train.ner",
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
def hf_train_ner(datasets: str,
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

    train_examples, valid_examples = produce_train_eval_datasets(datasets, eval_split)
    gen_train, gen_valid, label_list, id2lab, lab2id = into_hf_format(train_examples, valid_examples)

    prodigy_dataset = DatasetDict(
        train=Dataset.from_list(gen_train),
        eval=Dataset.from_list(gen_valid)
    )

    log("RECIPE: Applying tokenizer and aligning labels.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = prodigy_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
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


class EntityMerger:
    """Helper class to make merging of B- I- tokens from hf-transformer easier."""
    def __init__(self, text, label, start, end):
        self.text = text
        self.label = label.replace("B-", "")
        self.start = start
        self.end = end
    
    def __repr__(self) -> str:
        return f"<Entity {self.label} {self.text[self.start:self.end]}>"

    def append_hf_tok(self, tok: dict) -> None:
        assert tok['entity'].startswith("I-")
        assert tok['entity'].replace("I-", "") == self.label
        self.end = tok['end']

def to_spacy_doc(text: str, hf_model: TokenClassificationPipeline, nlp: Language) -> Doc:
    """Turns the predictions that come out of a HF model into a spaCy doc."""
    # The entities that come out of hf_model are ordered, so we can loop like this
    entities = []
    current = None
    for ex in hf_model(text):
        if ex['entity'][0] == 'B':
            if current: 
                entities.append(current)
            current = EntityMerger(text=text, label=ex['entity'], start=ex['start'], end=ex['end'])
        else:
            if current is not None:
                # Theoretically the model could output an `I` without a `B` before. 
                # Just to make sure that isn't hapening at the start we add this if.
                current.append_hf_tok(ex)
    entities.append(current)

    # Now that we have the entities merged, we can char_span it to the spaCy doc
    doc = nlp(text)
    spans = []
    for ent in entities:
        span = doc.char_span(ent.start, ent.end, label=ent.label)
        if span:
            # span can be None if the found span is invalid.
            spans.append(span)
    doc.ents = spans
    return doc


def get_hf_config_labels(hf_mod: TokenClassificationPipeline):
    # Hugging Face uses a BI (sortof like  BILOU) label standard. O stands for no entity.
    keys = hf_mod.model.config.label2id.keys()
    unique = set([k.replace("B-", "").replace("I-", "") for k in keys])
    return [k for k in unique if k != "O"]


@recipe(
    "hf.ner.correct",
    # fmt: off
    dataset=Arg(help="Dataset to write annotations into"),
    model=Arg(help="Path to transformer model. Can also point to model on hub."),
    source=Arg(help="Source file to annotate"),
    lang=Arg("--lang", "-l", help="Language of the spaCy tokeniser"),
    # fmt: on
)
def hf_ner_correct(dataset: str,
                 model: str,
                 source: str,
                 lang: str = "en"):
    log("RECIPE: train.hf.ner started.")
    set_transformers_verbosity_error()
    stream = get_stream(source, rehash=True, dedup=True)
    nlp = spacy.blank(lang)
    tfm_model: TokenClassificationPipeline = pipeline("ner", model=model)
    labels = get_hf_config_labels(tfm_model)
    log(f"RECIPE: Transformer model loaded with {labels=}.")

    @support_both_streams(stream_arg="stream")
    def attach_predictions(stream):
        for ex in stream:
            doc = to_spacy_doc(ex['text'], tfm_model, nlp)
            doc_dict = doc.to_json()
            ex['spans'] = doc_dict['ents']
            yield ex

    stream.apply(attach_predictions)
    stream.apply(add_tokens, nlp=nlp, stream=stream)

    return {
        "dataset": dataset,
        "view_id": "ner_manual",
        "stream": stream,
        "config": {
            "labels": labels
        }
    }
