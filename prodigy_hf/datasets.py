from pathlib import Path 
from typing import Optional

from datasets import Dataset, DatasetDict

from prodigy.core import recipe, Arg
from prodigy.util import ANNOTATOR_ID_ATTR, SESSION_ID_ATTR
from prodigy.components.db import connect 
from prodigy.util import log

def collect_datasets(datasets_str):
    """Turn the datasets string into examples."""
    db = connect()
    datasets = datasets_str.split(",")
    keyed_examples = {"train": []}
    for dataset in datasets:
        if ":" not in dataset:
            kind = "train"
            name = dataset
        else:
            kind, name = dataset.split(":")
            if kind not in keyed_examples:
                keyed_examples[kind] = []
        keyed_examples[kind].extend(db.get_dataset_examples(name))
    
    n_examples = 0 
    for kind, examples in keyed_examples.items():
        log(f"RECIPE: Collected {len(examples)} examples for {kind=}")
        n_examples += len(examples)
    log(f"RECIPE: Collected {n_examples} examples in total.")
    return keyed_examples


def validate(keyed_examples):
    """This is very basic validation that ensures that all examples have the same structure."""
    example_keys = keyed_examples["train"][0].keys()
    for kind, examples in keyed_examples.items():
        for ex in examples:
            assert ex.keys() == example_keys, f"Found examples with different keys. May be incompatible data. {ex.keys()} != {example_keys}"


def replace_annotator(examples):
    """This is very basic validation that ensures that all examples have the same structure."""
    mapper = {}
    for ex in examples:
        if ANNOTATOR_ID_ATTR in ex:
            annot_id = ex[ANNOTATOR_ID_ATTR]
            if annot_id not in mapper:
                mapper[annot_id] = len(mapper) + 1
            ex[ANNOTATOR_ID_ATTR] = f"annotator-{mapper[annot_id]}"
            ex[SESSION_ID_ATTR] = f"session-{mapper[annot_id]}"
        yield ex


@recipe(
    "upload.hf",
    # fmt: off
    datasets=Arg(help="Datasets with NER annotations to train model for"),
    repo_id=Arg(help="Name of upstream dataset to upload data into"),
    keep_annotator_ids=Arg("--keep-annotator-ids", "-k", help="Don't anonymise annotator ids."),
    no_validation=Arg("--no-validation", "-nv", help="Don't validate the datasets."),
    private=Arg("--private", "-p", help="Keep this dataset private."),
    # fmt: on
)
def upload_hf(datasets: str, repo_id:str, keep_annotator_ids: bool=False, no_validation: bool=False, private:bool = False):
    """Uploads annotated datasets from Prodigy to Huggingface."""
    # This recipe assumes that user ran `huggingface-cli login` beforehand.
    # https://huggingface.co/spaces/huggingface/datasets-tagging
    log(f"RECIPE: About to collect {datasets=}.")
    keyed_examples = collect_datasets(datasets)
    if not no_validation:
        validate(keyed_examples)
    if not keep_annotator_ids:
        for kind, examples in keyed_examples.items():
            keyed_examples[kind] = list(replace_annotator(examples))
    dataset_dict = DatasetDict({kind: Dataset.from_list(examples) for kind, examples in keyed_examples.items()})
    dataset_dict.push_to_hub(repo_id)
