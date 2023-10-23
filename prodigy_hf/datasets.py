from pathlib import Path 
from typing import Optional

import datasets as hf_datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard, DatasetCardData, create_repo
from huggingface_hub.utils import HfHubHTTPError

from prodigy.core import recipe, Arg
from prodigy.util import ANNOTATOR_ID_ATTR, SESSION_ID_ATTR
from prodigy.components.db import connect 
from prodigy.util import log, msg

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
    log("RECIPE: Replacing annotator references.")
    for ex in examples:
        if ANNOTATOR_ID_ATTR in ex:
            annot_id = ex[ANNOTATOR_ID_ATTR]
            if annot_id not in mapper:
                mapper[annot_id] = len(mapper) + 1
            ex[ANNOTATOR_ID_ATTR] = f"annotator-{mapper[annot_id]}"
            ex[SESSION_ID_ATTR] = f"session-{mapper[annot_id]}"
        yield ex


def init_repo(repo_id: str) -> None:
    """If the repo does not exist, add a card with Prodigy tag."""
    try:
        repo_id = create_repo(repo_id, repo_type="dataset").repo_id
        card = DatasetCard.from_template(card_data=DatasetCardData(tags=["prodigy"]))
        card.push_to_hub(repo_id=repo_id)
        log(f"RECIPE: Creating a new repo over at {repo_id}. Automatically adding `prodigy` tag to new card.")
    except HfHubHTTPError:
        log("RECIPE: Repo already exists. Won't create card.")


@recipe(
    "hf.upload",
    # fmt: off
    datasets=Arg(help="Datasets with NER annotations to train model for"),
    repo_id=Arg(help="Name of upstream dataset to upload data into"),
    keep_annotator_ids=Arg("--keep-annotator-ids", "-k", help="Don't anonymise annotator ids."),
    no_validation=Arg("--no-validation", "-nv", help="Don't validate the datasets."),
    private=Arg("--private", "-p", help="Keep this dataset private."),
    # fmt: on
)
def hf_upload(datasets: str, repo_id:str, keep_annotator_ids: bool=False, no_validation: bool=False, private:bool = False):
    """Uploads annotated datasets from Prodigy to Huggingface."""
    hf_datasets.utils.logging.set_verbosity_error()
    # This recipe assumes that user ran `huggingface-cli login` beforehand.
    # https://huggingface.co/spaces/huggingface/datasets-tagging
    log(f"RECIPE: About to collect {datasets=}.")
    keyed_examples = collect_datasets(datasets)
    if not no_validation:
        validate(keyed_examples)
    if not keep_annotator_ids:
        for kind, examples in keyed_examples.items():
            keyed_examples[kind] = list(replace_annotator(examples))
    init_repo(repo_id)
    dataset_dict = DatasetDict({kind: Dataset.from_list(examples) for kind, examples in keyed_examples.items()})
    dataset_dict.push_to_hub(repo_id)
    msg.good(f"Upload completed! You should be able to view repo at https://huggingface.co/datasets/{repo_id}.",)
    msg.info("If you haven't done so already: don't forget update the datacard!")
    log(f"RECIPE: Updated {repo_id=}.")
