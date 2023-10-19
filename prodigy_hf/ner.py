import numpy as np
import evaluate

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from prodigy.components.db import connect

def get_label_names(examples):
    names = {span['label'] for ex in examples for span in ex.get('spans', [])}
    result = []
    for name in names:
        result.append(f"B-{name}")
        result.append(f"I-{name}")
    return ['O'] + result 
    
def into_hf_format(examples):
    label_names = get_label_names(examples)
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}
    def generator():
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
    return generator, id2label, label2id

def tokenize_and_align_labels(examples):
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



if __name__ == "__main__":
    db = connect()
    model_name = "distilbert-base-uncased"
    examples = db.get_dataset_examples("hf-demo")
    generator, id2lab, lab2id = into_hf_format(examples)
    prodigy_dataset = DatasetDict(
        train=Dataset.from_generator(generator), 
        eval=Dataset.from_generator(generator)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = prodigy_dataset.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(id2lab), id2label=id2lab, label2id=lab2id
    )
    
    seqeval = evaluate.load("seqeval")

    label_list = get_label_names(examples)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
    }

    training_args = TrainingArguments(
        output_dir="my_prodigy_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
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
        compute_metrics=compute_metrics,
    )

    trainer.train()