import numpy as np
import torch
import argparse
import random

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, load_metric


def tokenize(examples, tokenizer):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


def compute_metrics_acc(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_f1(eval_preds):
    metric = load_metric("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


if __name__ == "__main__":
    seed = 0xDEADBEEF
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, type=str, help="name of dataset [imdb, trec, sst2]"
    )
    parser.add_argument(
        "--simple",
        required=False,
        default=False,
        action="store_true",
        help="set to true for frozen weights",
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="number of training epochs"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16

    if args.dataset == "trec":
        ds = load_dataset("SetFit/TREC-QC")
        ds = ds.map(
            remove_columns=[
                "label",
                "label_text",
                "label_original",
                "label_coarse_text",
                "label_coarse_original",
            ]
        )
        ds["train"] = ds["train"].rename_column("label_coarse", "label")
        ds["test"] = ds["test"].rename_column("label_coarse", "label")
        metrics = compute_metrics_acc

    elif args.dataset == "imdb":
        ds = load_dataset("stanfordnlp/" + args.dataset)
        metrics = compute_metrics_acc

    elif args.dataset == "sst2":
        ds = load_dataset("SetFit/" + args.dataset)
        ds["train"] = ds["train"].remove_columns("label_text")
        ds["test"] = ds["test"].remove_columns("label_text")
        metrics = compute_metrics_acc

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_ds = ds.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=max(ds["train"]["label"]) + 1
    ).to(device)
    if args.simple:
        print("Simple")
        for param in model.base_model.parameters():
            param.requires_grad = False

    # remove new set
    new_list = []
    with open("data/new.txt", "r") as f:
        for line in f:
            new_list.append(int(line.strip()))

    train_data = tokenized_ds["train"].select(
        [i for i in range(len(tokenized_ds["train"])) if i not in new_list]
    )

    if args.simple:
        simple = "_s"
        lr = 3e-4
    else:
        simple = ""
        lr = 5e-5

    training_args = TrainingArguments(
        output_dir="./",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        save_safetensors=False,
        learning_rate=lr,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=tokenized_ds["test"],
        compute_metrics=metrics,
    )

    trainer.train()
    torch.save(model.state_dict(), f"baseline{simple}.pt")
