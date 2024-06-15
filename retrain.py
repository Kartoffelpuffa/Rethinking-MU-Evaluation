import numpy as np
import torch
import argparse
import random
import time
import os
import copy

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch.nn.functional as F
from datasets import load_dataset, load_metric
from evaluate import evaluator, load


def tokenize(examples, tokenizer):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


if __name__ == "__main__":
    print("welcome!")

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
        "--influence",
        required=False,
        default=False,
        action="store_true",
        help="set to true for influential forget data",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="the batch size")
    parser.add_argument(
        "--epochs", default=1, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="0xDEADBEEF",
        help="seed for random number generation, default 0xDEADBEEF",
    )
    parser.add_argument(
        "--sizes", required=False, nargs="+", help="list of forget set sizes in %"
    )

    args = parser.parse_args()

    seed = int(args.seed, 0)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        metric = load("accuracy")
        label_mapping = {
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2,
            "LABEL_3": 3,
            "LABEL_4": 4,
            "LABEL_5": 5,
        }

    elif args.dataset == "imdb":
        ds = load_dataset("stanfordnlp/" + args.dataset)
        metric = load("accuracy")
        label_mapping = {"LABEL_0": 0, "LABEL_1": 1}

    elif args.dataset == "sst2":
        ds = load_dataset("SetFit/" + args.dataset)
        metric = load("accuracy")
        label_mapping = {"LABEL_0": 0, "LABEL_1": 1}
        ds["train"] = ds["train"].remove_columns("label_text")
        ds["test"] = ds["test"].remove_columns("label_text")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print("Tokenizing...")
    tokenized_ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    print("Tokenizing done!")
    data_collator = DataCollatorWithPadding(tokenizer)

    if args.simple:
        simple = "_s"
    else:
        simple = ""

    if args.influence:
        influence = "i_"
    else:
        influence = ""

    if args.simple:
        original = "baseline_s.pt"
    else:
        original = "baseline.pt"

    original_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=max(ds["train"]["label"]) + 1
    ).to(device)

    original_model.load_state_dict(torch.load(original))
    original_model.eval()

    for size in args.sizes:
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=max(ds["train"]["label"]) + 1
        ).to(device)

        if seed == 0xDEADBEEF and args.dataset != "trec":
            model.load_state_dict(torch.load(f"retrain_{influence}{size}{simple}.pt"))
        else:
            model.load_state_dict(
                torch.load(f"retrain_{influence}{size}{simple}_{seed}.pt")
            )
        model.eval()

        scrub_list = []

        with open(f"forget_{influence}{size}{simple}_{seed}.txt", "r") as f:
            for line in f:
                scrub_list.append(int(line.strip()))

        tokenized_ds["scrub"] = tokenized_ds["train"].select(scrub_list)

        print(len(tokenized_ds["scrub"]))

        task_evaluator = evaluator("text-classification")

        eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            data=tokenized_ds["scrub"],
            tokenizer=tokenizer,
            metric=metric,
            label_mapping=label_mapping,
        )

        print("Retrain Forget", eval_results)

        task_evaluator = evaluator("text-classification")

        eval_results = task_evaluator.compute(
            model_or_pipeline=original_model,
            data=tokenized_ds["scrub"],
            tokenizer=tokenizer,
            metric=metric,
            label_mapping=label_mapping,
        )

        print("Original Forget", eval_results)

        torch.cuda.empty_cache()
