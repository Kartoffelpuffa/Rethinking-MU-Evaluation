import random
import torch
import numpy as np
import pandas as pd
import datetime

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from torch_influence import BaseObjective
from torch_influence import LiSSAInfluenceModule
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset, Dataset, DatasetDict

import torch.nn as nn
import argparse


class TRANS(TorchDataset):
    def __init__(self, data_file, src, tgt):
        self.data = self.load_data(data_file, src, tgt)

    def load_data(self, data_file, src, tgt):
        Data = {}
        with open(data_file + f".{src}", "rt", encoding="utf-8") as sf:
            with open(data_file + f".{tgt}", "rt", encoding="utf-8") as tf:
                for idx, (sline, tline) in enumerate(zip(sf, tf)):
                    sample = {"src": sline.strip(), "tgt": tline.strip()}
                    Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MyObjective(BaseObjective):
    def train_outputs(self, model, batch):
        return model(**batch)

    def train_loss_on_outputs(self, outputs, batch):
        return outputs.loss

    def train_regularization(self, params):
        return 0

    def test_loss(self, model, params, batch):
        return model(**batch).loss


def tokenize(examples, tokenizer):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


def preprocess_function(examples):
    inputs = [ex["src"] for ex in examples["translation"]]
    targets = [ex["tgt"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


def sep_to_dict(text):
    split = text.split("<SEP>")
    return {"src": split[0], "tgt": split[1]}


def dedup(dataset):
    df = pd.DataFrame(dataset)
    print(f"{len(dataset)} -> ", end="")
    df.translation = df.translation.apply(lambda x: x["src"] + "<SEP>" + x["tgt"])
    df = df.drop_duplicates()
    print(len(df))
    return df.translation


if __name__ == "__main__":
    seed = 0xDEADBEEF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, type=str, help="path to orginal model"
    )
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

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8

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
        depth = 5000

    elif args.dataset == "imdb":
        ds = load_dataset("stanfordnlp/" + args.dataset)
        depth = 25000

    elif args.dataset == "sst2":
        ds = load_dataset("SetFit/" + args.dataset)
        ds["train"] = ds["train"].remove_columns("label_text")
        ds["test"] = ds["test"].remove_columns("label_text")
        depth = 7000

    elif args.dataset.startswith("iwlst"):
        if args.dataset == ("iwlst"):
            depth = 160000
        elif args.dataset == ("iwlst_small"):
            depth = 16000

        path = "data/"
        train = TRANS(path + "train", "de", "en")
        valid = TRANS(path + "valid", "de", "en")
        test = TRANS(path + "test", "de", "en")

        ds = DatasetDict()
        ds["train"] = Dataset.from_dict({"translation": train})
        ds["valid"] = Dataset.from_dict({"translation": valid})
        ds["test"] = Dataset.from_dict({"translation": test})

        ds_train = dedup(ds["train"])
        ds_valid = dedup(ds["valid"])
        ds_test = dedup(ds["test"])
        ds_valid = ds_valid[~ds_valid.isin(ds_test)]
        ds_train = ds_train[~ds_train.isin(ds_valid)]
        ds_train = ds_train[~ds_train.isin(ds_test)]
        ds["train"] = Dataset.from_pandas(pd.DataFrame(ds_train.apply(sep_to_dict)))
        ds["valid"] = Dataset.from_pandas(pd.DataFrame(ds_valid.apply(sep_to_dict)))
        ds["test"] = Dataset.from_pandas(pd.DataFrame(ds_test.apply(sep_to_dict)))
        ds = ds.remove_columns("__index_level_0__")

    if not args.dataset.startswith("iwlst"):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=max(ds["train"]["label"]) + 1
        ).to(device)

        model.load_state_dict(torch.load(args.model))
        tokenized_ds = ds.map(
            tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True
        )
        tokenized_ds["train"] = tokenized_ds["train"].remove_columns("text")
        tokenized_ds["test"] = tokenized_ds["test"].remove_columns("text")
        data_collator = DataCollatorWithPadding(tokenizer)

    else:
        batch_size = 4
        model_checkpoint = args.model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        max_length = 128

        if args.dataset == "iwlst_small":
            seed = 0xDEADBEEF
            random.seed(seed)
            small = [i for i in range(len(ds["train"])) if random.uniform(0, 1) <= 0.1]
            ds["train"] = ds["train"].select(small)

        tokenized_ds = ds.map(
            preprocess_function,
            batched=True,
            remove_columns=ds["train"].column_names,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    if args.simple:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # remove new set
    new_list = []
    with open(f"data/new.txt", "r") as f:
        for line in f:
            new_list.append(int(line.strip()))

    tokenized_ds["train"] = tokenized_ds["train"].select(
        [i for i in range(len(tokenized_ds["train"])) if i not in new_list]
    )

    print(len(tokenized_ds["train"]), len(tokenized_ds["train"]))

    train_dataloader = DataLoader(
        tokenized_ds["train"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=batch_size,
    )

    test_dataloader = DataLoader(
        tokenized_ds["test"], collate_fn=data_collator, batch_size=1
    )

    def callback(r, t, h):
        print(r, t)
        print(h)

    module = LiSSAInfluenceModule(
        model=model,
        objective=MyObjective(),
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        device=device,
        damp=3e-3,
        repeat=1,
        depth=depth,
        scale=1e4,
        debug_callback=callback,
    )

    start_time = datetime.datetime.now()

    try:
        module.stest([0])
    except RuntimeError:
        total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
        print(f"After {total_minutes} we can start :)")

    scores = module.influences(
        range(len(tokenized_ds["train"])), range(len(tokenized_ds["test"]))
    )

    total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0

    print(
        "Computed influence over a dataset of %d examples in %.2f minutes"
        % (len(tokenized_ds["train"]), total_minutes)
    )

    torch.save(scores, "scores.pt")
