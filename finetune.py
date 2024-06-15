import numpy as np
import pandas as pd
import torch
import argparse
import random

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import evaluate


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


def tokenize(examples, tokenizer):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
    )
    return outputs


def compute_metrics_acc(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_bleu(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


if __name__ == "__main__":
    print("Welcome!")

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
    parser.add_argument(
        "--sample_ratio",
        required=False,
        type=float,
        help="subsample part of training data",
    )
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

    sizes = args.sizes or [1, 2, 5, 10, 15, 20]

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

    elif args.dataset.startswith("iwlst"):
        if args.dataset == ("iwlst"):
            depth = 160000
        elif args.dataset == ("iwlst_small"):
            depth = 16000

        path = ""
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

    if args.dataset.startswith("iwlst"):
        model_checkpoint = "checkpoint-1491"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        max_length = 128

        if args.dataset == "iwlst_small":
            random.seed(seed)
            small = [i for i in range(len(ds["train"])) if random.uniform(0, 1) <= 0.1]
            ds["train"] = ds["train"].select(small)

        tokenized_ds = ds.map(
            preprocess_function,
            batched=True,
            remove_columns=ds["train"].column_names,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    else:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        print("Tokenizing...")
        tokenized_ds = ds.map(
            tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True
        )
        print("Tokenizing done!")

    if args.influence:
        influence = "i_"
    else:
        influence = ""

    if args.simple:
        simple = "_s"
        lr = 3e-4
    else:
        simple = ""
        lr = 5e-5

    for mode in ["forget", "new", "retrain"]:
        for size in sizes:
            if args.dataset.startswith("iwlst"):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    "Helsinki-NLP/opus-mt-de-en"
                ).to(device)
            else:
                model = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=max(ds["train"]["label"]) + 1,
                ).to(device)
            if args.simple:
                print("Simple")
                for param in model.base_model.parameters():
                    param.requires_grad = False

            if mode in ["forget", "new"]:
                forget_list = []

                with open(f"forget_{influence}{size}{simple}_{seed}.txt", "r") as f:
                    for line in f:
                        forget_list.append(int(line.strip()))

                new_list = []
                with open("new.txt", "r") as f:
                    for line in f:
                        new_list.append(int(line.strip()))

                if mode == "forget":
                    train_list = forget_list
                    exclude_list = new_list
                elif mode == "new":
                    train_list = new_list
                    exclude_list = forget_list

                assert 0 <= args.sample_ratio <= 1
                select_list = [
                    idx
                    for idx in range(len(tokenized_ds["train"]))
                    if random.random() <= args.sample_ratio
                ]

                include_indices = train_list + select_list

                train_data = tokenized_ds["train"].select(
                    [i for i in include_indices if i not in exclude_list]
                )

                print(mode, influence, size, simple)

            else:
                # remove new set
                new_list = []
                with open("new.txt", "r") as f:
                    for line in f:
                        new_list.append(int(line.strip()))

                remove_list = new_list

                if mode == "retrain":
                    with open(f"forget_{influence}{size}{simple}_{seed}.txt", "r") as f:
                        for line in f:
                            remove_list.append(int(line.strip()))

                    train_data = tokenized_ds["train"].select(
                        [
                            i
                            for i in range(len(tokenized_ds["train"]))
                            if i not in remove_list
                        ]
                    )

                    train_data = train_data.shuffle(seed=seed)

                print(mode, influence, size, simple)

            if mode == "retrain":
                eval_strat = "epoch"
            else:
                eval_strat = "no"

            if not args.dataset.startswith("iwlst"):
                training_args = TrainingArguments(
                    output_dir="./",
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    num_train_epochs=args.epochs,
                    evaluation_strategy=eval_strat,
                    save_strategy="no",
                    save_safetensors=False,
                    learning_rate=lr,
                )

                data_collator = DataCollatorWithPadding(tokenizer)
                print(len(train_data))

                trainer = Trainer(
                    model=model,
                    data_collator=data_collator,
                    args=training_args,
                    train_dataset=train_data,
                    eval_dataset=tokenized_ds["test"],
                    compute_metrics=metrics,
                )
            else:
                metric = evaluate.load("sacrebleu")
                training_args = Seq2SeqTrainingArguments(
                    output_dir="./",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    weight_decay=0.01,
                    save_total_limit=3,
                    num_train_epochs=3,
                    predict_with_generate=True,
                    fp16=True,
                )

                data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
                print(len(train_data))

                trainer = Seq2SeqTrainer(
                    model,
                    training_args,
                    train_dataset=tokenized_ds["train"],
                    eval_dataset=tokenized_ds["test"],
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics_bleu,
                )

            trainer.train()
            torch.save(
                model.state_dict(), f"{mode}_{influence}{size}{simple}_{seed}.pt"
            )
