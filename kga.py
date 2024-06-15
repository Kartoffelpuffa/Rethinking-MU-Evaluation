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
    parser.add_argument("--batch_size", type=int, default=16, help="the batch size")
    parser.add_argument("--name", type=str, help="name of output model")
    parser.add_argument(
        "--stop",
        default=True,
        action="store_true",
        help="whether to use KGA stopping. keep True",
    )

    args = parser.parse_args()

    seed = int(args.seed, 0)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "trec":
        ds = load_dataset(args.dataset)
        ds["train"] = ds["train"].rename_column("coarse_label", "label")
        ds["test"] = ds["test"].rename_column("coarse_label", "label")
        ds["train"] = ds["train"].remove_columns("fine_label")
        ds["test"] = ds["test"].remove_columns("fine_label")
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
        ds = load_dataset(args.dataset)
        metric = load("accuracy")
        label_mapping = {"LABEL_0": 0, "LABEL_1": 1}

    elif args.dataset == "sst2":
        ds = load_dataset("SetFit/" + args.dataset)
        metric = load("accuracy")
        label_mapping = {"LABEL_0": 0, "LABEL_1": 1}
        ds["train"] = ds["train"].remove_columns("label_text")
        ds["test"] = ds["test"].remove_columns("label_text")

    if args.simple:
        original = "baseline_s.pt"
    else:
        original = "baseline.pt"

    original_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=max(ds["train"]["label"]) + 1,
    ).to(device)
    original_model.load_state_dict(torch.load(original))
    original_model.eval()

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

    for size in args.sizes:
        forget_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=max(ds["train"]["label"]) + 1,
        ).to(device)
        forget_model.load_state_dict(
            torch.load(f"forget_{influence}{size}{simple}_{seed}.pt")
        )
        forget_model.eval()

        new_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=max(ds["train"]["label"]) + 1,
        ).to(device)
        new_model.load_state_dict(
            torch.load(f"new_{influence}{size}{simple}_{seed}.pt")
        )
        new_model.eval()

        model = copy.deepcopy(original_model)

        for p1, p2, p3 in zip(
            new_model.parameters(),
            forget_model.parameters(),
            original_model.parameters(),
        ):
            p1.requires_grad = False
            p2.requires_grad = False
            p3.requires_grad = False

        for param in model.parameters():
            param.requires_grad = True

        if args.simple:
            for param in model.base_model.parameters():
                param.requires_grad = False

        scrub_list = []

        with open(f"forget_{influence}{size}{simple}_{seed}.txt", "r") as f:
            for line in f:
                scrub_list.append(int(line.strip()))

        tokenized_ds["scrub"] = tokenized_ds["train"].select(scrub_list)

        scrub_dataloader = DataLoader(
            tokenized_ds["scrub"].remove_columns("text"),
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )

        new_list = []
        with open("new.txt", "r") as f:
            for line in f:
                new_list.append(int(line.strip()))

        tokenized_ds["new"] = tokenized_ds["train"].select(new_list)
        new_dataloader = DataLoader(
            tokenized_ds["new"].remove_columns("text"),
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )

        exclude_list = scrub_list + new_list
        tokenized_ds["residual"] = tokenized_ds["train"].select(
            [i for i in range(len(tokenized_ds["train"])) if i not in exclude_list]
        )

        if args.sample_ratio is not None:
            assert 0 <= args.sample_ratio <= 1
            select_list = [
                i
                for i in range(len(tokenized_ds["residual"]))
                if random.random() <= args.sample_ratio
            ]

            tokenized_ds["residual"] = tokenized_ds["residual"].select(select_list)

        residual_dataloader = DataLoader(
            tokenized_ds["residual"].remove_columns("text"),
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )

        print(
            "size of data: ",
            len(tokenized_ds["scrub"]),
            len(tokenized_ds["new"]),
            len(tokenized_ds["residual"]),
        )

        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            eps=1e-8,
        )
        scheduler = WarmupLinearSchedule(
            optimizer=optimizer,
            warmup_steps=500,
            t_total=min(50000, len(residual_dataloader)),
        )

        model.train()
        cur_time = time.time()

        stop = False
        stop_value = None
        total_updates = 0
        inner_step = 0
        freq = 10
        scrub_iterator = iter(scrub_dataloader)
        new_iterator = iter(new_dataloader)
        while total_updates < 2000 and not stop:
            for cur_step, batch_remain in enumerate(residual_dataloader, start=1):
                inner_step += 1
                batch_remain = batch_remain.to(device)
                pred_logits = F.log_softmax(model(**batch_remain).logits, dim=-1)
                tgt_logits = F.log_softmax(
                    original_model(**batch_remain).logits, dim=-1
                ).detach()
                loss_r = F.kl_div(
                    input=pred_logits,
                    target=tgt_logits,
                    log_target=True,
                    reduction="batchmean",
                )
                (loss_r * 0.1 / freq).backward()

                if inner_step % freq == 0:
                    total_updates += 1
                    try:
                        batch_forget = next(scrub_iterator)
                    except StopIteration:
                        scrub_iterator = iter(scrub_dataloader)
                        batch_forget = next(scrub_iterator)
                    try:
                        batch_new = next(new_iterator)
                    except StopIteration:
                        new_iterator = iter(new_dataloader)
                        batch_new = next(new_iterator)

                    batch_forget = batch_forget.to(device)
                    pred_logits = F.log_softmax(model(**batch_forget).logits, dim=-1)
                    tgt_logits = F.log_softmax(
                        forget_model(**batch_forget).logits, dim=-1
                    ).detach()
                    loss_align = F.kl_div(
                        input=pred_logits,
                        target=tgt_logits,
                        log_target=True,
                        reduction="batchmean",
                    )

                    batch_new = batch_new.to(device)
                    pred_logits = F.log_softmax(
                        original_model(**batch_new).logits, dim=-1
                    ).detach()
                    tgt_logits = F.log_softmax(
                        new_model(**batch_new).logits, dim=-1
                    ).detach()
                    tgt_align = F.kl_div(
                        input=pred_logits,
                        target=tgt_logits,
                        log_target=True,
                        reduction="batchmean",
                    )
                    loss_align = torch.abs(loss_align - tgt_align.item())
                    loss_align.backward()

                    if stop_value is None and args.stop == True:
                        stop_value = loss_align.item() * 0.1
                        print(f"Set stop value as {stop_value}")

                    optimizer.step()
                    scheduler.step(total_updates)
                    optimizer.zero_grad()

                    total_loss = loss_align.item() + 0.1 * loss_r.item()

                    if total_updates % 200 == 0:
                        print("Step", inner_step, "loss_align", loss_align.item())
                        if stop_value is not None and loss_align.item() <= stop_value:
                            print("Knowledge Gap achieved")
                            stop = True
                            break

                if total_updates >= 2000 or stop:
                    print("maximum number of updates")
                    break

        model.eval()
        print("finish unlearning!")
        print("total used time:", time.time() - cur_time)

        torch.save(model, f"unlearn_{influence}{size}{simple}_{seed}.pt")

        task_evaluator = evaluator("text-classification")

        eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            data=tokenized_ds["test"],
            tokenizer=tokenizer,
            metric=metric,
            label_mapping=label_mapping,
        )
        print("Test", eval_results)

        task_evaluator = evaluator("text-classification")

        eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            data=tokenized_ds["scrub"],
            tokenizer=tokenizer,
            metric=metric,
            label_mapping=label_mapping,
        )
        print("Forget", eval_results)

        torch.cuda.empty_cache()
