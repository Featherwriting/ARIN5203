#!/usr/bin/env python
"""Convenience script to train and evaluate the confidence probe outside the notebook."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.dataset.probe import confidence_function_dataset
from src.repe import repe_pipeline_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train confidence probe and evaluate layer-wise accuracy.")
    parser.add_argument("--model-path", default="./Mistral-7B-Instruct-v0.1", help="Path or identifier of the base model.")
    parser.add_argument("--data-path", default="./eval_data/confidence_pairs.csv", help="Path to the confidence training CSV.")
    parser.add_argument("--output-probe", default="./trained_probe/confidence/rep_reader.pkl", help="Where to save the trained probe reader.")
    parser.add_argument("--plot-output", default='./trained_probe/confidence/layer_acc.png', help="Optional path to save the accuracy plot. If omitted the plot is shown interactively.")
    parser.add_argument("--user-tag", default="[INST]", help="User tag prefix for prompts.")
    parser.add_argument("--assistant-tag", default="[/INST]", help="Assistant tag suffix for prompts.")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length for dataset prompts.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for hidden-state extraction.")
    parser.add_argument("--train-limit", type=int, default=0, help="Limit the number of training examples (0 means all).")
    parser.add_argument("--direction-method", default="pca", help="RepReader direction finding strategy.")
    parser.add_argument("--n-difference", type=int, default=1, help="Number of pairwise differences to compute per sample.")
    parser.add_argument("--rep-token", type=int, default=-1, help="Token position used when reading hidden states.")
    return parser.parse_args()


def _load_model_and_tokenizer(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    use_fast = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast,
        padding_side="left",
        legacy=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    return model, tokenizer


def _prepare_dataset(args: argparse.Namespace, tokenizer) -> dict:
    dataset = confidence_function_dataset(
        args.data_path,
        tokenizer,
        args.user_tag,
        args.max_seq_length,
        args.assistant_tag,
    )
    if args.train_limit and args.train_limit > 0:
        dataset["train"]["data"] = dataset["train"]["data"][: args.train_limit]
        dataset["train"]["labels"] = dataset["train"]["labels"][: args.train_limit]
    return dataset


def _train_probe(args: argparse.Namespace, rep_reading_pipeline, dataset, hidden_layers, rep_token):
    return rep_reading_pipeline.get_directions(
        dataset["train"]["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=args.n_difference,
        train_labels=dataset["train"]["labels"],
        direction_method=args.direction_method,
        batch_size=args.batch_size,
        show_progress=True,
        verbose=True,
    )


def _evaluate_probe(rep_reading_pipeline, dataset, confidence_rep_reader, hidden_layers, rep_token, batch_size):
    h_tests = rep_reading_pipeline(
        dataset["test"]["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        rep_reader=confidence_rep_reader,
        batch_size=batch_size,
    )

    results = {}
    for layer in hidden_layers:
        layer_h = [h[layer] for h in h_tests]
        paired = [layer_h[i:i + 2] for i in range(0, len(layer_h), 2)]
        sign = confidence_rep_reader.direction_signs[layer]
        eval_func = min if sign == -1 else max
        correct = [eval_func(pair) == pair[0] for pair in paired if len(pair) == 2]
        results[layer] = float(np.mean(correct)) if correct else 0.0
    return results


def _plot_results(results: dict, output_path: str | None = None):
    layers = list(results.keys())
    accuracies = [results[layer] for layer in layers]

    plt.figure(figsize=(10, 4))
    plt.plot(layers, accuracies, label="Test", marker="o")
    plt.title("Confidence Probe Accuracy by Layer")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main():
    args = _parse_args()
    repe_pipeline_registry()

    model, tokenizer = _load_model_and_tokenizer(args.model_path)
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    dataset = _prepare_dataset(args, tokenizer)

    confidence_rep_reader = _train_probe(args, rep_reading_pipeline, dataset, hidden_layers, args.rep_token)

    results = _evaluate_probe(
        rep_reading_pipeline,
        dataset,
        confidence_rep_reader,
        hidden_layers,
        args.rep_token,
        args.batch_size,
    )

    output_path = Path(args.output_probe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fout:
        pickle.dump(confidence_rep_reader, fout)
    print(f"Saved probe to {output_path}")

    _plot_results(results, args.plot_output)

    best_layer = max(results, key=results.get)
    print(f"Best layer {best_layer}: accuracy={results[best_layer]:.3f}")


if __name__ == "__main__":
    main()
