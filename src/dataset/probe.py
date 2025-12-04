import random
from itertools import chain
from typing import List, Tuple

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer


def _build_dialog_prompt(
    question: str,
    answer: str,
    persona_instruction: str,
    user_tag: str,
    assistant_tag: str,
    include_sys_prompt: bool = False,
) -> str:
    """Constructs a replay-style prompt describing a prior Q&A turn."""

    question = question.strip()
    answer = answer.strip()
    persona_instruction = persona_instruction.strip()

    sys_prompt = ""
    if include_sys_prompt:
        sys_prompt = (
            "<<SYS>>\n"
            "You are reviewing a finished exchange. "
            f"{persona_instruction}"
            "\n<<SYS>>\n"
        )

    conversation = (
        f"[USER]: {question}\n"
        f"[Model]: {answer}\n"
        "[SYS]: Please assess your confidence level in your answer.\n"
        "[Model]:"
    )

    return (
        f"{user_tag} {sys_prompt}{conversation} {assistant_tag}"
    ).strip()


def _flatten_pairs(paired_sequences: List[List[str]]) -> List[str]:
    return list(chain.from_iterable(paired_sequences))


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _slice_pairs(
    pairs: List[List[str]],
    labels: List[List[bool]],
    start: int,
    end: int,
) -> Tuple[List[List[str]], List[List[bool]]]:
    return pairs[start:end], labels[start:end]


def confidence_function_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    user_tag: str = "",
    n_train: int = 512,
    assistant_tag: str = "",
    seed: int = 0,
    pos_tag: str = "Respond with decisive confidence.",
    neg_tag: str = "Respond with visible uncertainty and hesitation.",
    include_sys_prompt: bool = False,
):
    """Builds confident/unconfident replay prompts for probe training.

    Expected data format:
        question,str | confident_answer,str | unconfident_answer,str

    Returns a dictionary with train/eval/test splits where each split
    contains flattened data plus pairwise labels identifying the
    confident variant.
    """

    random.seed(seed)
    df = pd.read_csv(data_path)

    required_cols = {"question", "confident_answer", "unconfident_answer"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {', '.join(sorted(missing_cols))}"
        )

    paired_sequences: List[List[str]] = []
    paired_labels: List[List[bool]] = []

    for _, row in df.iterrows():
        question = _clean_text(row.get("question"))
        confident_answer = _clean_text(row.get("confident_answer"))
        unconfident_answer = _clean_text(row.get("unconfident_answer"))

        if not question or not confident_answer or not unconfident_answer:
            continue

        confident_prompt = _build_dialog_prompt(
            question,
            confident_answer,
            pos_tag,
            user_tag,
            assistant_tag,
            include_sys_prompt,
        )
        unconfident_prompt = _build_dialog_prompt(
            question,
            unconfident_answer,
            neg_tag,
            user_tag,
            assistant_tag,
            include_sys_prompt,
        )

        pair = [confident_prompt, unconfident_prompt]
        random.shuffle(pair)
        labels = [p == confident_prompt for p in pair]

        paired_sequences.append(pair)
        paired_labels.append(labels)

    total_pairs = len(paired_sequences)
    if total_pairs == 0:
        raise ValueError("No valid question-answer pairs were found in the dataset.")

    n_train = min(n_train, total_pairs)
    train_pairs, train_pair_labels = _slice_pairs(paired_sequences, paired_labels, 0, n_train)

    eval_end = min(total_pairs, n_train * 2)
    eval_pairs, eval_pair_labels = _slice_pairs(paired_sequences, paired_labels, n_train, eval_end)

    test_start = max(0, total_pairs - min(300, total_pairs))
    test_pairs, test_pair_labels = _slice_pairs(paired_sequences, paired_labels, test_start, total_pairs)

    train_data = _flatten_pairs(train_pairs)
    eval_data = _flatten_pairs(eval_pairs)
    test_data = _flatten_pairs(test_pairs)

    print(f"Train data: {len(train_data)}")
    print(f"Eval data: {len(eval_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        "train": {"data": train_data, "labels": train_pair_labels},
        "eval": {"data": eval_data, "labels": eval_pair_labels},
        "test": {"data": test_data, "labels": test_pair_labels},
    }


def honesty_function_dataset(
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        user_tag: str = "",
        n_train: int = 512,
        assistant_tag: str = "",
        seed: int = 0,
):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1][
        "statement"
    ].values.tolist()  # list of true statements
    false_statements = df[df["label"] == 0][
        "statement"
    ].values.tolist()  # list of false statements

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)  # tokenize statement

        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(
                truncated_tokens
            )  # convert tokens to string
            honest_statements.append(
                f"{user_tag} {template_str.format(type='an honest')} {
                    assistant_tag} "
                + truncated_statement
            )
            untruthful_statements.append(
                f"{user_tag} {template_str.format(type='an untruthful')} {
                    assistant_tag} "
                + truncated_statement
            )

    # Create training data
    ntrain = n_train
    combined_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]  # combine honest and untruthful statements
    train_data = combined_data[
        :ntrain
    ]  # split into train and test data

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append(
            [s == true_s for s in d]
        )

    train_data = np.concatenate(
        train_data
    ).tolist()

    # Create test data
    reshaped_data = np.array(
        [
            [honest, untruthful]
            for honest, untruthful in zip(
                honest_statements[:-1], untruthful_statements[1:]
            )
        ]
    ).flatten()
    eval_data = reshaped_data[
        ntrain: ntrain * 2
    ].tolist()

    test_data = reshaped_data[
        -300:-1
    ].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Eval data: {len(eval_data)}")
    print(f"Test data: {len(test_data)}")
    return {
        "train": {"data": train_data, "labels": train_labels},
        "eval": {"data": eval_data, "labels": [[1, 0]] * len(eval_data)},
        "test": {"data": test_data, "labels": [[1, 0]] * len(test_data)},
    }
