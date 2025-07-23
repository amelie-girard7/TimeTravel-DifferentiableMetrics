import time
import json
import logging
import openai
import pandas as pd
import torch
import torch.nn.utils.rnn
import uuid  # Add this import statement
from src.cpo.utils.config import CONFIG

logger = logging.getLogger(__name__)


def count_json_lines(file_path):
    """
    Counts the number of lines in a JSON file, which is useful for estimating
    the dataset size or for iterative processing without loading the entire file.
    """
    logger.info(f"Counting lines in file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")


def load_first_line_from_json(file_path):
    """
    Loads and parses the first line from a JSON file. This is useful for inspecting
    the data structure without loading the entire file.
    """
    logger.info(f"Loading first line from JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.loads(next(file))
    except Exception as e:
        logger.error(f"Error reading from {file_path}: {e}")
        raise IOError(f"Error reading from {file_path}: {e}")




def preprocess_data(row, tokenizer):
    """
    Prepares a single row of data for model input by tokenizing the text fields.

    Args:
        row (dict): A single row of data containing the fields required for the input.
        tokenizer (Tokenizer): The tokenizer to use for tokenizing the text fields.

    Returns:
        dict: A dictionary containing tokenized input, attention masks, and labels.
    """
    try:
        dataset_type = CONFIG["dataset_type"]  # Access dataset_type from CONFIG
        separator_token = "</s>"

        if dataset_type in {"ART", "AblatedTimeTravel"}:
            # Input = premise + initial + counterfactual; Output = edited_ending
            input_sequence = (
                f"{row['premise']}"
                f"{row['initial']} {separator_token}"
                f"{row['premise']} {row['counterfactual']}"
            )
            target_sequence = row['edited_ending']

        elif dataset_type == "TimeTravel":
            # TimeTravel Dataset: Input = premise + initial + original_ending + counterfactual; Output = edited_ending
            input_sequence = (
                f"{row['premise']}"
                f"{row['initial']}"
                f"{row['original_ending']} {separator_token}"
                f"{row['premise']} {row['counterfactual']}"
            )
            target_sequence = row['edited_ending']

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # Tokenize the input sequence with truncation to max_length and no padding here.
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )

        # Tokenize the edited ending, which serves as the target sequence for the model to generate.
        tokenized_ending = tokenizer.encode_plus(
            row['edited_ending'], truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )

        # Prepare the final output dictionary
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_ending['input_ids'].squeeze(0),
            # Include non-tokenized data for metric calculations.
            'premise': row['premise'],
            'initial': row['initial'],
            'counterfactual': row['counterfactual'],
            'edited_ending': row['edited_ending'],
            # Include original_ending only if available
            **({'original_ending': row['original_ending']} if 'original_ending' in row else {})
        }

    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None


def collate_fn(batch, pad_token_id=0, attention_pad_value=0):
    """
    Collates a batch of preprocessed data into a format suitable for model input,
    including padding to equalize the lengths of sequences within the batch.
    """
    print(f"Batch before collation: {batch}")  # Debug print to show the raw batch data
    # Unpack the batch into separate lists for each field.
    # Extract fields explicitly to prevent ordering issues
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    premise = [item['premise'] for item in batch]
    initial = [item['initial'] for item in batch]
    original_ending = [item['original_ending'] for item in batch]
    counterfactual = [item['counterfactual'] for item in batch]
    edited_ending = [item['edited_ending'] for item in batch]

    # Padding sequences for 'input_ids', 'attention_masks', and 'labels'
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True,
                                                             padding_value=attention_pad_value)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    # Return the padded tensors along with the additional fields for evaluation.
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded,
        'premise': premise,
        'initial': initial,
        'original_ending': original_ending,
        'counterfactual': counterfactual,
        'edited_ending': edited_ending,
    }


