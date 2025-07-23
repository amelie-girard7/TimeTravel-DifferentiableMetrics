import os
import random
import time
import json
import logging
import pandas as pd
import torch
import torch.nn.utils.rnn
import uuid 
from src.mle.utils.config import CONFIG
from openai import OpenAI
import google.generativeai as genai
from google.generativeai import types

logger = logging.getLogger(__name__)

# shared client reads OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        dict: A dictionary containing tokenized input,  and labels.
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
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),  # Add this line
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
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    premise = [item['premise'] for item in batch]
    initial = [item['initial'] for item in batch]
    counterfactual = [item['counterfactual'] for item in batch]
    edited_ending = [item['edited_ending'] for item in batch]


    # Handle original_ending - use empty string if not present
    original_ending = []
    for item in batch:
        if 'original_ending' in item:
            original_ending.append(item['original_ending'])
        else:
            original_ending.append("")  # Default empty string

    # Padding sequences for 'input_ids', and 'labels'
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask_padded = (input_ids_padded != pad_token_id).long()

    # Return the padded tensors along with the additional fields for evaluation.
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'premise': premise,
        'initial': initial,
        'original_ending': original_ending,
        'counterfactual': counterfactual,
        'edited_ending': edited_ending,
    }

def chatgpt_zero_shot_inference(test_data: pd.DataFrame) -> list:
    """
    Perform zero-shot inference using the OpenAI GPT model.

    Steps:
    1. Iterate over each row in test_data.
    2. Build the fixed zero-shot prompt per row.
    3. Call client.chat.completions.create() with retry logic.
    4. Collect and return results as list of dicts.
    """
    results = []
    max_retries = 3
    retry_delay = 5  # seconds

    for idx, row in test_data.iterrows():
        # 2) Build prompt (exactly as before)
        prompt = (
            "Generate the adapted ending to fill these three aspects:\n"
            "1. Minimal Intervention: Adjust the story's original ending with the minimal changes required to align it with the counterfactual event. The edited ending should remain as close as possible to the original ending.\n"
            "2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.\n"
            "3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the initial event.\n\n"
            f"Premise: {row['premise']}\n"
            f"Initial event: {row['initial']}\n"
            f"Original ending: {row['original_ending']}\n"
            f"Counterfactual event: {row['counterfactual']}\n\n"
            "Now, generate the adapted ending:"
        )

        # 3) Call OpenAI API with retries
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=50,
                    #temperature=0.0
                )
                generated_text = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                logger.warning(f"[zero-shot] row {idx} failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    generated_text = "Error"

        # 4) Append result
        results.append({
            "story_id":        row.get("story_id", str(uuid.uuid4())),
            "premise":         row["premise"],
            "initial":         row["initial"],
            "counterfactual":  row["counterfactual"],
            "original_ending": row["original_ending"],
            "edited_ending":   row["edited_ending"],
            "generated_text":  generated_text
        })

    return results

def chatgpt_one_shot_inference(test_data: pd.DataFrame, example_selection: str) -> list:
    """
    Perform one-shot inference using the OpenAI GPT model.

    Steps:
    1. Choose a fixed or random example.
    2. Iterate over each row, building a few-shot prompt.
    3. Call client.chat.completions.create() with retry logic.
    4. Collect and return results as list of dicts.
    """
    results = []
    max_retries = 5
    retry_delay = 10  # seconds

    # 1) Prepare fixed example if needed
    fixed_example = test_data.iloc[0] if example_selection == "fixed" else None
    records = test_data.to_dict("records")  # for random sampling

    for idx, row in test_data.iterrows():
        # 1) Select example
        example = random.choice(records) if example_selection == "random" else fixed_example

        # 2) Build the few-shot prompt (original prompt + example)
        prompt = (
            "Generate the adapted ending to fill these three aspects:\n"
            "1. Minimal Intervention: Adjust the story's original ending with the minimal changes required to align it with the counterfactual event. The edited ending should remain as close as possible to the original ending.\n"
            "2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.\n"
            "3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the initial event.\n\n"
            f"Premise: {row['premise']}\n"
            f"Initial event: {row['initial']}\n"
            f"Original ending: {row['original_ending']}\n"
            f"Counterfactual event: {row['counterfactual']}\n\n"
            "Now, generate the adapted ending:"
        )

        # 3) Call API with retries
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=50,
                    #temperature=0.0
                )
                generated_text = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                logger.warning(f"[one-shot] row {idx} failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    generated_text = "Error"

        # 4) Append result
        results.append({
            "story_id":        row.get("story_id", str(uuid.uuid4())),
            "premise":         row["premise"],
            "initial":         row["initial"],
            "counterfactual":  row["counterfactual"],
            "original_ending": row["original_ending"],
            "edited_ending":   row["edited_ending"],
            "generated_text":  generated_text
        })

    return results

def gemini_zero_shot_inference(api_key, test_data):
    """
    Perform zero-shot inference using Gemini 2.0 (flash version) with text-only input.
    Uses the Google Generative AI Python SDK with a generation configuration similar to GPT.
    """
    # Corrected the logic in try except block
    genai.configure(api_key=api_key)
    # Create a client instance and use the flash model.
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Generation configuration: set max tokens and optional parameters.
    generation_config = types.GenerationConfig(
        max_output_tokens=50,  # Equivalent to GPT's max_tokens=50
    )

    results = []
    max_retries = 3
    retry_delay = 5  # seconds

    for idx, row in test_data.iterrows():
        prompt = (
            "Generate the adapted ending to fill these three aspects:\n"
            "1. Minimal Intervention: Adjust the story's original ending with the minimal changes required to align it with the counterfactual event. The edited ending should remain as close as possible to the original ending.\n"
            "2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.\n"
            "3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the initial event.\n\n"
            "Example:\n"
            f"Premise: {row['premise']}\n"
            f"Initial event: {row['initial']}\n"
            f"Original ending: {row['original_ending']}\n"
            f"Counterfactual event: {row['counterfactual']}\n\n"
            "Now, generate the adapted ending:"
        )

        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                generated_text = response.text
                break  # Exit loop on success
            except Exception as e:
                logger.error(f"Gemini API call failed for row {idx} with error: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    generated_text = 'Error'
        results.append({
            'story_id': row.get('story_id', str(uuid.uuid4())),
            'premise': row['premise'],
            'initial': row['initial'],
            'counterfactual': row['counterfactual'],
            'original_ending': row['original_ending'],
            'edited_ending': row['edited_ending'],
            'generated_text': generated_text
        })

    return results

def gemini_one_shot_inference(api_key, test_data, example_selection):
    """
    Perform one-shot inference using Gemini 2.0 (flash version) with a prompt that includes an example.
    Uses the Google Generative AI Python SDK with a generation configuration.
    """
    # Corrected the logic in try except block
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    generation_config = types.GenerationConfig(
        max_output_tokens=50,
    )

    results = []
    max_retries = 5
    retry_delay = 10  # seconds

    # Choose a fixed example if required.
    fixed_example = test_data.iloc[0] if example_selection == "fixed" else None

    for idx, row in test_data.iterrows():
        example = (
            test_data.sample(n=1).iloc[0]
            if example_selection == "random"
            else fixed_example
        )

        prompt = (
            "Generate the adapted ending to fill these three aspects:\n"
            "1. Minimal Intervention: Adjust the story's original ending with the minimal changes required to align it with the counterfactual event. The edited ending should remain as close as possible to the original ending.\n"
            "2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.\n"
            "3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the initial event.\n\n"
            "Example:\n"
            f"Premise: {row['premise']}\n"
            f"Initial event: {row['initial']}\n"
            f"Original ending: {row['original_ending']}\n"
            f"Counterfactual event: {row['counterfactual']}\n\n"
            "Now, generate the adapted ending:"
        )

        for attempt in range(max_retries):
            try:
                response =  model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                generated_text = response.text
                break  # Exit on success
            except Exception as e:
                logger.error(f"Gemini API call failed for row {idx} with error: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    generated_text = 'Error'
        results.append({
            'story_id': row['story_id'],
            'premise': row['premise'],
            'initial': row['initial'],
            'counterfactual': row['counterfactual'],
            'original_ending': row['original_ending'],
            'edited_ending': row['edited_ending'],
            'generated_text': generated_text
        })

    return results
