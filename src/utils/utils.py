# src/utils/utils.py
import time
import json
import logging
import openai
import pandas as pd
import torch
import torch.nn.utils.rnn
import uuid  # Add this import statement
from src.utils.config import CONFIG

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

def calculate_differential_weights(tokenized_labels, tokenizer, differences, high_weight=1, base_weight=1):
        """
        Calculate differential weights for tokenized labels (edited endings) based on diff
        erences.
        """
        # Initialize differential weights with base_weight
        differential_weights = torch.full(tokenized_labels.shape, fill_value=base_weight, dtype=torch.float)
        
        # Flatten the list of differences for easy checking
        difference_tokens_ids = set([item for sublist in [tokenizer.encode(diff, add_special_tokens=False) for diff in differences] for item in sublist])
        
        # Adjust weights for tokens present in differences
        for i, token_id in enumerate(tokenized_labels.squeeze().tolist()):
            if token_id in difference_tokens_ids:
                differential_weights[i] = high_weight
        
        return differential_weights    

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
            print(f"Input Sequence (ART/AblatdTimeTravel):{input_sequence}")
            print(f"Target Sequence: {target_sequence}")

        elif dataset_type == "TimeTravel":
            # TimeTravel Dataset: Input = premise + initial + original_ending + counterfactual; Output = edited_ending
            input_sequence = (
                f"{row['premise']}"
                f"{row['initial']}"
                f"{row['original_ending']} {separator_token}"
                f"{row['premise']} {row['counterfactual']}"
            )
            target_sequence = row['edited_ending']
            print(f"Input Sequence (Timetravel sequence):{input_sequence}")
            print(f"Target Sequence: {target_sequence}")

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        
        # Tokenize the input sequence with truncation to max_length and no padding here.
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )
        #print(f"Tokenized Inputs: {tokenized_inputs}")  # Debug print for tokenized inputs
              
        # Tokenize the edited ending, which serves as the target sequence for the model to generate.
        tokenized_ending = tokenizer.encode_plus(
            row['edited_ending'], truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )
        #print(f"Tokenized Ending: {tokenized_ending}") 
                
        # Calculate differential weights based on the list of differences provided for each token. This highlights tokens
        # that are directly associated with the differences, aiming to adjust the model's focus and learning priority.
        differential_weights = calculate_differential_weights(
            tokenized_ending['input_ids'].squeeze(), tokenizer, row['differences']
        )
        #print(f"Differential Weights: {differential_weights}")
        
        # Ensure that 'differential_weights' matches the length of 'labels'
        assert tokenized_ending['input_ids'].squeeze(0).size() == differential_weights.size(), "Mismatch between labels and differential weights length."
        
        #print(f"Input IDs: {tokenized_inputs['input_ids']}")
        #print(f"Attention Mask: {tokenized_inputs['attention_mask']}")
        #print(f"Labels: {tokenized_ending['input_ids']}")
        #print(f"Differential Weights: {differential_weights}")

        # Prepare the final output dictionary
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_ending['input_ids'].squeeze(0),
            'differential_weights': differential_weights.squeeze(0),  # Ensure the differential weights are correctly sized.
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
    
def collate_fn(batch, pad_token_id=0,attention_pad_value=0):
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
    differential_weights = [item['differential_weights'] for item in batch]
    premise = [item['premise'] for item in batch]
    initial = [item['initial'] for item in batch]
    original_ending = [item['original_ending'] for item in batch]
    counterfactual = [item['counterfactual'] for item in batch]
    edited_ending = [item['edited_ending'] for item in batch]

    print(f"Extracted Fields:\nPremises: {premise}\nInitials: {initial}\nOriginal Endings: {original_ending}\n"
          f"Counterfactuals: {counterfactual}\nEdited Endings: {edited_ending}")  # Debug print for field values


    # Padding sequences for 'input_ids', 'attention_masks', and 'labels'
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=attention_pad_value)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
   
    # Convert differential_weights to tensors and pad
    differential_weights_tensors = [dw.clone().detach().to(input_ids_padded.device) for dw in differential_weights]
    differential_weights_padded = torch.nn.utils.rnn.pad_sequence(differential_weights_tensors, batch_first=True, padding_value=1)

    # Debug prints
    #print(f"input_ids_padded shape: {input_ids_padded.shape}")
    #print(f"attention_masks_padded shape: {attention_masks_padded.shape}")
    #print(f"labels_padded shape: {labels_padded.shape}")
    #print(f"differential_weights_padded shape: {differential_weights_padded.shape}")


    # Return the padded tensors along with the additional fields for evaluation.
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded,
        'differential_weights': differential_weights_padded,
        'premise': premise,
        'initial': initial,
        'original_ending': original_ending,
        'counterfactual': counterfactual,
        'edited_ending': edited_ending,
    }

def chatgpt_zero_shot_inference(api_key, test_data):
    """
    Perform zero-shot inference using the OpenAI GPT model.

    Parameters:
        api_key (str): OpenAI API key.
        test_data (DataFrame): DataFrame containing the test data.

    Returns:
        results (list): List of dictionaries containing the results.
    """
    openai.api_key = api_key
    results = []

    max_retries = 3
    retry_delay = 5  # seconds

    for idx, row in test_data.iterrows():
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

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50
                )
                generated_text = response['choices'][0]['message']['content'].strip()
                break  # Exit the retry loop on success
            except Exception as e:
                #print(f"API call failed for row {idx} with error: {e}")
                if attempt < max_retries - 1:
                    #print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    #print("Max retries reached. Moving to the next item.")
                    generated_text = 'Error'  # Or any placeholder indicating a failure

        results.append({
            'story_id': row.get('story_id', str(uuid.uuid4())),  # Generate a UUID if story_id is not present
            'premise': row['premise'],
            'initial': row['initial'],
            'counterfactual': row['counterfactual'],
            'original_ending': row['original_ending'],
            'edited_ending': row['edited_ending'],
            'generated_text': generated_text
        })

    return results

def chatgpt_one_shot_inference(api_key, test_data, example_selection):
    """
    Perform one-shot inference using the OpenAI GPT model.

    Parameters:
        api_key (str): OpenAI API key.
        test_data (DataFrame): DataFrame containing the test data.
        example_selection (str): If "fixed", use a fixed example. If "random", select a random example for each query.

    Returns:
        results (list): List of dictionaries containing the results.
    """
    openai.api_key = api_key
    results = []

    # Prepare the fixed example (using the first row for simplicity)
    fixed_example = test_data.iloc[0] if example_selection == "fixed" else None

    for idx, row in test_data.iterrows():
        # Select a random example if required
        if example_selection == "random":
            example = test_data.sample(n=1).iloc[0]
        else:
            example = fixed_example

        prompt = (
            "Generate the adapted ending to fill these three aspects:\n"
            "1. Minimal Intervention: Adjust the story's original ending with the minimal changes required to align it with the counterfactual event. The edited ending should remain as close as possible to the original ending.\n"
            "2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.\n"
            "3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the initial event.\n\n"
            "Example:\n"
            f"Premise: {example['premise']}\n"
            f"Initial event: {example['initial']}\n"
            f"Original ending: {example['original_ending']}\n"
            f"Counterfactual event: {example['counterfactual']}\n"
            f"Adapted ending: {example['edited_ending']}\n\n"
            f"Premise: {row['premise']}\n"
            f"Initial event: {row['initial']}\n"
            f"Original ending: {row['original_ending']}\n"
            f"Counterfactual event: {row['counterfactual']}\n\n"
            "Now, generate the adapted ending:"
        )

        #print(f"Prompt for row {idx}: {prompt}")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50  # Use max_gen_length from the config if necessary
            )
            generated_text = response['choices'][0]['message']['content'].strip()

            # Remove "Adapted ending:" prefix if present
            if generated_text.lower().startswith("adapted ending:"):
                generated_text = generated_text[len("adapted ending:"):].strip()

            #print(f"Generated text for row {idx}: {generated_text}")

            results.append({
                'story_id': row['story_id'],
                'premise': row['premise'],
                'initial': row['initial'],
                'counterfactual': row['counterfactual'],
                'original_ending': row['original_ending'],
                'edited_ending': row['edited_ending'],
                'generated_text': generated_text  # Change key to "generated_text"
            })
        except Exception as e:
            print(f"API call failed for row {idx} with error: {e}")

    return results

def chatgpt_one_shot_inference(api_key, test_data, example_selection):
    """
    Perform one-shot inference using the OpenAI GPT model.

    Parameters:
        api_key (str): OpenAI API key.
        test_data (DataFrame): DataFrame containing the test data.
        example_selection (str): If "fixed", use a fixed example. If "random", select a random example for each query.

    Returns:
        results (list): List of dictionaries containing the results.
    """
    openai.api_key = api_key
    results = []

    max_retries = 5  # Increase the number of retries
    retry_delay = 10  # Increase the delay between retries (in seconds)

    # Prepare the fixed example (using the first row for simplicity)
    fixed_example = test_data.iloc[0] if example_selection == "fixed" else None

    for idx, row in test_data.iterrows():
        # Select a random example if required
        if example_selection == "random":
            example = test_data.sample(n=1).iloc[0]
        else:
            example = fixed_example

        prompt = (
            "Generate the adapted ending to fill these three aspects:\n"
            "1. Minimal Intervention: Adjust the story's original ending with the minimal changes required to align it with the counterfactual event. The edited ending should remain as close as possible to the original ending.\n"
            "2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.\n"
            "3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the initial event.\n\n"
            "Example:\n"
            f"Premise: {example['premise']}\n"
            f"Initial event: {example['initial']}\n"
            f"Original ending: {example['original_ending']}\n"
            f"Counterfactual event: {example['counterfactual']}\n"
            f"Adapted ending: {example['edited_ending']}\n\n"
            f"Premise: {row['premise']}\n"
            f"Initial event: {row['initial']}\n"
            f"Original ending: {row['original_ending']}\n"
            f"Counterfactual event: {row['counterfactual']}\n\n"
            "Now, generate the adapted ending:"
        )

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50  # Adjust if needed
                )
                generated_text = response['choices'][0]['message']['content'].strip()

                # Remove "Adapted ending:" prefix if present
                if generated_text.lower().startswith("adapted ending:"):
                    generated_text = generated_text[len("adapted ending:"):].strip()

                break  # Exit the retry loop on success
            except Exception as e:
                logging.error(f"API call failed for row {idx} with error: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Max retries reached for row {idx}. Moving to the next item.")
                    generated_text = 'Error'  # Or any placeholder indicating a failure

        results.append({
            'story_id': row['story_id'],
            'premise': row['premise'],
            'initial': row['initial'],
            'counterfactual': row['counterfactual'],
            'original_ending': row['original_ending'],
            'edited_ending': row['edited_ending'],
            'generated_text': generated_text  # Store the generated text or error message
        })

    return results