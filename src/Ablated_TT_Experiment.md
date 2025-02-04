### **Details: Ablated Time Travel Experiment**

The **Ablated Time Travel Experiment** is designed to investigate how well the model can rewrite story endings based on counterfactual events **without including the original ending** in the input. 

#### **Key Differences**
- In the **TimeTravel** dataset, the input includes:
  - `Premise`, `Initial Event`, `Original Ending`, and `Counterfactual Event`.
- In the **Ablated TimeTravel** experiment, the input excludes the **Original Ending** and includes only:
  - `Premise`, `Initial Event`, and `Counterfactual Event`.

#### **Goal**
Evaluate how removing the `Original Ending` from the input affects the quality of the rewritten ending. The outputs (`Edited Ending`) are still compared to the ground-truth reference.

---

### **Code Changes for Ablated TimeTravel Experiment**

#### **Step 1: Update the `preprocess_data` Function**
Modify the `preprocess_data` function to handle the **AblatedTimeTravel** dataset by excluding the `Original Ending` from the input sequence.

```python
def preprocess_data(row, tokenizer, dataset_type="TimeTravel"):
    """
    Prepares a single row of data for model input by tokenizing the text fields.

    Args:
        row (dict): A single row of data containing the fields required for the input.
        tokenizer (Tokenizer): The tokenizer to use for tokenizing the text fields.
        dataset_type (str): Specifies the type of dataset to determine preprocessing logic.
            Options: "ART", "TimeTravel", "AblatedTimeTravel".

    Returns:
        dict: A dictionary containing tokenized input, attention masks, and labels.
    """
    try:
        separator_token = "</s>"

        if dataset_type == "AblatedTimeTravel":
            # Exclude 'Original Ending' from input
            input_sequence = (
                f"{row['premise']} "
                f"{row['initial']} {separator_token} "
                f"{row['premise']} {row['counterfactual']}"
            )
            target_sequence = row['edited_ending']
            print(f"Input Sequence (AblatedTimeTravel): {input_sequence}")
            print(f"Target Sequence: {target_sequence}")

        elif dataset_type == "TimeTravel":
            # Include 'Original Ending' in the input
            input_sequence = (
                f"{row['premise']} "
                f"{row['initial']} "
                f"{row['original_ending']} {separator_token} "
                f"{row['premise']} {row['counterfactual']}"
            )
            target_sequence = row['edited_ending']
            print(f"Input Sequence (TimeTravel): {input_sequence}")
            print(f"Target Sequence: {target_sequence}")

        elif dataset_type == "ART":
            input_sequence = (
                f"{row['premise']} "
                f"{row['initial']} {separator_token} "
                f"{row['premise']} {row['counterfactual']}"
            )
            target_sequence = row['edited_ending']
            print(f"Input Sequence (ART): {input_sequence}")
            print(f"Target Sequence: {target_sequence}")

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # Tokenize the input sequence
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )

        # Tokenize the target sequence (Edited Ending)
        tokenized_ending = tokenizer.encode_plus(
            target_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )

        # Prepare the final output dictionary
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_ending['input_ids'].squeeze(0),
            'premise': row['premise'],
            'initial': row['initial'],
            'counterfactual': row['counterfactual'],
            'edited_ending': row['edited_ending'],
            **({'original_ending': row['original_ending']} if 'original_ending' in row else {})
        }

    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None
```

---

#### **Step 2: Modify the `create_dataloaders` Function**
Ensure the `dataset_type` is correctly passed as `AblatedTimeTravel` when loading the dataset.

```python
def create_dataloaders(data_path, tokenizer, batch_size, num_workers, dataset_type="TimeTravel"):
    """
    Creates DataLoader instances for each dataset specified by the configuration.

    Args:
        data_path (str): Path to the base directory containing data files.
        tokenizer (T5Tokenizer): The tokenizer to use for preprocessing the data.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of worker threads to use for loading data.
        dataset_type (str): Specifies the dataset type (e.g., "ART", "TimeTravel", "AblatedTimeTravel").

    Returns:
        dict: A dictionary of DataLoader objects, keyed by dataset type ('train', 'dev', 'test').
    """
    print(f"Creating dataloaders for dataset_type: {dataset_type}")  # Debug dataset type

    file_names = [CONFIG["train_file"], CONFIG["dev_file"], CONFIG["test_file"]]

    dataloaders = {}
    for file_name in file_names:
        file_path = Path(data_path) / file_name
        print(f"Loading file: {file_path} for dataset_type: {dataset_type}")  # Debug file path
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        # Create an instance of the dataset for each data file
        dataset = CustomJSONDataset(file_path, tokenizer, dataset_type=dataset_type)

        # Determine whether to shuffle: shuffle only for training
        shuffle = file_name == CONFIG["train_file"]

        # Create a DataLoader for each dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
            num_workers=num_workers,
            shuffle=shuffle
        )

        # Use the file name (without extension) as the key
        key = file_name.split('.')[0]
        dataloaders[key] = dataloader

    return dataloaders
```

---

#### **Step 3: Add AblatedTimeTravel Experiment in Main Script**
When setting up the model and training workflow, pass `dataset_type="AblatedTimeTravel"` to `create_dataloaders`.

```python
if CONFIG["dataset_type"] == "AblatedTimeTravel":
    print("Running Ablated TimeTravel Experiment...")
    dataloaders = create_dataloaders(
        data_path=CONFIG["data_dir"],
        tokenizer=tokenizer,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        dataset_type="AblatedTimeTravel"
    )
```

---

#### **Step 4: Update Configuration**
In the `CONFIG` file, set `dataset_type` to `AblatedTimeTravel` for this experiment.

```python
CONFIG = {
    ...
    "dataset_type": "AblatedTimeTravel",
    ...
}
```

---

### **Summary of Changes**
1. **Modified `preprocess_data`**:
   - Added logic to exclude `Original Ending` for `AblatedTimeTravel`.
2. **Updated `create_dataloaders`**:
   - Pass `AblatedTimeTravel` as the `dataset_type`.
3. **Main Script Update**:
   - Added conditional logic for running the Ablated TimeTravel experiment.
4. **Configuration Update**:
   - Added `AblatedTimeTravel` as a dataset type in the configuration.

This will enable the model to process the AblatedTimeTravel dataset and evaluate the impact of excluding the `Original Ending` during story rewriting. Let me know if you need further assistance!