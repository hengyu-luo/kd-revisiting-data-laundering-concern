#!/usr/bin/env python3
"""
Baseline Experiment for Data Contamination in Knowledge Distillation

This script runs the complete baseline experiment pipeline:
1. Trains teacher models (clean vs. contaminated) for various model sizes
2. Trains student models for classification:
   - Clean Student (purely supervised on clean data)
   - Dirty Student (purely supervised on contaminated data)
   - Distilled Students from either Clean or Dirty teacher with:
       (a) Hard label distillation
       (b) Soft label - forward KL
       (c) Soft label - reverse KL
3. Saves all models for further analysis.
4. Saves experiment summaries and checkpoints for downstream evaluation.

Usage:
    python 1_training_experiment.py --task [imdb|snli|agnews|emotion|banking77|tweet_sentiment|rotten_tomatoes|20newsgroups]
                                      --teacher_model [bert-large-uncased|bert-base-uncased]
                                      --student_model [bert-base-uncased|distilbert-base-uncased]
                                      --contamination_mode [add|replace]
                                      --seed 42
                                      --train_subset_ratio 0.1
"""

import os
import random
import numpy as np
import torch
import json
import argparse
import logging
from datetime import datetime
from functools import partial
import shutil
from typing import Dict

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
CONTAMINATION_RATIO = 1  # 100% contamination for baseline
NUM_EPOCHS = 10
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TRAIN_STUDENT = 128
BATCH_SIZE_EVAL = 4 # Not used for evaluation in this script anymore
LEARNING_RATE = 2e-5 # Default, might need tuning
# MAX_LENGTH is chosen dynamically from token-length statistics.
OUTPUT_DIR = './results'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, 'experiments')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics') # For results JSON

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
# --- Helper function to check if a model checkpoint is valid ---
def is_valid_checkpoint(path):
    """Checks if a path contains a valid Hugging Face model checkpoint."""
    if not os.path.isdir(path):
        return False
    # Check for common model files
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_pytorch_model = os.path.exists(os.path.join(path, "pytorch_model.bin")) or \
                        os.path.exists(os.path.join(path, "model.safetensors"))
    return has_config and has_pytorch_model

# -----------------------------------------------------------------------------------
# Helper Functions (Preprocessing, Dataset Info)
# -----------------------------------------------------------------------------------


# Extend this map as new aliases are introduced in experiments
MODEL_NAME_ALIASES: Dict[str, str] = {
    # Decoder-style LMs used as classification teachers
    "llama3.2-1B": "meta-llama/Llama-3.2-1B",
    "qwen3-0.6B": "Qwen/Qwen3-0.6B",
}


def resolve_model_name(identifier: str) -> str:
    """Return the concrete checkpoint path for a possibly-aliased identifier."""
    resolved = MODEL_NAME_ALIASES.get(identifier, identifier)
    # Always expand user directories so local mirrors like ~/checkpoints/... work.
    return os.path.expanduser(resolved)


def preprocess_function(examples, tokenizer, task, max_length):
    """Tokenize and process input data for supported classification tasks."""
    dataset_info = get_dataset_info(task)
    
    if task in ['imdb', 'agnews', 'emotion', 'banking77', 'tweet_sentiment', 'rotten_tomatoes', '20newsgroups']:
        # Single text field classification
        text_field = dataset_info.get('text_field', 'text')
        return tokenizer(examples[text_field], truncation=True, padding='max_length', max_length=max_length)

    if task == 'snli':
        # SNLI has premise and hypothesis
        premises = [p if p is not None else "" for p in examples['premise']]
        hypotheses = [h if h is not None else "" for h in examples['hypothesis']]
        return tokenizer(premises, hypotheses, truncation=True, padding='max_length', max_length=max_length)

    raise ValueError(f"Unsupported task: {task}")

def get_dataset_info(task):
    """Return dataset configuration (without hardcoded sizes)."""
    
    # Basic configuration for each dataset
    dataset_configs = {
        'imdb': {'name': 'imdb', 'train_split': 'train', 'test_split': 'test', 'num_labels': 2, 'text_field': 'text', 'label_field': 'label'},
        'snli': {'name': 'snli', 'train_split': 'train', 'test_split': 'test', 'num_labels': 3, 'premise_field': 'premise', 'hypothesis_field': 'hypothesis', 'label_field': 'label'},
        'agnews': {'name': 'ag_news', 'train_split': 'train', 'test_split': 'test', 'num_labels': 4, 'text_field': 'text', 'label_field': 'label'},
        'emotion': {'name': 'dair-ai/emotion', 'train_split': 'train', 'test_split': 'test', 'num_labels': 6, 'text_field': 'text', 'label_field': 'label'},
        'banking77': {'name': 'PolyAI/banking77', 'train_split': 'train', 'test_split': 'test', 'num_labels': 77, 'text_field': 'text', 'label_field': 'label'},
        'tweet_sentiment': {'name': 'tweet_eval', 'config': 'sentiment', 'train_split': 'train', 'test_split': 'test', 'num_labels': 3, 'text_field': 'text', 'label_field': 'label'},
        'rotten_tomatoes': {'name': 'rotten_tomatoes', 'train_split': 'train', 'test_split': 'test', 'num_labels': 2, 'text_field': 'text', 'label_field': 'label'},
        '20newsgroups': {'name': 'SetFit/20_newsgroups', 'train_split': 'train', 'test_split': 'test', 'num_labels': 20, 'text_field': 'text', 'label_field': 'label'},
    }
    
    if task not in dataset_configs:
        raise ValueError(f"Unsupported task: {task}")
    
    return dataset_configs[task]


def load_raw_dataset(task):
    """Load dataset splits without tokenization."""
    dataset_info = get_dataset_info(task)
    logger.info(f"Loading dataset: {dataset_info['name']} with config: {dataset_info.get('config')}")

    trust_remote_code_datasets = ['PolyAI/banking77']
    needs_trust_remote_code = any(name in dataset_info['name'] for name in trust_remote_code_datasets)

    if 'config' in dataset_info:
        raw_datasets = load_dataset(
            dataset_info['name'],
            dataset_info['config'],
            trust_remote_code=needs_trust_remote_code
        )
    else:
        raw_datasets = load_dataset(
            dataset_info['name'],
            trust_remote_code=needs_trust_remote_code
        )

    train_dataset = raw_datasets[dataset_info['train_split']]
    test_dataset = raw_datasets[dataset_info['test_split']]

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    if task == 'snli':
        dataset = dataset.filter(lambda x: x[dataset_info['label_field']] != -1)

    actual_train_size = len(dataset['train'])
    actual_test_size = len(dataset['test'])
    dataset_info['original_train_size'] = actual_train_size
    dataset_info['original_test_size'] = actual_test_size
    dataset_info['original_ab_ratio'] = actual_train_size / actual_test_size if actual_test_size > 0 else 0

    logger.info(f"Dataset loaded - Original sizes: Train={actual_train_size}, Test={actual_test_size}, A/B ratio={dataset_info['original_ab_ratio']:.2f}")
    return dataset, dataset_info


def _columns_to_remove(dataset_split, dataset_info):
    removable = list(dataset_split.column_names)
    label_field = dataset_info.get('label_field', 'label')
    if label_field in removable:
        removable.remove(label_field)
    if 'labels' in removable:
        removable.remove('labels')
    return removable


def tokenize_dataset_for_model(dataset_split, tokenizer, max_length, dataset_info, task):
    """Tokenize dataset for a single model (teacher or eval)."""
    preprocess_fn = partial(
        preprocess_function,
        tokenizer=tokenizer,
        task=task,
        max_length=max_length,
    )

    columns_to_remove = _columns_to_remove(dataset_split, dataset_info)
    tokenized = dataset_split.map(
        preprocess_fn,
        batched=True,
        remove_columns=columns_to_remove
    )

    label_field = dataset_info.get('label_field', 'label')
    if label_field in tokenized.column_names and label_field != 'labels':
        tokenized = tokenized.rename_column(label_field, 'labels')

    format_columns = [col for col in tokenized.column_names if col in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']]
    tokenized.set_format('torch', columns=format_columns)
    return tokenized


def tokenize_dataset_dual(dataset_split, teacher_tokenizer, student_tokenizer, max_length, dataset_info, task):
    """Tokenize dataset with both teacher and student tokenizers for online dual-tokenizer distillation."""
    def _tokenize(examples):
        teacher_encoded = preprocess_function(
            examples, teacher_tokenizer, task, max_length
        )
        student_encoded = preprocess_function(
            examples, student_tokenizer, task, max_length
        )
        result = {}
        for key, value in teacher_encoded.items():
            if key == 'labels':
                continue
            result[f"teacher_{key}"] = value
        for key, value in student_encoded.items():
            if key == 'labels':
                continue
            result[f"student_{key}"] = value
        label_field = dataset_info.get('label_field', 'label')
        if label_field in examples:
            result['labels'] = examples[label_field]
        elif 'labels' in examples:
            result['labels'] = examples['labels']
        return result

    columns_to_remove = _columns_to_remove(dataset_split, dataset_info)
    dual_dataset = dataset_split.map(
        _tokenize,
        batched=True,
        remove_columns=columns_to_remove
    )
    format_columns = [col for col in dual_dataset.column_names if col.startswith(("teacher_", "student_")) or col == 'labels']
    dual_dataset.set_format('torch', columns=format_columns)
    return dual_dataset
def analyze_tokenization_stats(dataset, tokenizer, max_length, task_info):
    """
    Analyzes and calculates detailed tokenization statistics for a given dataset.

    Args:
        dataset (Dataset): The raw, unprocessed Hugging Face dataset.
        tokenizer: The Hugging Face tokenizer.
        max_length (int): The configured maximum sequence length for truncation.
        task_info (dict): A dictionary containing dataset metadata (e.g., text field names).

    Returns:
        dict: A dictionary containing all calculated statistics.
    """
    logger.info(f"Analyzing tokenization stats for a dataset split...")
    
    original_lengths = []
    
    # Determine the text fields to use based on the task info
    text_fields = []
    if 'text_field' in task_info:
        text_fields.append(task_info['text_field'])
    if 'premise_field' in task_info:
        text_fields.append(task_info['premise_field'])
    if 'hypothesis_field' in task_info:
        text_fields.append(task_info['hypothesis_field'])
    if 'article_field' in task_info:
        text_fields.append(task_info['article_field'])

    # Iterate through the dataset to get the original token length of each sample
    for example in dataset:
        # Combine text from all relevant fields (handles single and multi-field tasks)
        full_text = " ".join([example[field] for field in text_fields if example.get(field)])
        
        # Tokenize without truncation or padding to get the true length
        token_ids = tokenizer.encode(full_text, add_special_tokens=True)
        original_lengths.append(len(token_ids))

    if not original_lengths:
        return {"error": "Dataset is empty or text fields are incorrect."}

    # Convert the list of lengths to a Numpy array for efficient calculations
    lengths_arr = np.array(original_lengths)
    total_samples = len(lengths_arr)

    # --- Calculate Statistics ---

    # 1. Basic stats about original length
    avg_original_token_count = np.mean(lengths_arr)
    
    # 2. Calculate padding and truncation counts for each sample
    padded_counts = np.maximum(0, max_length - lengths_arr)
    truncated_counts = np.maximum(0, lengths_arr - max_length)
    
    avg_padded_token_count = np.mean(padded_counts)
    avg_truncated_token_count = np.mean(truncated_counts)

    # 3. Advanced stats about truncation impact
    num_samples_truncated = np.sum(lengths_arr > max_length)
    percentage_samples_truncated = (num_samples_truncated / total_samples) * 100
    
    total_original_tokens = np.sum(lengths_arr)
    total_truncated_tokens = np.sum(truncated_counts)
    percentage_tokens_truncated = (total_truncated_tokens / total_original_tokens) * 100 if total_original_tokens > 0 else 0
    
    # 4. Distribution stats (often more useful than just the average)
    token_dist = {
        'min': int(np.min(lengths_arr)),
        'max': int(np.max(lengths_arr)),
        'median_p50': int(np.median(lengths_arr)),
        'p25': int(np.percentile(lengths_arr, 25)),
        'p75': int(np.percentile(lengths_arr, 75)),
        'p90': int(np.percentile(lengths_arr, 90)),
    }

    # 5. Compile all stats into a final dictionary
    stats = {
        'total_samples': total_samples,
        'max_length_setting': max_length,
        'avg_original_token_count': float(f"{avg_original_token_count:.2f}"),
        'token_count_distribution': token_dist,
        'avg_padded_token_count': float(f"{avg_padded_token_count:.2f}"),
        'avg_truncated_token_count': float(f"{avg_truncated_token_count:.2f}"),
        'num_samples_truncated': int(num_samples_truncated),
        'percentage_samples_truncated': float(f"{percentage_samples_truncated:.2f}"),
        'percentage_total_tokens_truncated': float(f"{percentage_tokens_truncated:.2f}")
    }
    
    logger.info(f"Analysis complete. {num_samples_truncated} samples ({percentage_samples_truncated:.2f}%) will be truncated.")
    return stats

def create_contaminated_train_data(clean_train, test_dataset, contamination_ratio, mode, seed, dataset_info):
    """Create contaminated training data by adding or replacing with test samples."""
    num_contaminate = int(len(test_dataset) * contamination_ratio)
    num_contaminate = min(num_contaminate, len(test_dataset)) # Cap at test set size
    
    random.seed(seed)
    contamination_indices = random.sample(range(len(test_dataset)), num_contaminate)
    
    # Ensure contamination_data has the same features as clean_train
    # This is especially important if test_dataset was not tokenized identically
    # For simplicity, assuming test_dataset is already tokenized and formatted like clean_train here.
    # If not, it should be processed with the same preprocess_function.
    contamination_data = test_dataset.select(contamination_indices)
    
    if mode == 'add':
        contaminated_train = concatenate_datasets([clean_train, contamination_data])
        contaminated_train = contaminated_train.shuffle(seed=seed)
        logger.info(f"Added {num_contaminate} contaminated samples to training data. New size: {len(contaminated_train)}")
    elif mode == 'replace':
        if num_contaminate > len(clean_train):
            logger.warning(f"Contamination samples ({num_contaminate}) exceed clean train samples ({len(clean_train)}). Replacing all clean samples.")
            num_contaminate = len(clean_train)
        
        replace_indices = set(random.sample(range(len(clean_train)), num_contaminate))
        clean_filtered = clean_train.filter(lambda example, idx: idx not in replace_indices, with_indices=True)
        contaminated_train = concatenate_datasets([clean_filtered, contamination_data])
        contaminated_train = contaminated_train.shuffle(seed=seed)
        logger.info(f"Replaced {num_contaminate} clean samples with contaminated samples. New size: {len(contaminated_train)}")
    else:
        raise ValueError(f"Unsupported contamination mode: {mode}")
    return contaminated_train, contamination_indices

# -----------------------------------------------------------------------------------
# MultiDistillationTrainer
# -----------------------------------------------------------------------------------
class MultiDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, distillation_type='supervised', temperature=2.0, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.temperature = temperature

        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.to(self.model.device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        student_input_ids = inputs.pop("student_input_ids", None)
        student_attention_mask = inputs.pop("student_attention_mask", None)
        student_token_type_ids = inputs.pop("student_token_type_ids", None)
        teacher_input_ids = inputs.pop("teacher_input_ids", None)
        teacher_attention_mask = inputs.pop("teacher_attention_mask", None)
        teacher_token_type_ids = inputs.pop("teacher_token_type_ids", None)
        
        output_hidden_states = False
        output_attentions = False

        def _build_model_inputs(ids, mask, token_type):
            model_inputs = {}
            if ids is not None:
                model_inputs['input_ids'] = ids
            if mask is not None:
                model_inputs['attention_mask'] = mask
            if token_type is not None:
                model_inputs['token_type_ids'] = token_type
            return model_inputs

        student_inputs = _build_model_inputs(student_input_ids, student_attention_mask, student_token_type_ids)
        if not student_inputs:
            student_inputs = inputs

        student_outputs = model(
            **student_inputs,
            labels=labels if labels is not None and self.distillation_type == "supervised" else None,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        student_logits = student_outputs.logits
        loss = None

        # Supervised modes
        if self.distillation_type == 'supervised': # Classification
            if labels is None: raise ValueError("Labels must be provided for supervised classification training.")
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(student_logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # Distillation modes
        elif self.teacher_model is not None:
            teacher_inputs = _build_model_inputs(teacher_input_ids, teacher_attention_mask, teacher_token_type_ids)
            if not teacher_inputs:
                teacher_inputs = inputs
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    **teacher_inputs,
                    output_hidden_states=output_hidden_states, 
                    output_attentions=output_attentions         
                )
                teacher_logits = teacher_outputs.logits

            if self.distillation_type == 'hard':
                teacher_preds = torch.argmax(teacher_logits, dim=-1)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(student_logits.view(-1, self.model.config.num_labels), teacher_preds.view(-1))
            
            elif self.distillation_type in ['soft_fwd', 'soft_rev']:
                if self.distillation_type == 'soft_fwd':
                    student_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
                    teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
                    loss_fct = nn.KLDivLoss(reduction='batchmean', log_target=False)
                    kl_loss = loss_fct(student_log_prob, teacher_prob)
                else:
                    teacher_log_prob = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                    student_prob = F.softmax(student_logits / self.temperature, dim=-1)
                    loss_fct = nn.KLDivLoss(reduction='batchmean', log_target=False)
                    kl_loss = loss_fct(teacher_log_prob, student_prob)

                loss = kl_loss * (self.temperature ** 2)
            elif self.distillation_type in ['hard_mix', 'soft_fwd_mix', 'soft_rev_mix']:
                if labels is None:
                    raise ValueError("Labels must be provided for mix distillation.")
                
                # --- Ground truth CE loss ---
                ce_loss_fct = nn.CrossEntropyLoss()
                ce_loss = ce_loss_fct(student_logits.view(-1, self.model.config.num_labels), labels.view(-1))

                # --- Distillation KD loss ---
                if self.distillation_type == 'hard_mix':
                    teacher_preds = torch.argmax(teacher_logits, dim=-1)
                    kd_loss = ce_loss_fct(student_logits.view(-1, self.model.config.num_labels), teacher_preds.view(-1))
                elif self.distillation_type == 'soft_fwd_mix':
                    student_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
                    teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
                    kd_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (self.temperature ** 2)
                elif self.distillation_type == 'soft_rev_mix':
                    teacher_log_prob = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                    student_prob = F.softmax(student_logits / self.temperature, dim=-1)
                    kd_loss = F.kl_div(teacher_log_prob, student_prob, reduction="batchmean") * (self.temperature ** 2)

                # --- Combine (0.5 * CE + 0.5 * KD) ---
                loss = 0.5 * ce_loss + 0.5 * kd_loss

        else:
            # This case implies supervised training but distillation_type was not 'supervised'
            # And no teacher model was provided. This shouldn't happen if train_one_student is called correctly.
             if self.distillation_type != 'supervised':
                raise ValueError(f"Teacher model is None, but distillation_type is {self.distillation_type}. Expected 'supervised'.")
             # If it IS 'supervised' but loss is still None here, it's an issue in the logic above.
             if loss is None: # Should have been handled by supervised blocks
                raise ValueError(f"Loss computation failed for supervised mode: {self.distillation_type}")
        if loss is None:
            raise ValueError(f"Loss not computed for distillation_type: {self.distillation_type}")

        return (loss, student_outputs) if return_outputs else loss


def train_teacher_model(model, train_dataset, eval_dataset, task, model_name_suffix, tokenizer, args):
    """Train a teacher model and return checkpoint path."""
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"teacher_{args.task}_{args.train_subset_ratio}_{model_name_suffix}_seed{args.seed}"
    )

    if is_valid_checkpoint(checkpoint_path):
        logger.info(f"Valid checkpoint already exists for teacher model: {checkpoint_path}. Skipping training.")
        return checkpoint_path
    
    experiment_name = f"{args.task}_{args.train_subset_ratio}_{model_name_suffix}_{'contaminated' if 'contaminated' in model_name_suffix else 'clean'}_seed{args.seed}"
    logger.info(f"Starting training for teacher model: {experiment_name}")

    temp_trainer_output_dir = os.path.join(OUTPUT_DIR, 'temp_teacher_training', experiment_name)
    if os.path.exists(temp_trainer_output_dir):
        logger.info(f"Removing existing temporary trainer output directory: {temp_trainer_output_dir}")
        shutil.rmtree(temp_trainer_output_dir)
    os.makedirs(temp_trainer_output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, 'temp_teacher_training', experiment_name),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        learning_rate=LEARNING_RATE,
        save_strategy='epoch',
        save_total_limit=1,
        warmup_ratio=0.06,
        gradient_accumulation_steps=2,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs_teacher', experiment_name),
        logging_steps=100,
        seed=args.seed,
        report_to="none",
        eval_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=None
    )
    
    logger.info("Training teacher model...")
    trainer.train()
    
    # Save both model and tokenizer
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)  # Explicitly save tokenizer
    logger.info(f"Saved teacher model and tokenizer to: {checkpoint_path}")
    
    if os.path.exists(temp_trainer_output_dir):
        logger.info(f"Removing temporary trainer output directory: {temp_trainer_output_dir}")
        shutil.rmtree(temp_trainer_output_dir)

    return checkpoint_path


def save_experiment_results(experiment_config, model_paths):
    """Save experimental config and model paths to a JSON file."""
    results = {
        'experiment_config': experiment_config,
        'model_checkpoint_paths': model_paths,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Sanitize model names for filename
    teacher_model_filename = experiment_config['teacher_model'].replace('/', '-')
    student_model_filename = experiment_config['student_model'].replace('/', '-')
    
    filename = f"{experiment_config['dataset']}_{experiment_config['train_subset_ratio']}_{teacher_model_filename}_to_{student_model_filename}_{experiment_config['contamination_mode']}_seed{experiment_config['seed']}.json"
    filepath = os.path.join(METRICS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved experiment results summary to: {filepath}")
    return filepath


def train_all_students(clean_teacher_ckpt_path, dirty_teacher_ckpt_path, base_student_model_name,
                       clean_train_data, contaminated_train_data,
                       task, teacher_tokenizer, student_tokenizer, args):
    """
    Train all student models as per the specification.
    Returns a dictionary of model_tag: checkpoint_path.
    """
    student_model_paths = {} # To store final checkpoint paths for each student

    # Base output dir for all students of this run
    # Sanitize model names for directory path
    teacher_model_name_short_for_path = args.teacher_model.split('/')[-1].replace('-', '_')
    student_model_name_short_for_path = base_student_model_name.split('/')[-1].replace('-', '_')
    
    students_base_output_dir_name = f"{args.task}_{args.train_subset_ratio}_S_{student_model_name_short_for_path}_T_{teacher_model_name_short_for_path}_seed{args.seed}"
    students_base_output_dir = os.path.join(CHECKPOINT_DIR, "students", students_base_output_dir_name)
    os.makedirs(students_base_output_dir, exist_ok=True)

    logger.info(f"Loading teacher models for student distillation (if applicable)...")
    
    # Helper to load teacher models
    def load_teacher(ckpt_path):
        if not ckpt_path: return None
        if not is_valid_checkpoint(ckpt_path):
            logger.error(f"Teacher checkpoint path {ckpt_path} is not valid or does not exist. Cannot load teacher.")
            return None # Or raise an error
        # For classification, num_labels is needed if not in config, but from_pretrained usually handles it.
        return AutoModelForSequenceClassification.from_pretrained(ckpt_path)

    clean_teacher_model = load_teacher(clean_teacher_ckpt_path)
    dirty_teacher_model = load_teacher(dirty_teacher_ckpt_path)

    dataset_info = get_dataset_info(task)

    dual_data_collator = DualTokenizerDataCollator(teacher_tokenizer, student_tokenizer)

    def train_one_student(distillation_type_full_tag, teacher_model_for_distill, current_train_data, 
                      student_distill_type_arg, student_descriptive_name_part):
        """Trains one student model, saves it, and returns its checkpoint path and logs."""
        
        student_model_short = base_student_model_name.split('/')[-1]
        full_student_tag = f"{args.task}_{args.train_subset_ratio}_{student_model_short}_{student_descriptive_name_part}_seed{args.seed}"
        student_specific_out_dir = os.path.join(students_base_output_dir, full_student_tag)
        
        if is_valid_checkpoint(student_specific_out_dir):
            logger.info(f"Valid checkpoint already exists for student model: {student_specific_out_dir}. Skipping training.")
            student_model_paths[full_student_tag] = student_specific_out_dir
            return student_specific_out_dir
        
        logger.info(f"Starting training for student model: {full_student_tag} with strategy: {student_distill_type_arg}")
        
        if "distilled" in student_descriptive_name_part and teacher_model_for_distill is None:
            logger.error(f"Teacher model is required for distillation type {student_distill_type_arg} but was not loaded. Skipping student {full_student_tag}.")
            student_model_paths[full_student_tag] = f"SKIPPED_DUE_TO_MISSING_TEACHER_{student_specific_out_dir}"
            return student_model_paths[full_student_tag]

        num_labels = dataset_info['num_labels']
        model = AutoModelForSequenceClassification.from_pretrained(base_student_model_name, num_labels=num_labels)

        student_vocab_size = len(student_tokenizer)
        if student_vocab_size != model.config.vocab_size:
            logger.info(
                f"Resizing student embeddings from {model.config.vocab_size} "
                f"to {student_vocab_size} to match student tokenizer vocabulary."
            )
            model.resize_token_embeddings(student_vocab_size)
        
        os.makedirs(student_specific_out_dir, exist_ok=True)

        temp_student_trainer_output_dir = os.path.join(OUTPUT_DIR, 'temp_student_training', full_student_tag)
        if os.path.exists(temp_student_trainer_output_dir):
            logger.info(f"Removing existing temporary student trainer output directory: {temp_student_trainer_output_dir}")
            shutil.rmtree(temp_student_trainer_output_dir)
        os.makedirs(temp_student_trainer_output_dir, exist_ok=True)

        training_args_student = TrainingArguments(
            output_dir=student_specific_out_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE_TRAIN_STUDENT,
            learning_rate=LEARNING_RATE,
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="epoch",
            seed=args.seed,
            report_to="none",
            eval_strategy="no",
            do_eval=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            gradient_accumulation_steps=2,
            warmup_ratio=0.06,
            remove_unused_columns=False,
        )
        

        trainer_student = MultiDistillationTrainer(
            teacher_model=teacher_model_for_distill,
            distillation_type=student_distill_type_arg,
            temperature=2.0,
            tokenizer=student_tokenizer,
            model=model,
            args=training_args_student,
            train_dataset=current_train_data,
            data_collator=dual_data_collator,
            callbacks=[]
        )
        
        logger.info(f"Training student model {full_student_tag} started...")
        trainer_student.train()
        logger.info(f"Training student model {full_student_tag} finished.")
        
        # Save model AND tokenizer
        trainer_student.save_model(student_specific_out_dir)
        student_tokenizer.save_pretrained(student_specific_out_dir)  # Explicitly save tokenizer
        logger.info(f"Student model and tokenizer {full_student_tag} saved to: {student_specific_out_dir}")

        if os.path.exists(temp_student_trainer_output_dir):
            logger.info(f"Removing temporary student trainer output directory: {temp_student_trainer_output_dir}")
            shutil.rmtree(temp_student_trainer_output_dir)
        
        student_model_paths[full_student_tag] = student_specific_out_dir
        
        return student_specific_out_dir


    # --- Train Supervised Students (Clean and Dirty) ---
    supervised_distill_type = 'supervised'

    logger.info("Starting training of Clean Student (supervised on clean data)...")
    clean_stud_desc_name = f"supervised_on_clean_data"
    train_one_student("clean_student_supervised_tag", None, clean_train_data, supervised_distill_type, clean_stud_desc_name)

    logger.info("Starting training of Dirty Student (supervised on contaminated data)...")
    dirty_stud_desc_name = f"supervised_on_contaminated_data"
    train_one_student("dirty_student_supervised_tag", None, contaminated_train_data, supervised_distill_type, dirty_stud_desc_name)

    # --- Train Distilled Students ---
    distillation_modes_cls = ["hard", "soft_fwd", "soft_rev", "hard_mix", "soft_fwd_mix", "soft_rev_mix"]
    if args.distillation_strategies:
        requested_modes = [mode.strip() for mode in args.distillation_strategies.split(',') if mode.strip()]
        invalid = [mode for mode in requested_modes if mode not in distillation_modes_cls]
        if invalid:
            raise ValueError(f"Unknown distillation strategies requested: {invalid}. Available: {distillation_modes_cls}")
        distillation_modes_cls = requested_modes
        logger.info(f"Restricting classification distillation strategies to: {', '.join(requested_modes)}")

    if not distillation_modes_cls:
        logger.info("No classification distillation strategies selected; skipping distillation student training.")
    else:
        for mode in distillation_modes_cls:
            # From Clean Teacher
            desc_name_ct = f"distilled_clean_teacher_{mode}"
            train_one_student(f"distilled_clean_teacher_{mode}_tag", clean_teacher_model, clean_train_data, mode, desc_name_ct)

            # From Dirty Teacher
            desc_name_dt = f"distilled_dirty_teacher_{mode}"
            train_one_student(f"distilled_dirty_teacher_{mode}_tag", dirty_teacher_model, clean_train_data, mode, desc_name_dt)

    return student_model_paths


class DualTokenizerDataCollator:
    """Pads teacher and student inputs separately for dual-tokenizer distillation."""
    def __init__(self, teacher_tokenizer, student_tokenizer, padding='longest'):
        self.teacher_collator = DataCollatorWithPadding(teacher_tokenizer, padding=padding)
        self.student_collator = DataCollatorWithPadding(student_tokenizer, padding=padding)

    def __call__(self, features):
        teacher_features = []
        student_features = []
        labels = []
        for feature in features:
            teacher_feat = {}
            if 'teacher_input_ids' in feature and feature['teacher_input_ids'] is not None:
                teacher_feat['input_ids'] = feature['teacher_input_ids']
            if 'teacher_attention_mask' in feature and feature['teacher_attention_mask'] is not None:
                teacher_feat['attention_mask'] = feature['teacher_attention_mask']
            if 'teacher_token_type_ids' in feature and feature['teacher_token_type_ids'] is not None:
                teacher_feat['token_type_ids'] = feature['teacher_token_type_ids']
            teacher_features.append(teacher_feat)

            student_feat = {}
            if 'student_input_ids' in feature and feature['student_input_ids'] is not None:
                student_feat['input_ids'] = feature['student_input_ids']
            if 'student_attention_mask' in feature and feature['student_attention_mask'] is not None:
                student_feat['attention_mask'] = feature['student_attention_mask']
            if 'student_token_type_ids' in feature and feature['student_token_type_ids'] is not None:
                student_feat['token_type_ids'] = feature['student_token_type_ids']
            student_features.append(student_feat)

            labels.append(feature['labels'])

        teacher_has_inputs = any('input_ids' in feat and len(feat['input_ids']) > 0 for feat in teacher_features)
        teacher_batch = None
        if teacher_has_inputs:
            teacher_batch = self.teacher_collator(teacher_features)
        student_batch = self.student_collator(student_features)

        batch = {
            'student_input_ids': student_batch['input_ids'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        if teacher_batch is not None:
            batch['teacher_input_ids'] = teacher_batch['input_ids']
            if 'attention_mask' in teacher_batch:
                batch['teacher_attention_mask'] = teacher_batch['attention_mask']
            if 'token_type_ids' in teacher_batch:
                batch['teacher_token_type_ids'] = teacher_batch['token_type_ids']
        if 'attention_mask' in student_batch:
            batch['student_attention_mask'] = student_batch['attention_mask']
        if 'token_type_ids' in student_batch:
            batch['student_token_type_ids'] = student_batch['token_type_ids']
        return batch

# -----------------------------------------------------------------------------------
# Main Experiment Runner
# -----------------------------------------------------------------------------------
def run_experiment(args):
    """Run the complete experiment pipeline."""
    logger.info(f"Starting experiment with config: {args}")
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model_identifier = resolve_model_name(args.teacher_model)
    student_model_identifier = resolve_model_name(args.student_model)
    if teacher_model_identifier != args.teacher_model:
        logger.info(f"Resolved teacher model alias '{args.teacher_model}' to '{teacher_model_identifier}'")
    if student_model_identifier != args.student_model:
        logger.info(f"Resolved student model alias '{args.student_model}' to '{student_model_identifier}'")

    # --- Tokenizer Setup ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_identifier)
    except OSError:
        logger.warning(
            f"Could not load tokenizer '{teacher_model_identifier}' "
            f"(resolved from '{args.teacher_model}'). Trying as local path."
        )
        tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(teacher_model_identifier))

    teacher_model_name_for_checks = teacher_model_identifier.lower()
    if "gpt" in teacher_model_name_for_checks or "wen" in teacher_model_name_for_checks or "llama" in teacher_model_name_for_checks:
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.pad_token = '[PAD]'
                logger.info(f"Added [PAD] token for teacher tokenizer {teacher_model_identifier}")
            else:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Setting pad_token to eos_token for {teacher_model_identifier} tokenizer")
        if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # --- Dataset Loading and Analysis Stage ---
    logger.info(f"Loading raw {args.task} dataset for analysis...")
    raw_datasets, dataset_info = load_raw_dataset(args.task)
    train_split_name = dataset_info['train_split']
    test_split_name = dataset_info['test_split']
        
    logger.info("Analyzing token counts to determine optimal max_length...")
    # We run a preliminary analysis with a high max_length to get the real distribution
    # The max_length argument here only acts as a reference for truncation calculation, not for the percentile logic.
    pre_analysis_stats = analyze_tokenization_stats(raw_datasets[train_split_name], tokenizer, 2048, dataset_info)
    p90_token_count = pre_analysis_stats['token_count_distribution']['p90']

    if p90_token_count <= 128:
        dynamic_max_length = 128
    else:
        dynamic_max_length = 512

    logger.info(f"90th percentile token count is {p90_token_count}. Setting dynamic_max_length to {dynamic_max_length}.")
    # Set MAX_LENGTH to this dynamic value for the rest of the script
    MAX_LENGTH = dynamic_max_length

    # --- End feature distillation section ---

    # --- Perform the statistical analysis ---
    train_stats = analyze_tokenization_stats(raw_datasets[train_split_name], tokenizer, MAX_LENGTH, dataset_info)
    test_stats = analyze_tokenization_stats(raw_datasets[test_split_name], tokenizer, MAX_LENGTH, dataset_info)

    # --- Pretty-print the stats to the console for immediate review ---
    logger.info("--- Tokenization Statistics (Train Set) ---")
    print(json.dumps(train_stats, indent=4))
    logger.info("--- Tokenization Statistics (Test Set) ---")
    print(json.dumps(test_stats, indent=4))

    # --- Dataset Loading and Preprocessing ---
    logger.info(f"Loading and preprocessing {args.task} dataset...")
    dataset_info_loaded = dataset_info
    full_train_dataset = raw_datasets['train']
    test_dataset = raw_datasets['test']

    # Store original sizes
    original_train_size = len(full_train_dataset)
    test_size = len(test_dataset)
    
    # --- Training Data Subsetting ---
    # --- Training Data Subsetting ---
    if args.train_subset_ratio and args.train_subset_ratio < 1.0:
        # Check if the task is a classification task with labels
        label_field = dataset_info_loaded.get('label_field', 'label')
        if label_field in full_train_dataset.features:
            logger.info(f"Performing stratified sampling for subset ratio: {args.train_subset_ratio}")
            
            # Use train_test_split for stratified sampling.
            # It returns a DatasetDict, we'll take the 'train' part.
            stratified_split = full_train_dataset.train_test_split(
                train_size=args.train_subset_ratio,
                stratify_by_column=label_field,
                seed=args.seed               # Ensure reproducibility
            )
            train_dataset = stratified_split['train']
            
            logger.info(f"Using a stratified subset of training data: {len(train_dataset)} samples.")

        else:
            # For datasets without a label column, fall back to random sampling.
            logger.info(f"Performing random sampling for subset ratio: {args.train_subset_ratio} (no label column found).")
            train_subset_size = int(len(full_train_dataset) * args.train_subset_ratio)
            train_subset_size = max(1, train_subset_size)
            train_dataset = full_train_dataset.select(range(train_subset_size))
            logger.info(f"Using a random subset of training data: {len(train_dataset)} samples.")

        # Update dataset info with ACTUAL sizes
        dataset_info_loaded['train_subset_ratio'] = args.train_subset_ratio
        dataset_info_loaded['original_train_size'] = original_train_size
        dataset_info_loaded['train_size'] = len(train_dataset)
        dataset_info_loaded['test_size'] = test_size
        dataset_info_loaded['ab_ratio'] = len(train_dataset) / test_size if test_size > 0 else 0
    else:
        # This part for using the full dataset remains the same
        train_dataset = full_train_dataset
        dataset_info_loaded['train_subset_ratio'] = 1.0
        dataset_info_loaded['original_train_size'] = original_train_size
        dataset_info_loaded['train_size'] = original_train_size
        dataset_info_loaded['test_size'] = test_size
        dataset_info_loaded['ab_ratio'] = original_train_size / test_size if test_size > 0 else 0

    
    logger.info(f"Final dataset sizes - Train: {len(train_dataset)} (original: {original_train_size}), Test: {test_size}")
    logger.info(f"Actual A/B ratio: {len(train_dataset) / test_size:.2f}")

    # --- Contamination Data Creation ---
    # This will use the SUBSETTED train_dataset, not the full one
    contaminated_train_data, contamination_indices = create_contaminated_train_data(
        train_dataset,  # Uses the subsetted training data
        test_dataset, 
        CONTAMINATION_RATIO, 
        args.contamination_mode, 
        args.seed, 
        dataset_info_loaded
    )
    logger.info("Contaminated training data created.")
    
    contamination_info_to_save = {
        'contamination_ratio_on_test_samples': CONTAMINATION_RATIO, # Ratio of test set used for contamination
        'contamination_mode': args.contamination_mode,
        'num_contaminated_samples_added_or_replaced': len(contamination_indices),
        'seed': args.seed,
        # 'contamination_indices_in_test_set': contamination_indices, # Can be very long
    }
    contamination_filename_suffix = f"{args.task}_{args.train_subset_ratio}_contam_mode_{args.contamination_mode}_seed{args.seed}"
    contamination_file = os.path.join(EXPERIMENT_DIR, f"contamination_info_{contamination_filename_suffix}.json")
    with open(contamination_file, 'w') as f:
        json.dump(contamination_info_to_save, f, indent=4)
    logger.info(f"Saved contamination info to: {contamination_file}")

    teacher_train_dataset = tokenize_dataset_for_model(train_dataset, tokenizer, MAX_LENGTH, dataset_info_loaded, args.task)
    teacher_contaminated_train_dataset = tokenize_dataset_for_model(contaminated_train_data, tokenizer, MAX_LENGTH, dataset_info_loaded, args.task)
    teacher_test_dataset = tokenize_dataset_for_model(test_dataset, tokenizer, MAX_LENGTH, dataset_info_loaded, args.task)

    # --- Teacher Model Training ---
    teacher_model_name_short = args.teacher_model.split('/')[-1]

    def load_and_configure_model(model_name_or_path, num_labels_for_cls=None):
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels_for_cls)
        if tokenizer.pad_token_id is not None:
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id
                logger.info(f"Set pad_token_id on model config for {model_name_or_path}")
        return model

    logger.info("Preparing Clean Teacher model...")
    clean_teacher = load_and_configure_model(teacher_model_identifier, dataset_info_loaded.get('num_labels'))
    
    untrained_teacher_path = os.path.join(CHECKPOINT_DIR, f"teacher_untrained_{args.task}_{args.train_subset_ratio}_{teacher_model_name_short}_seed{args.seed}")
    # clean_teacher.save_pretrained(untrained_teacher_path)
    # logger.info(f"Saved untrained teacher model (conceptual copy) structure to: {untrained_teacher_path}")

    logger.info("Training Clean Teacher model...")
    clean_teacher_ckpt_path = train_teacher_model(
        clean_teacher, teacher_train_dataset, teacher_test_dataset, args.task, 
        f"{teacher_model_name_short}_clean", tokenizer, args
    )
    # Free up memory if possible
    del clean_teacher 
    if torch.cuda.is_available(): torch.cuda.empty_cache()


    logger.info("Preparing Contaminated Teacher model...")
    dirty_teacher = load_and_configure_model(teacher_model_identifier, dataset_info_loaded.get('num_labels'))

    logger.info("Training Contaminated Teacher model...")
    dirty_teacher_ckpt_path = train_teacher_model(
        dirty_teacher, teacher_contaminated_train_dataset, teacher_test_dataset, args.task, 
        f"{teacher_model_name_short}_contaminated", tokenizer, args
    )
    del dirty_teacher
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_identifier)
    if student_tokenizer.pad_token is None:
        if student_tokenizer.eos_token is not None:
            student_tokenizer.pad_token = student_tokenizer.eos_token
            logger.info(f"Setting pad_token to eos_token for student tokenizer {student_model_identifier}")
        else:
            student_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            student_tokenizer.pad_token = '[PAD]'
            logger.info(f"Added [PAD] token for student tokenizer {student_model_identifier}")
    if student_tokenizer.pad_token_id is None and student_tokenizer.pad_token is not None:
        student_tokenizer.pad_token_id = student_tokenizer.convert_tokens_to_ids(student_tokenizer.pad_token)

    student_clean_dual_dataset = tokenize_dataset_dual(
        train_dataset, tokenizer, student_tokenizer, MAX_LENGTH, dataset_info_loaded, args.task
    )
    student_contaminated_dual_dataset = tokenize_dataset_dual(
        contaminated_train_data, tokenizer, student_tokenizer, MAX_LENGTH, dataset_info_loaded, args.task
    )

    # --- Student Model Training ---
    logger.info("Starting training of all student models...")
    # Pass tokenizer to train_all_students
    student_model_paths = train_all_students(
        clean_teacher_ckpt_path,
        dirty_teacher_ckpt_path,
        student_model_identifier,
        student_clean_dual_dataset,
        student_contaminated_dual_dataset,
        args.task,
        tokenizer,
        student_tokenizer,
        args
    )

    # --- Save Experiment Summary (Paths and Config) ---
    experiment_config_summary = {
        'dataset': args.task,
        'dataset_details': dataset_info_loaded, # Includes splits, actual sizes, subset ratio
        'teacher_model': args.teacher_model,
        'teacher_model_resolved': teacher_model_identifier,
        'student_model': args.student_model,
        'student_model_resolved': student_model_identifier,
        'contamination_info_file': contamination_file,
        'seed': args.seed,
        'max_seq_length': MAX_LENGTH,
        'num_train_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'train_batch_size': BATCH_SIZE_TRAIN,
        'train_batch_size_student': BATCH_SIZE_TRAIN_STUDENT,
        # Add any other relevant args
        'train_subset_ratio': args.train_subset_ratio, 
        'contamination_mode': args.contamination_mode
    }

    experiment_config_summary['tokenization_statistics'] = {
        'train_set': train_stats,
        'test_set': test_stats
    }

    # Consolidate all model paths
    all_model_checkpoint_paths = {
        'clean_teacher_checkpoint': clean_teacher_ckpt_path,
        'dirty_teacher_checkpoint': dirty_teacher_ckpt_path,
        **student_model_paths # student_model_paths is already a dict of {tag: path}
    }

    results_summary_filepath = save_experiment_results(experiment_config_summary, all_model_checkpoint_paths)
    logger.info(f"Experiment completed successfully! Summary saved to: {results_summary_filepath}")
    return results_summary_filepath


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment for data contamination in knowledge distillation")
    parser.add_argument('--task', type=str, required=True,
                        choices=['imdb', 'snli', 'agnews', 'emotion', 'banking77', 'tweet_sentiment', 'rotten_tomatoes', '20newsgroups'],
                        help="Dataset to use for the experiment")
    parser.add_argument('--teacher_model', type=str, required=True,
                        help="Teacher Hugging Face model name or path (e.g., bert-large-uncased, gpt2)")
    parser.add_argument('--student_model', type=str, required=True,
                        help="Student Hugging Face model name or path (e.g., bert-base-uncased, distilbert-base-uncased, gpt2)")
    parser.add_argument('--contamination_mode', type=str, required=True,
                        choices=['add', 'replace'],
                        help="How to contaminate training data (add or replace)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--train_subset_ratio', type=float, default=1.0,
                        help="Ratio of training data to use (e.g., 0.1 for 10%%). Use 1.0 for full dataset.")
    parser.add_argument('--distillation_strategies', type=str, default=None,
                        help="Optional comma-separated list of classification distillation strategies to train (e.g., 'soft_fwd').")
    
    args = parser.parse_args()

    # Validate subset ratio
    if not (0.0 < args.train_subset_ratio <= 1.0):
        raise ValueError("train_subset_ratio must be between 0.0 (exclusive) and 1.0 (inclusive).")

    try:
        result_file = run_experiment(args)
        logger.info(f"Final experiment summary JSON saved to: {result_file}")
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True) # Log traceback
        raise

if __name__ == "__main__":
    # Example usage:
    # python 1_training_experiment.py --task imdb --teacher_model bert-base-uncased --student_model distilbert-base-uncased --contamination_mode replace --seed 42 --train_subset_ratio 0.1
    main()





