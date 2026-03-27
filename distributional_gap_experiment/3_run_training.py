#!/usr/bin/env python3
# 3_run_training.py

import os
import random
import numpy as np
import torch
import json
import argparse
import logging
from datetime import datetime
import shutil
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CONTAMINATION_RATIO = 1
NUM_EPOCHS = 10
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TRAIN_STUDENT = 128
LEARNING_RATE = 2e-5
OUTPUT_DIR = './results'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, 'experiments')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def is_valid_checkpoint(path):
    if not os.path.isdir(path): return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_model = os.path.exists(os.path.join(path, "pytorch_model.bin")) or os.path.exists(os.path.join(path, "model.safetensors"))
    return has_config and has_model

def create_contaminated_train_data(clean_train, test_dataset, contamination_ratio, mode, seed):
    num_contaminate = min(int(len(test_dataset) * contamination_ratio), len(test_dataset))
    random.seed(seed)
    contamination_indices = random.sample(range(len(test_dataset)), num_contaminate)
    contamination_data = test_dataset.select(contamination_indices)
    if 'labels' not in clean_train.features and 'label' in clean_train.features:
        clean_train = clean_train.rename_column('label', 'labels')
    if 'labels' not in contamination_data.features and 'label' in contamination_data.features:
        contamination_data = contamination_data.rename_column('label', 'labels')
    if mode == 'replace':
        num_replace = min(num_contaminate, len(clean_train))
        replace_indices = set(random.sample(range(len(clean_train)), num_replace))
        clean_filtered = clean_train.filter(lambda _, idx: idx not in replace_indices, with_indices=True)
        contaminated_train = concatenate_datasets([clean_filtered, contamination_data]).shuffle(seed=seed)
        logger.info(f"Replaced {num_replace} clean samples with contaminated samples. New size: {len(contaminated_train)}")
    else: # add mode
        contaminated_train = concatenate_datasets([clean_train, contamination_data]).shuffle(seed=seed)
        logger.info(f"Added {num_contaminate} contaminated samples. New size: {len(contaminated_train)}")
    return contaminated_train, contamination_indices

class MultiDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, distillation_type='supervised', temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model:
            self.teacher_model.eval().to(self.model.device)
        self.distillation_type = distillation_type
        self.temperature = temperature
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits; loss = None
        if self.distillation_type == 'supervised':
            loss = nn.CrossEntropyLoss()(student_logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif self.teacher_model:
            with torch.no_grad():
                teacher_logits = self.teacher_model(**inputs).logits
            if self.distillation_type == 'soft_fwd':
                soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
                log_soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
                loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        if loss is None: raise ValueError("Loss not computed")
        return (loss, student_outputs) if return_outputs else loss

def train_teacher_model(model, train_dataset, model_name_suffix, tokenizer, args):
    # Use experiment_tag and contamination_mode for unique checkpoint path
    if model_name_suffix == "contaminated":
        model_name_suffix = f"contaminated_{args.contamination_mode}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"teacher_{args.experiment_tag}_{model_name_suffix}_seed{args.seed}")

    if is_valid_checkpoint(checkpoint_path):
        logger.info(f"Valid checkpoint already exists for teacher: {checkpoint_path}. Skipping.")
        return checkpoint_path
    
    temp_dir = os.path.join(OUTPUT_DIR, 'temp_training', os.path.basename(checkpoint_path))
    training_args = TrainingArguments(output_dir=temp_dir, num_train_epochs=NUM_EPOCHS, per_device_train_batch_size=BATCH_SIZE_TRAIN, learning_rate=LEARNING_RATE, save_strategy='epoch', save_total_limit=1, seed=args.seed, report_to="none")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    logger.info(f"Saved teacher model to: {checkpoint_path}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    return checkpoint_path

def train_all_students(clean_teacher_ckpt_path, dirty_teacher_ckpt_path, base_student_model_name, clean_train_data, contaminated_train_data, num_labels, tokenizer, args):
    student_model_paths = {}
    clean_teacher_model = AutoModelForSequenceClassification.from_pretrained(clean_teacher_ckpt_path)
    dirty_teacher_model = AutoModelForSequenceClassification.from_pretrained(dirty_teacher_ckpt_path)

    def train_one_student(teacher, train_data, distill_type, name_part):
        student_model_short = base_student_model_name.split('/')[-1]
        # Use experiment_tag for a unique checkpoint path
        full_tag = f"{args.experiment_tag}_{student_model_short}_{name_part}_seed{args.seed}"
        out_dir = os.path.join(CHECKPOINT_DIR, "students", full_tag)

        if is_valid_checkpoint(out_dir):
            logger.info(f"Valid checkpoint exists for student: {full_tag}. Skipping.")
            student_model_paths[name_part] = out_dir; return
            
        logger.info(f"Training student: {full_tag} with strategy: {distill_type}")
        model = AutoModelForSequenceClassification.from_pretrained(base_student_model_name, num_labels=num_labels)
        temp_dir = os.path.join(OUTPUT_DIR, 'temp_training', full_tag)
        train_args = TrainingArguments(output_dir=temp_dir, num_train_epochs=NUM_EPOCHS, per_device_train_batch_size=BATCH_SIZE_TRAIN_STUDENT, learning_rate=LEARNING_RATE, save_strategy="epoch", save_total_limit=1, seed=args.seed, report_to="none")
        trainer = MultiDistillationTrainer(teacher_model=teacher, distillation_type=distill_type, model=model, args=train_args, train_dataset=train_data)
        trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        student_model_paths[name_part] = out_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    train_one_student(None, clean_train_data, 'supervised', "supervised_on_clean_data")
    train_one_student(None, contaminated_train_data, 'supervised', f"supervised_on_contaminated_data_{args.contamination_mode}")
    train_one_student(clean_teacher_model, clean_train_data, "soft_fwd", "student_kd_from_clean_soft")
    train_one_student(dirty_teacher_model, clean_train_data, "soft_fwd", f"student_kd_from_dirty_soft_{args.contamination_mode}")
    return student_model_paths

def save_experiment_results(experiment_config, model_paths, contamination_file):
    results = {'experiment_config': experiment_config, 'model_checkpoint_paths': model_paths, 'contamination_info_file': contamination_file, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    teacher_filename = experiment_config['teacher_model'].replace('/', '-')
    student_filename = experiment_config['student_model'].replace('/', '-')
    
    # Use experiment_tag for a unique summary filename
    filename = f"{experiment_config['experiment_tag']}_{teacher_filename}_to_{student_filename}_{experiment_config['contamination_mode']}_seed{experiment_config['seed']}.json"
    filepath = os.path.join(METRICS_DIR, filename)
    with open(filepath, 'w') as f: json.dump(results, f, indent=4)
    logger.info(f"Saved experiment summary to: {filepath}")
    return filepath

def run_experiment(args):
    logger.info(f"Starting experiment with config: {args}")
    set_seed(args.seed)

    logger.info(f"Loading pre-split dataset from: {args.dataset_path}")
    if not os.path.isdir(args.dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")
    
    dataset = load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True).rename_column("label", "labels")
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataset, test_dataset = tokenized_datasets['train'], tokenized_datasets['test']
    num_labels = train_dataset.features['labels'].num_classes
    
    contaminated_train_data, _ = create_contaminated_train_data(train_dataset, test_dataset, CONTAMINATION_RATIO, args.contamination_mode, args.seed)
    
    # Use experiment_tag for a unique contamination info filename
    contamination_filename = f"contamination_info_{args.experiment_tag}_{args.contamination_mode}_seed{args.seed}.json"
    contamination_file = os.path.join(EXPERIMENT_DIR, contamination_filename)
    with open(contamination_file, 'w') as f: json.dump({'mode': args.contamination_mode, 'seed': args.seed}, f)

    clean_teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
    clean_teacher_ckpt = train_teacher_model(clean_teacher, train_dataset, "clean", tokenizer, args)
    del clean_teacher; torch.cuda.empty_cache()

    dirty_teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
    dirty_teacher_ckpt = train_teacher_model(dirty_teacher, contaminated_train_data, "contaminated", tokenizer, args)
    del dirty_teacher; torch.cuda.empty_cache()

    student_paths = train_all_students(clean_teacher_ckpt, dirty_teacher_ckpt, args.student_model, train_dataset, contaminated_train_data, num_labels, tokenizer, args)
    
    all_paths = {'clean_teacher_checkpoint': clean_teacher_ckpt, 'dirty_teacher_checkpoint': dirty_teacher_ckpt, **student_paths}
    save_experiment_results(vars(args), all_paths, contamination_file)
    logger.info("Experiment completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment on a pre-split dataset.")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--experiment_tag', type=str, required=True, help="Unique identifier for the experiment run (e.g., 'emotion-fixedtest_by_similarity_level_1').")
    parser.add_argument('--teacher_model', type=str, required=True)
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--contamination_mode', type=str, required=True, choices=['add', 'replace'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, required=True, help="Task name (metadata for logging/compatibility).")
    parser.add_argument('--train_subset_ratio', type=float, default=1.0, help="Subset ratio metadata (kept for compatibility).")
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()

