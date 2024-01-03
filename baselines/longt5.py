import os
import torch
import pandas as pd
import numpy as np
import random
import nltk
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers.models.longt5.modeling_longt5 import LongT5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import logging as hf_logging

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Preprocess function for tokenizing and preparing the data
def preprocess_function(examples):
    inputs = examples["Paper_Body"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["News_Body"], max_length=max_target_length, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    print(result)
    
    return {k: round(v, 4) for k, v in result.items()}

# Generate predicted summaries
def generate_answers(batch):
    inputs_dict = tokenizer(
        batch["Paper_Body"], max_length=max_input_length, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    output_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=max_target_length, min_length=min_target_length,
        length_penalty=2.0, num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
    )
    batch["Prediction"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return batch

if __name__ == "__main__":
    # Set HF logging to info
    # hf_logging.set_verbosity_info()

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    torch.cuda.init()
    torch.cuda.empty_cache()

    # Set random seeds
    set_seed(2023)

    # Set the output file paths
    output_file = "output.json"

    # Set the model checkpoint
    model_checkpoint = "google/long-t5-tglobal-large"

    # Define the maximum input and target lengths
    max_input_length = 8192
    max_target_length = 1024
    min_target_length = 512


    # Load the dataset
    raw_datasets = load_dataset('json', data_files={'train': '../dataset/train.json', 'test': '../dataset/test.json'})

    # Load the metric for evaluation
    metric = load_metric("rouge")

    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Preprocess the dataset
    tokenized_data = raw_datasets.map(preprocess_function, batched=True)

    # Load the LED model
    model = LongT5ForConditionalGeneration.from_pretrained(model_checkpoint)

    # Set the training arguments
    batch_size = 1
    args = Seq2SeqTrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=15,
        predict_with_generate=True,
        warmup_ratio=0.1,
        optim="adafactor",
        load_best_model_at_end=True,
        group_by_length=True,
        fp16=True,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=16,
    )

    # Create the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Create the trainer
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Generate summaries for the test set
    model.eval()
    with torch.no_grad():
        result = raw_datasets['test'].map(generate_answers, batched=True, batch_size=1)
        result_df = pd.DataFrame(result)

    # Save the results to a CSV file
    result_df.to_json(output_file, orient="records", lines=True, indent=4)
