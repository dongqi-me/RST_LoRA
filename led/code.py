import torch
import pandas as pd
model_checkpoint = "allenai/led-large-16384"

import os

from datasets import load_dataset, load_metric
raw_datasets = load_dataset("tomasg25/scientific_lay_summarisation", "elife")
metric = load_metric("rouge")

from rst_transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 16384
max_target_length = 512

import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(2023)

def preprocess_function(examples):
    inputs = examples["article"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

import transformers
from rst_transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from rst_transformers.models.led.modeling_led import LEDForConditionalGeneration
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, 
                         r=8, lora_alpha=32, lora_dropout=0.1, 
                         target_modules=["q_proj","v_proj"])


model = LEDForConditionalGeneration.from_pretrained(model_checkpoint)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


model.config.max_length = 512
model.config.min_length = 256
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-elife",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.1,
    save_total_limit=3,
    num_train_epochs=50,
    predict_with_generate=True,
    warmup_ratio=0.2,
    optim= "adafactor",
    load_best_model_at_end = True,
    group_by_length=True,
    fp16=False,
    lr_scheduler_type="cosine",
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

import torch
import pandas as pd
def generate_answers(batch):
    inputs_dict = tokenizer(
        batch["article"], max_length=max_input_length, padding=True,truncation=True, return_tensors="pt"
    )
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_target_length, num_beams=4)
    batch["predicted_summary"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return batch

model.eval()
with torch.no_grad():
    result = raw_datasets['test'].map(generate_answers, batched=True, batch_size=16)
    result_df = pd.DataFrame(result)

result_df.to_csv("result.csv")




