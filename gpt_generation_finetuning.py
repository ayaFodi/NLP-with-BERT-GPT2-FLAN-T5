import os
import sys

import torch
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# Load pre-trained GPT-2 models and tokenizer
model_label_0 = GPT2LMHeadModel.from_pretrained("gpt2")
model_label_1 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_label_0 = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer_label_1 = GPT2Tokenizer.from_pretrained("gpt2")

# Function to tokenize dataset and rename 'label' column
def tokenize_reviews(dataset, tokenizer, max_length=150):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)

    # Tokenize and rename 'label' to 'labels'
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    return tokenized_dataset

# Get paths for dataset, output file, and model directory from command-line arguments
input_1_path = sys.argv[1]
input_2_path = sys.argv[2]
input_3_path = sys.argv[3]

# Load the dataset from the given path
try:
    subset = load_from_disk(input_1_path)
except Exception as e:
    print(f"Error loading dataset from '{input_1_path}': {e}")
    sys.exit(1)
#subset = load_from_disk("imdb_subset")

# Filter and split dataset by labels (0 and 1)
try:
    dataset_label_0 = subset.filter(lambda example: example['label'] == 0).select(range(100))
    dataset_label_1 = subset.filter(lambda example: example['label'] == 1).select(range(100))
except Exception as e:
    print(f"Error filtering datasets: {e}")
    sys.exit(1)

# Tokenize the dataset
try:
    tokenized_dataset_label_0  = tokenize_reviews(dataset_label_0, tokenizer_label_0)
    tokenized_dataset_label_1 = tokenize_reviews(dataset_label_1, tokenizer_label_1)

except Exception as e:
    print(f"Error during tokenization: {e}")
    sys.exit(1)


# Compute average text lengths for both labels
text_lengths_label_0  = [len(example['text']) for example in tokenized_dataset_label_0]
average_length_label_0 = int(sum(text_lengths_label_0 ) / 100)
text_lengths_label_1  = [len(example['text']) for example in tokenized_dataset_label_1]
average_length_label_1 = int(sum(text_lengths_label_1 ) / 100)

# Define training arguments for both models
training_args_label_0 = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=3,
    weight_decay=0.01
)

training_args_label_1 = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=3,
    weight_decay=0.01
)

# Create trainers for both labels
trainer_label_0 = Trainer(
    model=model_label_0,
    args=training_args_label_0,
    train_dataset=tokenized_dataset_label_0,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_label_0, mlm=False),
)

trainer_label_1 = Trainer(
    model=model_label_1,
    args=training_args_label_1,
    train_dataset=tokenized_dataset_label_1,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_label_1, mlm=False),
)
directory_name = input_3_path
"""
# Train both models
try:
    trainer_label_0.train()
except Exception as e:
    print(f"Error training model for label 0: {e}")
    sys.exit(1)

try:
    trainer_label_1.train()
except Exception as e:
    print(f"Error training model for label 1: {e}")
    sys.exit(1)

# Create directory to save models
#directory_name = "saved_models_dir"
try:
    os.makedirs(directory_name, exist_ok=True)
except Exception as e:
    print(f"Error creating directory '{directory_name}': {e}")
    sys.exit(1)

# Save both models and tokenizer
trainer_label_0.model.save_pretrained(f"{directory_name}/model_label_0")
tokenizer_label_0.save_pretrained(f"{directory_name}/model_label_0")

trainer_label_1.model.save_pretrained(f"{directory_name}/model_label_1")
tokenizer_label_1.save_pretrained(f"{directory_name}/model_label_1")
"""
# Load saved models and tokenizer
try:
    model_label_0 = GPT2LMHeadModel.from_pretrained(f"{directory_name}/model_label_0")
    tokenizer_label_0 = GPT2Tokenizer.from_pretrained(f"{directory_name}/model_label_0")
except Exception as e:
    print(f"Error loading model_label_0 from '{directory_name}/model_label_0': {e}")
    sys.exit(1)

try:
    model_label_1 = GPT2LMHeadModel.from_pretrained(f"{directory_name}/model_label_1")
    tokenizer_label_1 = GPT2Tokenizer.from_pretrained(f"{directory_name}/model_label_1")
except Exception as e:
    print(f"Error loading model_label_1 from '{directory_name}/model_label_1': {e}")
    sys.exit(1)

# Generate text using both models
prompt = "My hobby is to"
input_ids = tokenizer_label_0.encode(prompt, return_tensors="pt")
attention_mask = input_ids.ne(tokenizer_label_0.pad_token_id)

with torch.no_grad():
    output_label_0 = model_label_0.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=average_length_label_0,
        temperature=1.2,
        top_k=15,
        top_p=0.6,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=5,
    )

input_ids = tokenizer_label_1.encode(prompt, return_tensors="pt")
attention_mask1 = input_ids.ne(tokenizer_label_1.pad_token_id)

with torch.no_grad():
    output_label_1 = model_label_1.generate(
        input_ids,
        attention_mask=attention_mask1,
        max_length=average_length_label_1,
        temperature=1.05,
        top_k=40,
        top_p=0.8,
        repetition_penalty=1.1,
        do_sample=True,
        num_return_sequences=5,
    )
# Decode generated texts
generated_texts_label_0 = [tokenizer_label_0.decode(generated, skip_special_tokens=True) for generated in output_label_0]
generated_texts_label_1 = [tokenizer_label_1.decode(generated, skip_special_tokens=True) for generated in output_label_1]

# Write generated texts to the output file
output_file_name = input_2_path
try:
    with open(output_file_name, "w", encoding="utf-8") as file:
        file.write("Reviews generated by positive model:\n")
        for idx, text in enumerate(generated_texts_label_1, 1):
            file.write(f"{idx}. {text}\n\n")

        file.write("Reviews generated by negative model:\n")
        for idx, text in enumerate(generated_texts_label_0, 1):
            file.write(f"{idx}. {text}\n\n")

except Exception as e:
    print(f"Error writing to file '{output_file_name}': {e}")
    sys.exit(1)