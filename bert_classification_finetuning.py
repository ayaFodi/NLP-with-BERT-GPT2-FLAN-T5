from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import  numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments
from transformers import Trainer
import sys

# Tokenizes text data for the model
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

# Computes accuracy for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Load the pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the dataset path from command line argument
#subset = load_from_disk("imdb_subset")
input_path = sys.argv[1]
try:
    subset = load_from_disk(input_path)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Rename column 'label' to 'labels' to match model input
subset = subset.rename_column("label", "labels")

try:
    # Tokenize the dataset
    tokenized_datasets = subset.map(tokenize_function, batched=True)
except Exception as e:
    print(f"Error during tokenization: {e}")
    sys.exit(1)

# Load the pretrained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Split the dataset into training and testing subsets
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2, seed=42)

# Define training arguments for the model
training_args = TrainingArguments(
    output_dir="../../Desktop/NLP-with-BERT-GPT2-FLAN-T5/results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=3,
    weight_decay=0.01
)

# Initialize the trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

try:
    # Train the model
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)

try:
    # Evaluate the model on the test dataset
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Accuracy: {results['eval_accuracy']:.2f}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    sys.exit(1)