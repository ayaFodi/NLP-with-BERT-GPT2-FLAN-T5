import sys
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
from datasets import Dataset

# Load pretrained models and tokenizers
zero_shot_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
zero_shot_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

few_shot_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
few_shot_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

Instruction_based_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
Instruction_based_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Get dataset and output file paths from arguments
input_1_path = sys.argv[1]
input_2_path = sys.argv[2]

# Load the dataset from the given path
try:
    subset = load_from_disk(input_1_path)
except Exception as e:
    print(f"Error loading dataset from '{input_1_path}': {e}")
    sys.exit(1)
#subset = load_from_disk("imdb_subset")

# Function to filter short reviews
def is_short_and_labeled_correctly(example, max_length=490):
    return len(example['text']) <= max_length

# Select 25 positive and 25 negative reviews
try:
    dataset_label_0 = subset.filter(lambda example: example['label'] == 0 and is_short_and_labeled_correctly(example)).select(range(25))
    dataset_label_1 = subset.filter(lambda example: example['label'] == 1 and is_short_and_labeled_correctly(example)).select(range(25))
except Exception as e:
    print(f"Error filtering datasets: {e}")
    sys.exit(1)

# Get example reviews for few-shot prompting
example_label_0 = dataset_label_0[-1]['text']
example_label_1 = dataset_label_1[-1]['text']

# Combine datasets
all_examples = [example for example in dataset_label_0] + [example for example in dataset_label_1]
combined_dataset = Dataset.from_dict({'text': [ex['text'] for ex in all_examples], 'labels': [ex['label'] for ex in all_examples]})

# Function to add prefix to text
def add_prefix(example,prefix ):
    example['text'] = prefix + example['text']
    return example

#-------------------------------------zero shot Prompting--------------------------------------------------------------------------------
zero_shot_prefix = "Request: Tell me if the movie review is positive or negative. The review: "
# Add the prefix to each text entry in the dataset
zero_shot_data = combined_dataset.map(lambda example: add_prefix(example, zero_shot_prefix))

# Tokenize the dataset for input to the model
tokenized_zero_shot_data = zero_shot_tokenizer(zero_shot_data['text'], return_tensors="pt", padding=True).input_ids

# Generate predictions using the zero-shot model
zero_shot_outputs = zero_shot_model.generate(tokenized_zero_shot_data)

# Decode the model's predictions into text labels
decoded_zero_shot_outputs = zero_shot_tokenizer.batch_decode(zero_shot_outputs, skip_special_tokens=True)

# Convert text labels to numeric values (1 = positive, 0 = negative)
numeric_predictions_zero_shot_outputs = [1 if pred == 'positive' else 0 for pred in decoded_zero_shot_outputs]

# Calculate accuracy by comparing predictions to actual labels
zero_shot_accuracy = accuracy_score(zero_shot_data['labels'], numeric_predictions_zero_shot_outputs)

#print(f'Zero-shot Prompting Accuracy: {zero_shot_accuracy:.2f}')

#-------------------------------------few shot Prompting--------------------------------------------------------------------------------

# Create a few-shot prompt with two labeled examples before the new review
few_shot_prefix = f"Tag the movie review if it is positive or negative: 1.'{example_label_0}':'Negative' 2. '{example_label_1}': 'Positive' , Now tag this review: "
few_shot_data = combined_dataset.map(lambda example: add_prefix(example, few_shot_prefix))

# Tokenize the few-shot data
tokenized_few_shot_data = few_shot_tokenizer(few_shot_data['text'], return_tensors="pt", padding=True).input_ids

# Generate predictions using the few-shot model
few_shot_outputs = few_shot_model.generate(tokenized_few_shot_data)

# Decode the model's predictions
decoded_few_shot_outputs = few_shot_tokenizer.batch_decode(few_shot_outputs, skip_special_tokens=True)

# Convert text labels to numeric values
numeric_predictions_few_shot_outputs = [1 if pred == 'positive' else 0 for pred in decoded_few_shot_outputs]

# Compute accuracy for the few-shot model
few_shot_accuracy = accuracy_score(few_shot_data['labels'], numeric_predictions_few_shot_outputs)

#print(f'Few-shot Prompting Accuracy: {few_shot_accuracy:.2f}')

#-------------------------------------Instruction-based Prompting--------------------------------------------------------------------------------
# Create a structured instruction-based prompt
Instruction_based_prefix = "You are an expert movie review classifier. Please classify this new movie review as positive or negative: "
Instruction_based_data = combined_dataset.map(lambda example: add_prefix(example, Instruction_based_prefix))

# Tokenize the instruction-based dataset
tokenized_Instruction_based_data = Instruction_based_tokenizer(Instruction_based_data['text'], return_tensors="pt", padding=True).input_ids

# Generate predictions using the instruction-based model
Instruction_based_outputs = Instruction_based_model.generate(tokenized_Instruction_based_data)

# Decode predictions into text
decoded_Instruction_based_outputs = Instruction_based_tokenizer.batch_decode(Instruction_based_outputs, skip_special_tokens=True)

# Convert predictions into numeric format
numeric_predictions_Instruction_based_outputs = [1 if pred == 'positive' else 0 for pred in decoded_Instruction_based_outputs]

# Compute accuracy for the instruction-based model
Instruction_based_accuracy = accuracy_score(Instruction_based_data['labels'], numeric_predictions_Instruction_based_outputs)
#print(f'Instruction_based Prompting Accuracy: {Instruction_based_accuracy:.2f}')

#-------------------------------------Write outputs--------------------------------------------------------------------------------

#output_file_name = "flan_t5_imdb_results.txt"
output_file_name = input_2_path

with open(output_file_name, "w", encoding="utf-8") as file:
    for i, (review_text, true_label, zero_shot_pred, few_shot_pred, instruction_based_pred) in enumerate(
        zip(combined_dataset['text'], combined_dataset['labels'], decoded_zero_shot_outputs, decoded_few_shot_outputs, decoded_Instruction_based_outputs), start=1):

        file.write(f"Review {i}: {review_text}\n")
        file.write(f"Review {i} true label: {'positive' if true_label == 1 else 'negative'}\n")
        file.write(f"Review {i} zero-shot: {zero_shot_pred}\n")
        file.write(f"Review {i} few-shot: {few_shot_pred}\n")
        file.write(f"Review {i} instruction-based: {instruction_based_pred}\n")