# 🎓 NLP Homework 5 — Transformers, GPT2, and Prompt Engineering

This project was completed as part of the NLP course at the University of Haifa, Fall 2024.

## 📚 Overview

The assignment explores multiple aspects of modern NLP:
- Fine-tuning BERT for sentiment classification
- Fine-tuning GPT2 for text generation (positive/negative movie reviews)
- Prompt engineering using the FLAN-T5 model (zero-shot, few-shot, instruction-based)
- Analysis and comparison of outcomes
- Exploration of bias in LLMs like ChatGPT

## 🧠 Tasks and Structure

### 1. BERT Sentiment Classification
- File: `bert_classification_finetuning.py`
- Fine-tunes `bert-base-uncased` on IMDB reviews.
- Outputs classification accuracy.

### 2. GPT-2 Review Generation
- File: `gpt_generation_finetuning.py`
- Fine-tunes two GPT-2 models: one for positive, one for negative reviews.
- Generates 5 reviews per sentiment using a common prompt.

### 3. Prompt Engineering with FLAN-T5
- File: `flan_t5_prompt_engineering.py`
- Applies three prompting strategies to classify reviews:
  - Zero-shot
  - Few-shot (1 pos + 1 neg example)
  - Instruction-based

### 4. Results
- `generated_reviews.txt`: Generated outputs from GPT-2
- `flan_t5_imdb_results.txt`: Output of FLAN-T5 prompt strategies

### 5. Report
- `212001812_212266829_hw5_report.pdf`: Hebrew report summarizing decisions, results, and answers to analysis questions

## 💾 Dataset

A subset of 500 samples from the IMDB dataset was used.
- See folder: `imdb_subset/`

## 🧪 Running the Scripts

```bash
# Fine-tune and evaluate BERT
python bert_classification_finetuning.py imdb_subset/

# Generate reviews using GPT2
python gpt_generation_finetuning.py imdb_subset/ results/generated_reviews.txt saved_models_dir/

# Run FLAN-T5 prompt engineering
python flan_t5_prompt_engineering.py imdb_subset/ results/flan_t5_imdb_results.txt
```

## 📦 Dependencies

- `transformers`
- `datasets`
- `torch`
- `pandas`, `numpy`
- Python 3.8+

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## 🔍 Notes

- The project is GPU-compatible but runs on CPU with smaller data.
- All fine-tuning is done on small samples to save compute time.

---

## 📎 Report Language

The final report is in Hebrew and includes:
- Explanations of fine-tuning choices
- Prompt design justifications
- Accuracy comparisons
- Bias exploration with ChatGPT

---

## 👥 Authors

- Aya Fodi 
- Rema Hanna

NLP Course – University of Haifa, 2024
