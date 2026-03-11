# Emotion-Aware Conversational AI for Mental Health Support

## A Resource-Constrained Fine-Tuning Study of a 7B LLM for Empathetic Dialogue

---

## Overview

This project documents the end-to-end design, development, and evaluation of an emotion-aware conversational AI system built using open-source large language models and publicly available emotional-support datasets.

The primary objective was to create a chatbot capable of generating **empathetic, context-aware, supportive, and safe responses** for users experiencing emotional distress — while operating under severe hardware constraints (Google Colab T4 GPU, limited VRAM, RAM caps, storage limits, and session timeouts).

Rather than focusing purely on model performance, this work emphasizes the **real engineering challenges of training large language models on constrained resources**, including dataset processing, memory optimization, runtime stability, and deployment feasibility.

This project reflects extensive experimentation, iterative problem-solving, and practical machine learning engineering under real-world limitations.

---

## Motivation

Generic conversational models often produce fluent text but fail in emotionally sensitive contexts. In mental-health-related interactions, responses must be:

- Empathetic
- Contextually relevant
- Supportive rather than generic
- Safe, especially in crisis situations

The goal of this project was to fine-tune a model specifically for emotional support dialogue while minimizing harmful or unsafe outputs.

---

## Base Model

An open-source ~7B parameter language model was selected as the foundation:

**Base Model:** `mistralai/Mistral-7B-v0.1`

This size offers strong conversational ability while remaining marginally trainable on a single consumer GPU using parameter-efficient techniques.

---

## Fine-Tuning Method

Training was performed using **QLoRA (Quantized Low-Rank Adaptation)**:

- 4-bit NF4 quantization of base model
- LoRA adapters for trainable parameters
- Frozen base model weights
- Reduced memory footprint
- Feasible training on Colab T4 GPU

This approach enables tuning large models without requiring high-end hardware.

---

## Datasets Used

Two public emotional-support datasets were merged.

### 1. EmpatheticDialogues (Facebook Research)

A large corpus of emotionally grounded conversations.

Key properties:

- Natural dialogue structure
- Emotion labels (e.g., sadness, anxiety, loneliness)
- High conversational quality

Fields used:

- `prompt` → user situation
- `utterance` → assistant response
- `context` → emotion category

Loaded rows: **76,673**

---

### 2. ESConv (Emotional Support Conversations)

Counseling-style multi-turn conversations.

Characteristics:

- Structured supportive dialogue
- Strategy-oriented responses
- Stored as JSON-encoded text

Processing required parsing nested dialogue turns and extracting valid user → system exchanges.

Loaded conversations: **910**

---

## Data Preparation Pipeline

A significant portion of the project involved building a robust preprocessing pipeline to unify heterogeneous datasets.

### Dataset Exploration

Initial inspection revealed:

- Different schemas
- Different storage formats (tabular vs JSON-encoded)
- Irrelevant metadata fields
- Inconsistent dialogue structures

---

### Conversation Pair Extraction

Each training sample was converted into:

```

User message → Assistant response

```

#### EmpatheticDialogues

- User input: `prompt`
- Assistant reply: `utterance`
- Emotion labels retained

#### ESConv

- Parsed JSON dialogues
- Extracted valid consecutive `usr → sys` turns
- Removed incomplete exchanges

---

### Schema Unification

All samples were transformed into a consistent format:

```

User: <message>
Assistant: <reply>

```

---

### Cleaning and Normalization

Operations included:

- Replacing placeholder tokens (e.g., `_comma_`)
- Normalizing whitespace
- Removing malformed samples
- Dropping empty or extremely short responses

---

### Safety Pre-Filtering

Potentially harmful content was removed before training, including:

- Self-harm encouragement
- Toxic or abusive language
- Dangerous advice

---

### Deduplication

Duplicate conversation pairs were removed to reduce memorization bias and improve generalization.

---

### Instruction Formatting

Final supervised training format:

```

### Instruction:

The user is feeling <emotion>. Provide emotional support.

### User:

User: <message>

### Assistant:

<response>
```

Emotion labels were incorporated when available to improve emotional awareness.

---

## Dataset Statistics

After preprocessing:

* Extracted conversation pairs: ~86,864
* After filtering and deduplication: ~86,563

---

## Compute Constraints & Training Journey

The dominant challenge of this project was not model design but **hardware limitations**.

Training large language models on free cloud resources required extensive experimentation and compromise.

### Initial Full-Dataset Attempt

Training on the full dataset (~86k samples):

* Estimated runtime: ~36 hours
* High VRAM pressure
* Risk of session termination
* Not reproducible on Colab

---

### Progressive Dataset Reduction

To achieve a stable training run, the dataset was gradually reduced:

| Stage          | Dataset Size | Approx Time | Outcome            |
| -------------- | ------------ | ----------- | ------------------ |
| Full dataset   | ~86k         | ~36 hours   | Not feasible       |
| Reduction 1    | ~30k         | ~20 hours   | Still unstable     |
| Reduction 2    | ~15k         | High        | Memory/time issues |
| Final training | ~12k         | Completed   | Stable             |

For reproducible notebook execution:

* Training set: **6,000 samples**
* Validation set: **800 samples**

---

## Memory Optimization Techniques

Multiple strategies were required to prevent out-of-memory errors:

* QLoRA 4-bit quantization
* LoRA adapters
* Small batch size
* Gradient accumulation
* Reduced sequence length (256 tokens)
* Gradient checkpointing
* Explicit GPU memory clearing
* Careful runtime management

Despite these measures, repeated RAM and VRAM failures occurred during experimentation.

---

## Training Configuration

**Quantization**

* 4-bit NF4
* Double quantization
* Float16 compute

**LoRA**

* r = 16
* alpha = 32
* dropout = 0.05

**Training Arguments**

* Batch size: 2
* Gradient accumulation: 4
* Learning rate: 2e-4
* Epochs: 1
* Optimizer: paged_adamw_8bit
* Sequence length: 256

---

## Training Snapshot

Environment: Google Colab T4 GPU

* Global steps: 750
* Training loss: 1.2586
* Runtime: ~10,327 seconds (~2.87 hours)

Model artifacts were successfully saved and pushed to the Hugging Face Hub.

---

## Evaluation Methodology

Evaluation combined automatic metrics with human judgment.

### Automatic Metrics

* Training loss
* Limited validation metrics (memory constraints restricted full evaluation)

---

### Human Evaluation (Primary)

Thirty prompts were manually designed across six emotional categories:

* Mild sadness
* Anxiety
* Loneliness
* Failure / self-worth
* Ambiguous suicidal tone
* Direct self-harm statements

Each response was rated on:

* Empathy (1–5)
* Relevance (1–5)
* Helpfulness (1–5)
* Safety (Safe / Unsafe)

---

## Key Metrics

### Emotional Quality Score (EQS)

```
EQS = (Empathy + Relevance + Helpfulness) / 3
```

---

### Safety Precision

```
Safety Precision = Correct Crisis Overrides / Total Crisis Prompts
```

---

### Unsafe Response Rate

```
Unsafe Rate = Unsafe Responses / Total Prompts
```

---

## Observed Model Behavior

### Strengths

* Consistently empathetic tone
* Coherent conversational flow
* Supportive language
* Cautious responses to crisis signals

---

### Limitations

* Occasional generic reassurance
* Limited actionable advice
* Reduced nuance due to smaller training dataset
* Not suitable for clinical deployment

---

## Deployment Readiness

The model was prepared for:

* Standard text-generation inference pipelines
* Hugging Face Hub hosting
* Web demo deployment using Gradio / Spaces

---

## Key Learnings

1. Data quality outweighs raw dataset size in emotional tasks
2. Parameter-efficient tuning enables large-model experimentation on limited hardware
3. Compute constraints strongly influence engineering decisions
4. Iterative experimentation is essential for successful training
5. Safety evaluation is critical for mental-health-related AI systems

---

## Ethical Considerations

This system is **not a substitute for professional mental health care**.
Real-world deployment would require crisis escalation mechanisms, safeguards, and human oversight.

---

## Author

**Nihal Jaiswal**
AI/ML Practitioner — Machine Learning, NLP, and Generative AI
Focus: Conversational AI, LLM Fine-Tuning, Safe AI Systems



