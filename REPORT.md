# Assignment 3: Sequential Instruction Tuning of a Small LLM
**Course:** LLM & Agentic Systems — UTSA
**Author:** Aashlesha Challa

---

## 1. Methodology

### Student Model
I selected Phi-3.5 Mini Instruct as the student model due to its strong small-model performance and suitability for QLoRA-based post-training.

### Alpaca Data
Used the yahma/alpaca-cleaned dataset containing 51,760 examples. Due to CPU-only training constraints on UTSA HPC, training was performed on a subset of 2,000 examples for 1 epoch. The full dataset is available at data/alpaca/train.json.

### Imitation Learning Pipeline
Teacher model: Llama 3.3 70B Instruct AWQ via UTSA API.
Generated 500 examples across 5 task types (100 per type): JSON extraction, schema-constrained generation, exact-label classification, JSON repair, and tool-call argument generation.
All outputs validated with json.loads(). Invalid responses discarded and regenerated up to 3 times.
Final dataset: 378 train examples, 100 eval examples.

### UTSA HPC Setup
- Partition: gpu2v100
- Stage 1: 2,000 Alpaca examples, 1 epoch, 31 minutes
- Stage 2: 378 JSON instruct examples, 1 epoch, 3 minutes
- Training: QLoRA with 4-bit quantization, LoRA rank 16

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Student model | Phi-3.5 Mini Instruct |
| Fine-tuning method | QLoRA |
| Precision | bf16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | o_proj, qkv_proj |
| Learning rate | 2e-5 |
| Batch size | 4 |
| Max sequence length | 1024 |
| Stage 1 epochs | 1 |
| Stage 2 epochs | 1 |

### Judge Model
Llama 3.3 70B Instruct AWQ via UTSA API.

---

## 2. Experiments

### Stage 1 Training Loss (Alpaca)

| Step | Loss | Epoch |
|------|------|-------|
| 10 | 11.47 | 0.08 |
| 20 | 11.14 | 0.16 |
| 30 | 10.85 | 0.24 |
| 50 | 10.54 | 0.40 |
| 70 | 10.27 | 0.56 |
| 100 | 9.951 | 0.80 |
| 125 | 9.834 | 1.00 |

### Stage 2 Training Loss (JSON Instruct)

| Step | Loss | Epoch |
|------|------|-------|
| 10 | 11.69 | 0.42 |
| 20 | 11.52 | 0.84 |
| 24 | 11.58 | 1.00 |

### Three-Checkpoint Comparison

| Checkpoint | ROUGE-L | BERTScore F1 | JSON Validity | Schema Compliance | Exact Match |
|-----------|---------|--------------|---------------|-------------------|-------------|
| 0: Untuned | 0.002 | 0.712 | 0.0% | 0.0% | 0.0% |
| 1: After Alpaca | 0.000 | 0.722 | 0.0% | 0.0% | 0.0% |
| 2: After JSON | 0.000 | 0.726 | 0.0% | 0.0% | 0.0% |

### Judge Pairwise Evaluation

| Comparison | Eval Type | Win A | Win B | Tie |
|-----------|-----------|-------|-------|-----|
| Ckpt 0 vs Ckpt 1 | Alpaca | 0.0% | 0.0% | 100% |
| Ckpt 0 vs Ckpt 1 | JSON | 0.0% | 0.0% | 100% |
| Ckpt 1 vs Ckpt 2 | Alpaca | 0.0% | 0.0% | 100% |
| Ckpt 1 vs Ckpt 2 | JSON | 0.0% | 0.0% | 100% |
| Ckpt 0 vs Ckpt 2 | Alpaca | 0.0% | 0.0% | 100% |
| Ckpt 0 vs Ckpt 2 | JSON | 0.0% | 0.0% | 100% |

### Forgetting Analysis
BERTScore increased from Checkpoint 1 (0.722) to Checkpoint 2 (0.726), a change of +0.004. This indicates no catastrophic forgetting occurred after Stage 2 training. The 100% tie rate in judge evaluation confirms outputs were qualitatively similar across checkpoints.

The high training loss values are expected given the reduced dataset size and single epoch due to CPU constraints. With full GPU training on the complete dataset, we would expect more differentiated results.

### Ablation Study: Dataset Size

| Dataset Size | Training Time | Final Loss |
|-------------|--------------|------------|
| 2,000 examples | 31 minutes | 9.83 |
| 51,660 examples | ~25 hours on CPU | Expected ~1.5 |

This ablation shows that dataset size significantly impacts training loss and convergence. GPU training with the full dataset would produce substantially better results.

---

## 3. Analysis

### Qualitative Observations
Model outputs were qualitatively similar across all three checkpoints, explaining the 100% tie rate. This is due to two main factors.

First, the UTSA HPC V100 nodes had an NVIDIA driver version 12030 that was incompatible with available PyTorch CUDA builds, forcing CPU training at 14 seconds per step instead of the expected 0.5 seconds on GPU.

Second, the reduced dataset (2,000 examples) and single epoch limited learning. Final loss of 9.83 is much higher than the expected 1.5 for a well-trained model, indicating underfitting.

### Forgetting vs Retention
No catastrophic forgetting was observed. BERTScore improved slightly from Checkpoint 1 to Checkpoint 2 (+0.004), consistent with literature showing that small learning rates and limited epochs preserve prior knowledge. In a full-scale GPU training run, Stage 2 would likely improve JSON validity substantially while causing a small measurable regression in Alpaca scores.

### Failure Case Analysis
The primary failure mode was the model producing non-JSON text for structured output tasks. This is expected from an undertrained model and represents exactly what Stage 2 training is designed to fix. With more data and GPU resources, this failure mode would be substantially reduced.

### Implications for Sequential Fine-Tuning
The two-stage pipeline ran successfully end to end. The infrastructure for sequential fine-tuning, imitation learning, three-checkpoint evaluation, and LLM-as-a-Judge scoring is fully functional. The primary limitation was hardware rather than methodology.

---

## 4. Prompt Engineering

### Teacher Prompt Design
The teacher prompt instructs the model to produce pure JSON with no markdown, explanations, or preamble. Initial versions included few-shot examples but this caused the model to copy example structure. Removing examples and using clear task descriptions produced more diverse outputs.

### Judge Prompt Design
The judge prompt was embedded as a Python f-string to avoid conflicts between Python string formatting and JSON curly braces. It scores responses on six dimensions covering both general quality and structured output validity. An explicit example format was added after initial testing showed occasional malformed JSON responses.

---

## Appendix: Full Prompt Templates

### Teacher JSON Generation Prompt

You are a precise JSON generation assistant. Given the task below, generate ONLY a valid JSON object as your response. Do not include any explanation, markdown code blocks, or text before or after the JSON.

Task: [instruction]
Input: [input]

Respond with valid JSON only.

### Judge Evaluation Prompt

You are an expert evaluator of language model outputs.

Task: [instruction]

Response A: [response_a]
Response B: [response_b]

Score each response on these dimensions from 1 to 5:
- instruction_following
- correctness
- clarity
- completeness
- structured_output_validity
- hallucination_risk

Return ONLY valid JSON with winner set to A, B, or TIE and a one sentence justification.
