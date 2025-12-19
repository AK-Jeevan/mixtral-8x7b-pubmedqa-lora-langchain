# Mixtral 8x7B Fine-Tuning on PubMedQA  
### Biomedical QA with LoRA & LangChain

This repository demonstrates how to **fine-tune the Mixtral 8x7B-Instruct model**
using **LoRA** on the **PubMedQA biomedical question-answering dataset**, and
integrate the fine-tuned model into **LangChain** for structured biomedical reasoning.

The focus is on **domain adaptation** and **educational clarity**, not benchmark scores.

---

## 🚀 What This Project Covers

- Loading Mixtral 8x7B-Instruct with 4-bit quantization
- Applying LoRA for parameter-efficient fine-tuning
- Formatting PubMedQA (question + context → answer)
- Fine-tuning with Hugging Face Trainer
- Saving biomedical-adapted LoRA weights
- Custom LangChain LLM wrapper
- Chained prompting:
  - Explain biomedical answers
  - Generate quiz questions for reinforcement

---

## 🧠 Why PubMedQA?

- Real biomedical QA dataset
- Tests reasoning over **scientific context**
- Demonstrates domain-specific LLM adaptation
- Useful for healthcare & research-focused applications

---

## 📦 Tech Stack

- **Model:** Mixtral 8x7B-Instruct (MoE)
- **Framework:** PyTorch
- **Fine-tuning:** Hugging Face Transformers + PEFT (LoRA)
- **Dataset:** PubMedQA
- **Chaining:** LangChain
- **Optimization:** 4-bit quantization, gradient accumulation

---

## 🔍 Key Concepts Demonstrated

- Domain-specific LLM fine-tuning
- Biomedical question answering
- Instruction-style prompt formatting
- LoRA for large language models
- LangChain sequential chains
- Reasoning + explanation pipelines

---

## 🎯 Who This Is For

- ML Engineers working on LLM adaptation
- AI engineers in healthcare / biomedical NLP
- Developers using LangChain with custom models
- Researchers exploring MoE architectures

---

## 📜 License
MIT License
