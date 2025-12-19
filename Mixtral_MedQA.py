# =========================================================
# Mixtral 8x7B + LoRA Fine-Tuning + PubMedQA Dataset + LangChain Chaining
# =========================================================
#
# Description:
#   This script demonstrates how to fine-tune the Mixtral 8x7B-Instruct model
#   using LoRA (Low-Rank Adaptation) on the PubMedQA dataset, and then integrate
#   the fine-tuned model into LangChain for biomedical question answering and
#   structured prompting workflows.
#
# Workflow:
#   1. Load Mixtral 8x7B-Instruct model and tokenizer from Hugging Face.
#   2. Apply LoRA adapters for parameter-efficient fine-tuning.
#   3. Load and preprocess the PubMedQA dataset (question, context, answer).
#   4. Fine-tune the model using Hugging Face Trainer.
#   5. Wrap the model in a custom LangChain LLM class for inference.
#   6. Use LangChain PromptTemplate and LLMChain to build chained workflows
#      (e.g., explain biomedical answers, then generate quiz questions).
#
# Output:
#   - Fine-tuned model saved in ./mixtral-pubmedqa-lora
#   - Example LangChain chaining pipeline that explains biomedical answers
#     and generates quiz questions for deeper understanding.
#
# =========================================================


import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from typing import Optional, List

# ---------------------------------------------------------
# 1. Load Mixtral 8x7B
# ---------------------------------------------------------
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

# ---------------------------------------------------------
# 2. Apply LoRA
# ---------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------
# 3. Load REAL dataset: PubMedQA
# ---------------------------------------------------------
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# ---------------------------------------------------------
# 4. Format prompts for Mixtral
# ---------------------------------------------------------
MAX_LEN = 512

def format_example(example):
    question = example["question"]
    context = example["context"]
    answer = example["final_decision"]  # yes/no/maybe

    user_prompt = f"Question: {question}\nContext: {context}"
    text = (
        "<s>[INST] "
        f"{user_prompt} "
        "[/INST] "
        f"{answer}"
    )

    return tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)

# ---------------------------------------------------------
# 5. Trainer setup
# ---------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./mixtral-pubmedqa-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# ---------------------------------------------------------
# 6. Inference helper
# ---------------------------------------------------------
def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------
# 7. LangChain HF wrapper
# ---------------------------------------------------------
class HFLLM(LLM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if stop:
            for s in stop:
                text = text.split(s)[0]
        return text

    @property
    def _identifying_params(self):
        return {"model": "Mixtral-8x7B-PubMedQA-LoRA"}

    @property
    def _llm_type(self):
        return "custom_hf"

llm = HFLLM(model, tokenizer)

# ---------------------------------------------------------
# 8. LangChain chaining
# ---------------------------------------------------------
explain_prompt = PromptTemplate.from_template(
    "<s>[INST] You are a biomedical tutor.\nExplain the answer to: {concept} [/INST]"
)

quiz_prompt = PromptTemplate.from_template(
    "<s>[INST] Create 3 quiz questions to test understanding of biomedical concept: {concept} [/INST]"
)

explain_chain = LLMChain(llm=llm, prompt=explain_prompt)
quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)

sequential_chain = SimpleSequentialChain(chains=[explain_chain, quiz_chain])

# ---------------------------------------------------------
# 9. Run the chain
# ---------------------------------------------------------
concept = "Are group 2 innate lymphoid cells increased in chronic rhinosinusitis?"
result = sequential_chain.run(concept)

print("Final Output:\n", result)
