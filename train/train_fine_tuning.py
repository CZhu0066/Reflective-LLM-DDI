
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

MICRO_BATCH_SIZE = 2 
BATCH_SIZE = 64      
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 5
LEARNING_RATE = 2e-4 
MAX_LENGTH = 2048
LORA_R = 16        
LORA_ALPHA = 32      # Typically, alpha = 2 * rank or alpha = rank
LORA_DROPOUT = 0.05
MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
OUTPUT_DIR = "/root/autodl-tmp/Result"

class QwenTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Padding Token，Qwen typically uses EOS as the pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2" # open Flash Attention 2
        )

    def setup_peft(self):
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return self.model

def process_func(example, tokenizer):
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = str(example.get("output", "")) if example.get("output") is not None else ""
    
    if user_input:
        full_user_content = f"Instruction:\n{instruction}\n\nInput:\n{user_input}"
    else:
        full_user_content = f"Instruction:\n{instruction}"
    prompt_messages = [{"role": "user", "content": full_user_content}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=True, 
        add_generation_prompt=True
    )
    prompt_length = len(prompt_ids)

    messages = [
        {"role": "user", "content": full_user_content},
        {"role": "assistant", "content": output}
    ]
    full_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        max_length=MAX_LENGTH, 
        truncation=True
    )
    
    labels = list(full_ids)

    if len(full_ids) >= prompt_length:
        for i in range(prompt_length):
            labels[i] = -100
    else:
        labels = [-100] * len(labels)

    return {
        "input_ids": full_ids,
        "labels": labels
    }

trainer_helper = QwenTrainer(MODEL_PATH)
tokenizer = trainer_helper.tokenizer
model = trainer_helper.setup_peft()

data_files = {
    "train": "/root/autodl-tmp/data/train.json",
    "validation": "/root/autodl-tmp/data/dev.json"
}

try:
    raw_datasets = load_dataset("json", data_files=data_files)
except:
    raw_datasets = load_dataset("json", data_files={"train": "/root/autodl-tmp/data/train.json"})

tokenized_datasets = raw_datasets.map(
    lambda x: process_func(x, tokenizer),
    batched=False,
    remove_columns=raw_datasets["train"].column_names
)

# Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps" if "validation" in raw_datasets else "no",
    eval_steps=100,
    optim="paged_adamw_32bit",
    bf16=True,
    fp16=False, 
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    group_by_length=True,
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
    tokenizer=tokenizer, 
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
