import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from utils import load_config, read_jsonl


class CommandOnlyDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        pad_id = self.tokenizer.pad_token_id
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main(args):
    cfg = load_config(args.config)
    tcfg = cfg['training']
    rows = read_jsonl(f"{cfg['data']['output_dir']}/train_{cfg['prompt']['mode']}.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(tcfg['base_model'])
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    bnb = None
    if tcfg.get('use_4bit', True):
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(tcfg['base_model'], quantization_config=bnb, device_map='auto')
    lora = LoraConfig(r=tcfg['lora_r'], lora_alpha=tcfg['lora_alpha'], lora_dropout=tcfg['lora_dropout'], task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)

    def tok(ex):
        prompt_ids = tokenizer(ex['prompt'], add_special_tokens=True)['input_ids']
        command_ids = tokenizer(
            " " + ex['target_command'].strip(),
            add_special_tokens=False,
        )['input_ids'] + [tokenizer.eos_token_id]

        available_for_command = max(tcfg['max_seq_len'] - len(prompt_ids), 1)
        command_ids = command_ids[:available_for_command]
        input_ids = (prompt_ids + command_ids)[:tcfg['max_seq_len']]
        labels = [-100] * len(prompt_ids) + command_ids
        labels = labels[:len(input_ids)]

        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': labels,
        }

    ds = Dataset.from_list(rows).map(tok, remove_columns=list(rows[0].keys()))
    args_t = TrainingArguments(
        output_dir=tcfg['output_dir'],
        per_device_train_batch_size=tcfg['per_device_batch_size'],
        gradient_accumulation_steps=tcfg['gradient_accumulation_steps'],
        learning_rate=tcfg['learning_rate'],
        num_train_epochs=tcfg['num_train_epochs'],
        logging_steps=10,
        save_strategy='epoch',
        bf16=tcfg.get('bf16', False),
    )
    trainer = Trainer(
        model=model,
        args=args_t,
        train_dataset=ds,
        data_collator=CommandOnlyDataCollator(
            tokenizer,
            pad_to_multiple_of=tcfg.get('pad_to_multiple_of', 8),
        ),
    )
    trainer.train()
    model.save_pretrained(tcfg['output_dir'])
    tokenizer.save_pretrained(tcfg['output_dir'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    main(p.parse_args())
