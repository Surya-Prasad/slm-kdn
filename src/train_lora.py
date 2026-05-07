import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from utils import load_config, read_jsonl


def main(args):
    cfg = load_config(args.config)
    tcfg = cfg['training']
    rows = read_jsonl(f"{cfg['data']['output_dir']}/train_{cfg['prompt']['mode']}.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(tcfg['base_model'])
    tokenizer.pad_token = tokenizer.eos_token

    bnb = None
    if tcfg.get('use_4bit', True):
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(tcfg['base_model'], quantization_config=bnb, device_map='auto')
    lora = LoraConfig(r=tcfg['lora_r'], lora_alpha=tcfg['lora_alpha'], lora_dropout=tcfg['lora_dropout'], task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)

    def tok(ex):
        full = tokenizer(ex['text'], truncation=True, max_length=tcfg['max_seq_len'])
        prompt = tokenizer(ex['prompt'], truncation=True, max_length=tcfg['max_seq_len'])
        labels = full['input_ids'][:]
        for i in range(min(len(prompt['input_ids']), len(labels))):
            labels[i] = -100
        full['labels'] = labels
        return full

    ds = Dataset.from_list(rows).map(tok)
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
    trainer = Trainer(model=model, args=args_t, train_dataset=ds)
    trainer.train()
    model.save_pretrained(tcfg['output_dir'])
    tokenizer.save_pretrained(tcfg['output_dir'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    main(p.parse_args())
