import random
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

model_path = "./gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 添加pad token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token
# 加载数据集
dataset = load_dataset("json", data_dir='./data')


# 数据处理函数
def process_function(examples):
    processed = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for text in examples['content']:
        # 分词并截断
        tokenized = tokenizer(text, truncation=True, max_length=1024)
        input_ids = tokenized['input_ids']

        if len(input_ids) < 2:
            continue

        # 随机分割点
        split = random.randint(len(input_ids)//5, len(input_ids)//5*4)

        # 创建标签（前部分忽略，后部分保留）
        labels = [-100] * split + input_ids[split:]

        processed['input_ids'].append(input_ids)
        processed['attention_mask'].append([1] * len(input_ids))
        processed['labels'].append(labels)
    return processed


# 应用数据处理
processed_dataset = dataset.map(
    process_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
)

# 过滤无效样本
processed_dataset = processed_dataset.filter(lambda x: len(x['input_ids']) > 0)

# 拆分训练/验证集
split_dataset = processed_dataset['train'].train_test_split(test_size=0.1,seed=42)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# 训练参数配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,    
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
