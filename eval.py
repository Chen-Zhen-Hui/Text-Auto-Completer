from bert_score import BERTScorer
from datasets import load_dataset
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_path = "./gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path, pad_token='<|endoftext|>')
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# metric
bertscorer = BERTScorer(lang="en", device=device)

def process_function(examples):
    processed = {'input_ids': [], 'attention_mask': [], 'text': [], 'split_point': []}
    for text in examples['content']:
        tokenized = tokenizer(
            text, 
            truncation=True, 
            max_length=1024,
            return_attention_mask=True,
        )
            
        split = random.randint(len(tokenized['input_ids'])//5, len(tokenized['input_ids'])//5*4)
        
        # Store all necessary information
        processed['input_ids'].append(tokenized['input_ids'])
        processed['attention_mask'].append(tokenized['attention_mask'])
        processed['text'].append(text)
        processed['split_point'].append(split)
    return processed

def evaluate_generation(model, tokenizer, test_data):
    predictions, references = [], []
    
    for example in test_data.select(range(100)):
        full_text = example['text']
        split_point = example['split_point']
        
        inputs = torch.tensor(example['input_ids'][:split_point]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example['attention_mask'][:split_point]).unsqueeze(0).to(model.device)
        outputs = model.generate(
            inputs,
            max_length=split_point + 50,  # Generate 50 new tokens
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask
        )
        prediction = tokenizer.decode(outputs[0][split_point:], skip_special_tokens=True)
        
        # Get reference text
        reference = full_text[split_point:split_point+50]  # Control reference length
        
        predictions.append(prediction)
        references.append(reference)
    
    # Compute BERTScore
    P, R, F1 = bertscorer.score(predictions, references)
    bertscore = F1.mean().item()  # We are interested in the F1 score

    return {
        "BERTScore": bertscore
    }

dataset = load_dataset("json", data_dir='./data')

# Apply data processing
processed_dataset = dataset.map(
    process_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
)

processed_dataset = processed_dataset.filter(lambda x: len(x['input_ids']) > 0)

# Split into training/validation
split_dataset = processed_dataset['train'].train_test_split(test_size=0.1, seed=42)

# Use the evaluation function
metrics = evaluate_generation(model, tokenizer, split_dataset['test'])
print(f"Evaluation results: {metrics}")
