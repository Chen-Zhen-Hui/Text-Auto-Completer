import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple
from TextAutoCompleter import TextAutoCompleter

if __name__ == "__main__":
    completer = TextAutoCompleter(model_name='./gpt2-finetuned')

    example_text = 'The quick brown fox jumps over the lazy dog. Suddenly,'
    cursor_position = len(example_text)

    full_text, completion = completer.generate_completion(
        text=example_text,
        cursor_pos=cursor_position,
        max_new_tokens=500,
        temperature=0.7
    )

    print("原始文本:", example_text)
    print("补全部分:", completion)
    print("完整结果:", full_text)
