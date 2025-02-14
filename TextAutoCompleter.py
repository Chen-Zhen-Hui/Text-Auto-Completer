import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple

class TextAutoCompleter:
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 模型最大上下文长度
        self.max_sequence_length = self.model.config.max_position_embeddings
        self.max_context_length = int(self.max_sequence_length * 0.8)

    def _truncate_context(self, text: str, cursor_pos: int) -> str:
        # 基于token截断
        prefix = text[:cursor_pos]
        tokens = self.tokenizer.tokenize(prefix)
        if len(tokens) > self.max_context_length:
            tokens = tokens[-self.max_context_length:]
        return self.tokenizer.convert_tokens_to_string(tokens)

    def _format_prompt(self, context: str) -> torch.Tensor:
        # 将上下文格式化为模型输入
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length
        ).to(self.device)
        return inputs

    def generate_completion(
            self,
            text: str,
            cursor_pos: int,
            max_new_tokens: int = 50,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 30
    ) -> Tuple[str, str]:
        """
        主生成函数
        返回补全后的完整文本和生成的补全部分
        """
        # 处理上下文
        context = self._truncate_context(text, cursor_pos)
        inputs = self._format_prompt(context)

        # 生成参数配置
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                num_return_sequences=1
            )

        # 解码并后处理
        full_generation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = full_generation[len(context):]
        final_text = text[:cursor_pos] + new_text
        return final_text, new_text
