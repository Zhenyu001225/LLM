import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
in_text = "Zhenyu is a student in RPI"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198]) # line break symbol
out_token = None
kvcache = None
out_text = in_text
i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, kvcache = model(in_tokens, past_key_values=kvcache) # add parameter 'past_key_values'
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = out_token # don't concat 
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1
        out_text += text
print(f' Input: {in_text}')
print(f'Output: {out_text}')