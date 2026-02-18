import torch
import time
from transformers import AutoTokenizer
from Qwen3 import Qwen3ForCausalLM
from sampler import Sampler
import os


def warmup(model, tokenizer, warmup_steps=3, max_length=50):
    """
    Warmup function to prepare model for inference
    """
    print(f"Warming up model with {warmup_steps} steps...")
    
    for step in range(warmup_steps):
        input_ids = torch.randint(0, len(tokenizer), (1, 10))
        warmup_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        inference(model, tokenizer, warmup_prompt, max_length=max_length, temperature=0.7)
    
    print("Warmup completed!")


def preprocess(model, tokenizer, prompts):
    all_input_ids = []
    cu_seqlens = [0]
    all_positions = []
    device = next(model.parameters()).device

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[0]

        cu_seqlens.append(cu_seqlens[-1] + seq_len)
        all_input_ids.append(input_ids)
        all_positions.append(torch.arange(seq_len, device=device))
        
    cu_seqs = torch.cat(all_input_ids).unsqueeze(0)
    cu_positions = torch.cat(all_positions).unsqueeze(0)
    cu_seqlens = torch.tensor(cu_seqlens, device=device)

    return cu_seqs, cu_seqlens, cu_positions


def inference(model, tokenizer, prompts, max_length=50, temperature=0.7):
    device = next(model.parameters()).device
    max_token_id = len(tokenizer) - 1
    sampler = Sampler(temperature=temperature)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151645

    batch_size = len(prompts)
    finished = [False] * batch_size
    generated_tokens = [0] * batch_size

    step = 0
    while not all(finished) and step < max_length:
        cu_seqs, cu_seqlens, cu_positions = preprocess(model, tokenizer, prompts)
        hidden_states = model(cu_seqs, cu_positions, cu_seqlens)
        logits = model.compute_logits(hidden_states)

        last_logits = logits[:, cu_seqlens[1:] - 1, :]
        last_logits = last_logits.squeeze(0)
        next_token = sampler.sample(last_logits)
        next_token = next_token.tolist()
        
        for j in range(len(next_token)):
            if not finished[j]:
                out = tokenizer.decode(next_token[j], skip_special_tokens=False)
                prompts[j] += out
                generated_tokens[j] += 1
                
                if next_token[j] == eos_token_id or generated_tokens[j] >= max_length:
                    finished[j] = True
        
        step += 1
    
    for prompt in prompts:
        print(prompt)
    # print(prompts)
    return prompts

def main():
    device = torch.device("cuda:1")
    model_path = os.path.expanduser("~/huggingface/Qwen/Qwen3-0.6B/")
    model = Qwen3ForCausalLM.from_pretrained(model_path)
    model = model.to(device).half()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # warmup(model, tokenizer)

    test_cases = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a short joke.",
        "How to make a cup of coffee?"
    ]

    inference(model, tokenizer, test_cases)

    # for i, prompt in enumerate(test_cases, 1):
    #     print(f"\n{'-'*60}")
    #     print(f"Test Case {i}/{len(test_cases)}")
    #     print(f"Prompt: {prompt}")
        
    #     result = inference(model, tokenizer, prompt, max_length=100, temperature=0.7)
        
    #     print(f"Output: {result['output']}")
    #     print(f"TTFT: {result['ttft'] * 1000:.2f} ms")
    #     print(f"Total time: {result['total_time']:.4f} s")
    #     print(f"Generated tokens: {result['total_generated']}")
    #     print(f"Throughput: {result['throughput']:.2f} tokens/s")
        
    #     all_metrics.append(result)
    #     print(f"{'-'*60}")
        



if __name__ == "__main__":
    main()
