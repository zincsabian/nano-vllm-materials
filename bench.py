import torch
import time
from transformers import AutoTokenizer
from Qwen3 import Qwen3ForCausalLM
from sampler import Sampler
import os


def inference(model, tokenizer, prompt, max_length=50, temperature=0.7):
    """
    Simple inference function for Qwen3 model with metrics
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    
    # Move input_ids to the same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Get maximum token ID that tokenizer can decode
    max_token_id = len(tokenizer) - 1
    
    # Initialize generated tokens
    generated_ids = input_ids
    
    # Initialize sampler
    sampler = Sampler(temperature=temperature)
    
    # Generate tokens with timing
    start_time = time.perf_counter()
    ttft = None
    
    for step in range(max_length - input_ids.shape[1]):
        # Get current positions
        current_positions = torch.arange(0, generated_ids.shape[1], device=generated_ids.device)
        current_positions = current_positions.unsqueeze(0)  # (1, seq_len)
        
        # Forward pass
        hidden_states = model(generated_ids, current_positions)
        
        # Compute logits
        logits = model.compute_logits(hidden_states)
        
        # Get logits for last token
        last_logits = logits[:, -1, :]
        
        # Mask out tokens beyond tokenizer's vocab size
        mask = torch.ones_like(last_logits)
        mask[:, max_token_id + 1:] = -float('inf')
        last_logits = last_logits + mask
        
        # Sample next token
        next_token = sampler.sample(last_logits)
        
        # Record TTFT (time to first token)
        if ttft is None:
            ttft = time.perf_counter() - start_time
        
        # Append to generated tokens
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Check for EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    total_time = time.perf_counter() - start_time
    total_generated = generated_ids.shape[1] - prompt_len
    throughput = total_generated / total_time if total_time > 0 else 0
    
    # Decode generated tokens
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return {
        "output": output,
        "ttft": ttft,
        "total_time": total_time,
        "total_generated": total_generated,
        "throughput": throughput
    }


def main():
    """
    Main function to load model and run inference
    """
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/huggingface/Qwen/Qwen3-0.6B/"), help="Path to pretrained model")
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = Qwen3ForCausalLM.from_pretrained(args.model_path)
    
    # Move model to specified device
    device = torch.device(args.device)
    model = model.to(device)
    print(f"Model loaded successfully on {device}!")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Tokenizer loaded successfully! Vocab size: {tokenizer.vocab_size}, Len: {len(tokenizer)}")
    
    # Test cases
    test_cases = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a short joke.",
        "How to make a cup of coffee?"
    ]
    
    all_metrics = []
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"Prompt: {prompt}")
        
        result = inference(model, tokenizer, prompt, max_length=100, temperature=0.7)
        
        print(f"Output: {result['output']}")
        print(f"TTFT: {result['ttft'] * 1000:.2f} ms")
        print(f"Total time: {result['total_time']:.4f} s")
        print(f"Generated tokens: {result['total_generated']}")
        print(f"Throughput: {result['throughput']:.2f} tokens/s")
        
        all_metrics.append(result)
        print(f"{'='*60}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    avg_ttft = sum(m['ttft'] for m in all_metrics) / len(all_metrics)
    avg_throughput = sum(m['throughput'] for m in all_metrics) / len(all_metrics)
    total_tokens = sum(m['total_generated'] for m in all_metrics)
    total_time = sum(m['total_time'] for m in all_metrics)
    overall_throughput = total_tokens / total_time if total_time > 0 else 0
    
    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/s")
    print(f"Overall Throughput: {overall_throughput:.2f} tokens/s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.4f} s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
