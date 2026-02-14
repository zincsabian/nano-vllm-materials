import torch


class Sampler:
    """
    Token sampler for text generation
    """
    def __init__(self, temperature=0.7):
        self.temperature = temperature
    
    def topK(self, logits, k):
        """
        Apply top-k sampling
        """
        # Get top k logits
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Create mask for other tokens
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices, value=False)

        return logits.masked_fill(mask, float('-inf'))

    def sample(self, logits, topk = None):
        """
        Sample next token from logits
        """
        # Apply temperature
        if self.temperature > 0:
            logits = logits / self.temperature
        

        # Apply top-k sampling if specified
        if topk is not None:
            logits = self.topK(logits, topk) 
        
        # Compute probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sample from distribution
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        return next_token
