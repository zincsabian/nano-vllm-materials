import torch


class Sampler:
    """
    Token sampler for text generation
    """
    def __init__(self, temperature=0.7):
        self.temperature = temperature
    
    def sample(self, logits):
        """
        Sample next token from logits
        """
        # Apply temperature
        if self.temperature > 0:
            logits = logits / self.temperature
        
        # Compute probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sample from distribution
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        return next_token
