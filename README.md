This branch is only for implement a Qwen3 Model.

We need to download model from huggingface or modelscope.

## Startup

1. Download model from huggingface or ModelScope(faster in Chinese Mainland)

2. Modify the default model path in bcench.py, or just parse the arguments to bcench.py

## Current Status(Branch)

1. Qwen3Model
   1. Implemented Qwen3 Model without any optimize
2. contineous_batching
   1. Implemented contineous batching, and cu_seqlens optimize
   2. Implementing cu_seqlens kernel
3. kv_cache
   1. Implementing

## TODO

~~1. load checkpoint and model config~~

~~2. using torch.device('cuda') to do some model infer~~

3. implement basic kvcache to do PD desperate


## What should we do next (optimize)

1. FlashAttention(branch1)

2. PagedAttention(branch2)
   
2.1 KVCache, PD desperate

2.2 PagedAttention

3. torch.distributed(branch3)
    like TP, SP, CP(may not), EP(other model)

4. SpecDecoding(branch4)
