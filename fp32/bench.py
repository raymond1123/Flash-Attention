import math
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn_fp32(Q, K, V):
    ### forward ###
    softmax_scale = 1.0 / math.sqrt(K.size(-1))
    # (B, n_head, seq_len, dim)@(B, n_head, dim, seq_len) = (B, n_head, seq_len, seq_len)
    att = (Q @ K.transpose(-2, -1)) * softmax_scale

    # Softmax attention
    P = F.softmax(att, dim=-1)

    # (B, n_head, seq_len, seq_len)@(B, n_head, seq_len, dim) = (B, n_head, seq_len, dim)
    y = P @ V

    ### backward
    #dO = torch.ones_like(y).cuda()
    #dP = dO@(V.transpose(-2,-1))
    #dS = P * (dP - torch.sum(P * dP, dim=-1, keepdim=True))

    #dV = P.transpose(-2,-1)@dO # (batch_size, n_head, seq_len, head_embd)
    #dQ = (dS@K)*softmax_scale
    #dK = (dS.transpose(-2,-1))@Q*softmax_scale

    #return y, dQ, dK, dV
    return y


if __name__ == '__main__':
    # Load the CUDA kernel as a python module
    #minimal_attn = load(name='minimal_attn', sources=['main.cpp', './fa_1/flash_base.cu'], extra_cuda_cflags=['-O2'])
    minimal_attn = load(name='minimal_attn', sources=['main.cpp', './fa_2/flash_base.cu'], extra_cuda_cflags=['-O2'])

    # Use small model params, otherwise slower than manual attention. See caveats in README.
    batch_size = 2
    n_head = 12
    seq_len = 128
    head_embd = 32

    dropout_p = 0.0
    mask_p = 0.0

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda().requires_grad_(True)
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda().requires_grad_(True)
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda().requires_grad_(True)

    l = torch.zeros(batch_size, n_head, seq_len).cuda()
    m = torch.full((batch_size, n_head, seq_len), float('-inf')).cuda()

    print('=== profiling manual attention forward ===')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        manual_result_fp32 = manual_attn_fp32(q, k, v)
    prof_data = prof.key_averages()
    total_cuda_time = sum(item.cuda_time_total for item in prof_data)
    print(f"profiling manual attention forward: {total_cuda_time=} μs\n")
    #print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    print('=== profiling minimal flash attention forward === ')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        l,m,O= minimal_attn.forward(q, k, v, l, m)
    prof_data = prof.key_averages()
    total_cuda_time = sum(item.cuda_time_total for item in prof_data)
    print(f"profiling minimal attention forward: {total_cuda_time=} μs\n")
    #print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    manual_result = manual_result_fp32.detach().cpu().numpy()
    minimal_result = O.detach().cpu().numpy()
    err = np.max(abs(minimal_result - manual_result))
    print(f'{err=}')
    if err < 1e-5:
        print(f"fp32 forward: {GREEN}PASS{RESET}")
    else:
        print(f"fp32 forward: {RED}FAILED{RESET}")

    print('=== profiling manual attention backward ===')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        torch.sum(manual_result_fp32).backward()
    prof_data = prof.key_averages()
    total_cuda_time = sum(item.cuda_time_total for item in prof_data)
    print(f"profiling manual attention backward: {total_cuda_time=} μs\n")
    #print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    dO = torch.ones(batch_size, n_head, seq_len, head_embd).cuda()
    print('=== profiling minimal flash attention backward === ')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        minimal_q_grad, minimal_k_grad, minimal_v_grad = minimal_attn.backward(q, k, v, O, dO, l, m)
    prof_data = prof.key_averages()
    total_cuda_time = sum(item.cuda_time_total for item in prof_data)
    print(f"profiling minimal attention backward: {total_cuda_time=} μs\n")
    #print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    tch_manual_q_grad = q.grad.detach().cpu().numpy()
    tch_manual_k_grad = k.grad.detach().cpu().numpy()
    tch_manual_v_grad = v.grad.detach().cpu().numpy()

    minimal_q_grad = minimal_q_grad.detach().cpu().numpy()
    minimal_k_grad = minimal_k_grad.detach().cpu().numpy()
    minimal_v_grad = minimal_v_grad.detach().cpu().numpy()

    q_err = np.max(abs(tch_manual_q_grad- minimal_q_grad))
    k_err = np.max(abs(tch_manual_k_grad- minimal_k_grad))
    v_err = np.max(abs(tch_manual_v_grad- minimal_v_grad))
    
    print(f'{q_err=}, {k_err=}, {v_err=}')
    if (q_err < 1e-5) and (k_err < 1e-5) and (v_err < 1e-5):
        print(f"f32 backward: {GREEN}PASS{RESET}")
    else:
        print(f"fp32 backward: {RED}FAILED{RESET}")

    print()

