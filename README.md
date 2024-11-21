# Flash-Attention

complish version1 & version2 in **naively** manner, both forward and backward

**Just for fun...**

## environment

* pytorch 
* cuda

```
├── fp16
│   ├── bench.py
│   ├── fa_1
│   │   └── flash_base.cu
│   ├── fa_2
│   │   └── flash_base.cu
│   └── main.cpp
├── fp32
│   ├── bench.py
│   ├── fa_1
│   │   └── flash_base.cu
│   ├── fa_2
│   │   └── flash_base.cu
│   └── main.cpp
└── README.md
```



## fp32 

### version 1

```
=== profiling manual attention forward ===
profiling manual attention forward: total_cuda_time=935441 μs

=== profiling minimal flash attention forward === 
Max shared memory: 49152, requested shared memory: 16384 
profiling minimal attention forward: total_cuda_time=129 μs

err=4.172325e-07
fp32 forward: PASS
=== profiling manual attention backward ===
profiling manual attention backward: total_cuda_time=2028 μs

=== profiling minimal flash attention backward === 
Max shared memory: 49152, requested shared memory: 45056 
profiling minimal attention backward: total_cuda_time=495 μs

q_err=4.7683716e-07, k_err=8.34465e-07, v_err=4.7683716e-07
f32 backward: PASS

```

### version 2

```
=== profiling manual attention forward ===
profiling manual attention forward: total_cuda_time=979501 μs

=== profiling minimal flash attention forward === 
Max shared memory: 49152, requested shared memory: 16384 
profiling minimal attention forward: total_cuda_time=137 μs

err=4.7683716e-07
fp32 forward: PASS
=== profiling manual attention backward ===
profiling manual attention backward: total_cuda_time=2218 μs

=== profiling minimal flash attention backward === 
Max shared memory: 49152, requested shared memory: 45056 
profiling minimal attention backward: total_cuda_time=557 μs

q_err=6.556511e-07, k_err=1.9073486e-06, v_err=5.9604645e-07
f32 backward: PASS
```



## fp16

### version 1

```
=== profiling manual attention forward ===
profiling manual attention forward: total_cuda_time=961884 μs

=== profiling minimal flash attention forward === 
Bc=64, Br=64
Max shared memory: 65536, requested shared memory: 65536 
profiling minimal attention forward: total_cuda_time=138 μs

err=0.009766
fp16 forward: PASS
=== profiling manual attention backward ===
profiling manual attention backward: total_cuda_time=2068 μs

=== profiling minimal flash attention backward === 
Max shared memory: 65536, requested shared memory: 34304 
profiling minimal attention backward: total_cuda_time=529 μs

q_err=0.009766, k_err=0.009766, v_err=0.006836
f16 backward: PASS
```

### version 2

```
=== profiling manual attention forward ===
profiling manual attention forward: total_cuda_time=931775 μs

=== profiling minimal flash attention forward === 
Bc=64, Br=64
allocate 66048 shared memory
profiling minimal attention forward: total_cuda_time=153 μs

err=0.01123
fp16 forward: PASS
=== profiling manual attention backward ===
profiling manual attention backward: total_cuda_time=2202 μs

=== profiling minimal flash attention backward === 
profiling minimal attention backward: total_cuda_time=633 μs

q_err=0.010254, k_err=0.01367, v_err=0.006836
f16 backward: PASS
```



NOTE:

both fp32 and fp16 version2 are slower than version1, this should be something wrong in coding...