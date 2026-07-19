// Fused LayerNorm + Linear kernel for MLX
// Each threadgroup handles one (batch, seq) position.
// Within that position: all threads cooperate on mean/var then each
// thread writes one d_out element.

#include <metal_stdlib>
using namespace metal;

constant float LN_EPS = 1e-5f;

template <typename T>
kernel void fused_ln_linear(
    device const T*  x          [[buffer(0)]],
    device const T*  w          [[buffer(1)]],
    device const T*  gamma      [[buffer(2)]],
    device const T*  beta       [[buffer(3)]],
    device const T*  bias       [[buffer(4)]],
    device       T*  output     [[buffer(5)]],
    constant    int& d_model    [[buffer(6)]],
    constant    int& d_out      [[buffer(7)]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]],
    uint  thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint  threads_per_threadgroup [[threads_per_threadgroup]]
) {
    int pos = threadgroup_position.x;
    int tid = thread_position_in_threadgroup;
    int TPG = threads_per_threadgroup;
    
    // ── Step 1: mean + var over d_model ──
    float sum = 0.0f, sum_sq = 0.0f;
    int base = pos * d_model;
    for (int i = tid; i < d_model; i += TPG) {
        float v = float(x[base + i]);
        sum += v;
        sum_sq += v * v;
    }
    
    // Threadgroup reduction (tree)
    // Use shared memory for the reduction
    threadgroup float tg_mem[256];
    
    // Reduce sum
    tg_mem[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int off = TPG >> 1; off > 0; off >>= 1) {
        if (tid < off) tg_mem[tid] += tg_mem[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = tg_mem[0];
    
    // Reduce sum_sq
    tg_mem[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int off = TPG >> 1; off > 0; off >>= 1) {
        if (tid < off) tg_mem[tid] += tg_mem[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum_sq = tg_mem[0];
    
    float inv_d = 1.0f / float(d_model);
    float mean_v = total_sum * inv_d;
    float var_v = total_sum_sq * inv_d - mean_v * mean_v;
    float inv_std = 1.0f / sqrt(max(var_v, LN_EPS));
    
    // ── Step 2: each thread computes one output row ──
    // out_idx = tid (if tid < d_out), else skip
    if (tid >= d_out) return;
    
    float acc = 0.0f;
    int w_off = tid * d_model;  // row tid of weight matrix
    for (int k = 0; k < d_model; k++) {
        float x_k = float(x[base + k]);
        float n_k = (x_k - mean_v) * inv_std * float(gamma[k]) + float(beta[k]);
        acc += n_k * float(w[w_off + k]);
    }
    
    if (bias) acc += float(bias[tid]);
    output[pos * d_out + tid] = T(acc);
}
