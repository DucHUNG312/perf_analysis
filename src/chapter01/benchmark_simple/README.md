# PyTorch Performance Optimization: Batch Fusion + FP16

## Overview
This benchmark demonstrates neural network training optimizations using PyTorch, achieving **~10x speedup** through batch fusion and mixed precision.

## Files
- `baseline_performance.py` - Original implementation with small batches
- `optimized_performance.py` - Optimized with batch fusion + FP16

## Key Optimizations

### 1. **FP16 Mixed Precision** ([optimized_performance.py:63-65](../optimized_performance.py#L63-L65))
```python
self.model = self.model.half()  # Convert to FP16
dtype = torch.float16
```
### 2. **Pre-fused Batches** ([optimized_performance.py:106-113](../optimized_performance.py#L106-L113))
```python
# Concatenate 8 microbatches into larger batches
for start in range(0, len(self.microbatches), self.fusion):
    batch = torch.cat(self.microbatches[start: start + self.fusion], dim=0)
    self._fused_batches.append(batch)
```
- **Benefit**: Reduces kernel launch overhead, better GPU utilization
- **Impact**: 8 small forward/backward passes → 1 large pass

### 3. **Simplified Training Loop** ([optimized_performance.py:120-125](../optimized_performance.py#L120-L125))
```python
# Baseline: nested loops with gradient accumulation
for group in groups:
    for data in group:
        loss = model(data)
        (loss / group_size).backward()  # Accumulate gradients

# Optimized: single pass
logits = model(fused_batch)
loss.backward()  # No accumulation needed
```
- **Benefit**: Eliminates redundant backward passes
- **Impact**: Fewer backward passes

## Why It Works

**Problem**: Small batches → GPU underutilized → kernel launch overhead dominates

**Solution**:
1. Fuse small batches into larger batches (32 → 256)
2. Use FP16 to double effective bandwidth
3. Single forward/backward instead of accumulation loop

**Result**: GPU runs at full capacity with minimal overhead

## Run Benchmarks
```bash
# Baseline
python src/chapter01/benchmark_simple/baseline_performance.py

# Optimized
python src/chapter01/benchmark_simple/optimized_performance.py
```
