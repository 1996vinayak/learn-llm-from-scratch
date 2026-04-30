"""
Quantization — Shrink Models 4x
================================
Normal: FP32 (32 bits per parameter) or FP16 (16 bits)
Quantized: INT8 (8 bits) or INT4 (4 bits)

A 7B model:
  FP32: 28 GB
  FP16: 14 GB
  INT8:  7 GB
  INT4:  3.5 GB  <-- fits on a laptop!

How it works (simplified):
  1. Find the range of weights: [min, max]
  2. Map that range to integers: [0, 255] for INT8
  3. Store the scale factor to convert back

Quality loss is surprisingly small!
"""

import torch


def quantize_tensor(tensor, bits=8):
    """Quantize a float tensor to N-bit integers."""
    qmin = 0
    qmax = 2**bits - 1
    
    # Find range
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Compute scale and zero point
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    zero_point = torch.clamp(torch.round(zero_point), qmin, qmax)
    
    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax).to(torch.uint8)
    
    return quantized, scale, zero_point


def dequantize_tensor(quantized, scale, zero_point):
    """Convert quantized integers back to floats."""
    return (quantized.float() - zero_point) * scale


def demo_quantization():
    print("=" * 60)
    print("Quantization Demo")
    print("=" * 60)
    
    # Create a sample weight tensor
    weights = torch.randn(1000, 1000)
    
    for bits in [8, 4]:
        q, scale, zp = quantize_tensor(weights, bits)
        deq = dequantize_tensor(q, scale, zp)
        error = (weights - deq).abs().mean()
        
        orig_size = weights.numel() * 4  # FP32 = 4 bytes
        quant_size = weights.numel() * bits / 8
        
        print(f"\n  INT{bits} Quantization:")
        print(f"    Original size:  {orig_size/1024:.1f} KB")
        print(f"    Quantized size: {quant_size/1024:.1f} KB")
        print(f"    Compression:    {orig_size/quant_size:.1f}x")
        print(f"    Mean error:     {error:.6f}")


if __name__ == "__main__":
    demo_quantization()
