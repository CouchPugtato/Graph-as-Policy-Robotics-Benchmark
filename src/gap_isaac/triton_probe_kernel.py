"""One fixed-size Triton kernel, imported only by the explicit GPU probe."""

import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, output, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    values = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)
