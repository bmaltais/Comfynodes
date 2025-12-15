## 2024-05-21 - GPU->CPU transfers in batch processing loop
**Learning:** Iterating through image batches and using CPU-bound libraries like NumPy within the loop is a major performance bottleneck. Each iteration involves a costly data transfer from the GPU to the CPU (`.cpu().numpy()`) and back.
**Action:** Vectorize operations using PyTorch tensor functions (e.g., `torch.nn.functional.interpolate`) to keep all processing on the GPU. This avoids CPU-GPU data transfers and leverages parallel processing for the entire batch.
