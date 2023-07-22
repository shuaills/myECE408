# myECE408

ECE408 is a great course offered by UIUC on deep learning. However, for students not enrolled at UIUC, it can be challenging to follow along with the labs and milestone projects which require access to specific computing resources. 

This project aims to make it easier to learn the material in ECE408 using your own local environment (GPU, system etc). It provides implementations of key assignments from the course, so you can get hands-on deep learning experience.


## Outcomes

By completing the notebooks in this project, you will:

- Gain practical experience with CUDA

Even without access to UIUC's specific infrastructure, you can get hands-on practice with the key concepts from ECE408. Let me know if any part of the course content is unclear or missing - I'm happy to add more notebooks to cover the full curriculum.

## Contents

- MP1 - Image Classification
- MP2 - Object Detection
- MP3 - GANs
- Milestone 1 - CPU Conv
- Milestone 2 - GPU Conv Baseline
## Usage

The labs and milestones are set up as standalone Python notebooks. To run them:

1. Clone this repository
2. Install dependencies
3. Run the notebooks, modifying as needed to use your local compute resources

### GPU Details

There is 1 device supporting CUDA

Device 0 name: NVIDIA GeForce RTX 4090

- Computational Capabilities: 8.9
- Maximum global memory size: 25756696576 
- Maximum constant memory size: 65536
- Maximum shared memory size per block: 49152
- Maximum block dimensions: 1024 x 1024 x 64
- Maximum grid dimensions: 2147483647 x 65535 x 65535
- Warp size: 32

This shows my system has 1 GPU (NVIDIA GeForce RTX 4090) available for CUDA computations, with 24GB of memory and compute capability 8.9.

The GPU can support up to 1024 x 1024 x 64 threads per block, and very large grid dimensions for parallel processing.

The output validates that my GPU setup is correctly configured for running CUDA programs and neural network training.