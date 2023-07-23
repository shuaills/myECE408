# myECE408

ECE408 is a great course offered by UIUC on learning CUDA. However, for students not enrolled at UIUC, it can be challenging to follow along with the labs and milestone projects which require access to specific computing resources. 

This project aims to make it easier to learn the material in ECE408 using your own local environment (GPU, system etc). It provides implementations of key assignments from the course, so you can get hands-on CUDA experience.


## Outcomes

By completing this project, you will:

- Gain practical experience with CUDA

Even without access to UIUC's specific infrastructure, you can get hands-on practice with the key concepts from ECE408. 
Let me know if any part of the course content is unclear or missing - I'm happy to add more details to cover the full curriculum.

## Usage

This is a repository containing subfolders for individual ECE408 CUDA labs and milestone projects. 

To run them, follow these steps:

1. Clone this repository locally, either using git clone or by downloading the zip file from GitHub and extracting it.

2. Each lab and project has a CMake file for compilation. Go into the subfolder, create a build directory, and run the cmake .. command to generate the Makefile.

3. Use the make command to compile and generate the executable. 

Here are separate explanations for the milestone projects and labs in English:

### For milestone projects:

1. The project code is in mini-dnn/GPU_conv_baseline.cu. This is a baseline CUDA convolution layer implementation.

2. You need to modify this file and utilize learned CUDA optimization techniques to accelerate convolution computation, such as shared memory, loop unrolling, etc. 

3. You can incrementally optimize in steps, implementing one technique at a time. Test and record speedups.

4. The goal is to maximize convolution layer execution speed as much as possible, achieving peak performance. 

5. Commit your optimized code to GitHub for comparison with others' results.

### For labs:

1. Each lab has a separate folder containing code templates. 

2. You need to edit the specified files and complete the code according to lab requirements.

3. The labs cover various aspects of CUDA programming, such as memory, threads, etc.

4. Run the 'run_datasets' to verify correct implementation.


### MP0 results : GPU Details

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