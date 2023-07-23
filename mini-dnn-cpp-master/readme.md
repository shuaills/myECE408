# mini-dnn-cpp
**mini-dnn-cpp** is a C++ demo of deep neural networks. It is implemented purely in C++, whose only dependency, Eigen, is header-only. 

## Usage
Download and unzip [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in `mini-dnn-cpp/data/mnist/`.

```shell
mkdir build
cd build
cmake ..
make
```

Run `./train` first to get model parameters.
Then run './inference' to test your CUDA code.

Result: 
simple neural network with 3 FC layers can obtain `0.86+` accuracy on Fashion MNIST testset.